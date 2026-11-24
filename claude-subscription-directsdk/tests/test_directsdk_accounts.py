"""#441: several Claude logins in one profile (one provider per account) and plan-limit failover between them.

Every turn here spawns a fake native that records the login it was started with (CLAUDE_CONFIG_DIR, the token);
the failover turns go through the real relay to a local peer that answers Claude's used-up-window 429 for the
accounts the test marks limited. No real login or Claude Code is used.
"""
import itertools
import json
import os
import shutil
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from conftest import PLUGIN_NAME, REPO_ROOT
from test_directsdk import FAKE

TOKEN = "CLAUDE_CODE_OAUTH_TOKEN"
REQUEST = dict(model="sonnet", messages=[{"role": "user", "content": "go"}])
TOOL_REQUEST = dict(REQUEST, tools=[{"type": "function", "function": {
    "name": "probe", "description": "TAIL", "parameters": {"type": "object", "properties": {}}}}])
LIMIT_BODY = (b'{"type":"error","error":{"type":"rate_limit_error",'
              b'"message":"This request would exceed your account\'s rate limit. Please try again later."}}')


def _write_settings(home, **settings):
    body = json.dumps({"plugins": {"enabled": [], "entries": {PLUGIN_NAME: {"settings": settings}}}})
    (home / "config.yaml").write_text(body, encoding="utf-8")  # JSON is YAML


def _discover(tmp_path, monkeypatch, **settings):
    """Install the plugin into a fresh HERMES_HOME with ``settings`` and run core's provider discovery."""
    import providers
    home = tmp_path / "hermes-home"
    shutil.copytree(REPO_ROOT, home / "plugins" / PLUGIN_NAME,
                    ignore=shutil.ignore_patterns(".git", "tests", "evals", "__pycache__"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    _write_settings(home, **settings)
    monkeypatch.setenv("CLAUDE_SUBSCRIPTION_DIRECTSDK_COMMAND", str(tmp_path / "no-such-claude"))
    for name in tuple(sys.modules):
        if name.startswith("_hermes_user_provider_"):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(providers, "_REGISTRY", {})
    monkeypatch.setattr(providers, "_ALIASES", {})
    monkeypatch.setattr(providers, "_PROVIDER_LIST_CACHE", None)
    monkeypatch.setattr(providers, "_discovered", False)
    providers._discover_providers()
    return home, providers


def _ours(providers):
    return sorted(p.name for p in providers.list_providers() if p.name.startswith(PLUGIN_NAME))


@pytest.fixture(autouse=True)
def fresh_limits(monkeypatch):
    import account_pool
    monkeypatch.setattr(account_pool, "_limits", {})
    monkeypatch.setattr(account_pool, "_turns", itertools.count())
    for key in ("ANTHROPIC_BASE_URL", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY",
                "CLAUDE_CONFIG_DIR", "CLAUDE_SUBSCRIPTION_DIRECTSDK_CONFIG_DIR", TOKEN, "ANTHROPIC_CUSTOM_HEADERS"):
        monkeypatch.delenv(key, raising=False)


# --- registration and settings ---------------------------------------------------------------------------------

def test_each_account_registers_its_own_provider(tmp_path, monkeypatch):
    main, work = str(tmp_path / "main-login"), str(tmp_path / "work-login")
    _, providers = _discover(tmp_path, monkeypatch, accounts={"main": main, "work": work})
    assert _ours(providers) == [PLUGIN_NAME, f"{PLUGIN_NAME}-main", f"{PLUGIN_NAME}-work"]
    for name in ("main", "work"):
        account = providers.get_provider_profile(f"{PLUGIN_NAME}-{name}")
        assert account.display_name == f"Claude Subscription · {name}"
        assert account.account == name
        assert account.base_url == f"process://{PLUGIN_NAME}-{name}"
        assert account.auth_type == "external_process"
        # One replay carrier for all accounts, so a conversation survives a switch or a failover.
        assert account.native_reasoning_details_type == providers.get_provider_profile(PLUGIN_NAME).native_reasoning_details_type
    assert providers.get_provider_profile(PLUGIN_NAME).account is None


def test_no_accounts_registers_only_the_one_provider(tmp_path, monkeypatch):
    _, providers = _discover(tmp_path, monkeypatch, claude_config_dir=str(tmp_path / "login"))
    assert _ours(providers) == [PLUGIN_NAME]


def test_invalid_account_entries_are_skipped(tmp_path, monkeypatch, caplog):
    good = str(tmp_path / "good")
    _, providers = _discover(tmp_path, monkeypatch, accounts={
        "good": good, "Work": "~/x", "has space": "~/y", "under_score": "~/z", "empty": "  ", "number": 7,
        "x" * 33: "~/long"})
    assert _ours(providers) == [PLUGIN_NAME, f"{PLUGIN_NAME}-good"]
    import directsdk_setup
    assert directsdk_setup.accounts() == {"good": good}


@pytest.mark.parametrize("value", [["~/a", "~/b"], "~/a", 3])
def test_accounts_that_are_not_a_mapping_mean_none(tmp_path, monkeypatch, value):
    _, providers = _discover(tmp_path, monkeypatch, accounts=value)
    assert _ours(providers) == [PLUGIN_NAME]


@pytest.mark.parametrize("value,expected", [
    (None, None), ("", None), ("off", None), ("ordered", "ordered"), ("round-robin", "round-robin"),
    ("Round_Robin", "round-robin"), ("random", None), (True, None)])
def test_strategy_values(tmp_path, monkeypatch, value, expected):
    import directsdk_setup
    settings = {} if value is None else {"strategy": value}
    _discover(tmp_path, monkeypatch, **settings)
    assert directsdk_setup.strategy() == expected


def test_manifest_declares_the_settings_under_the_keys_the_code_reads(tmp_path, monkeypatch):
    from hermes_cli.plugins_settings import plugin_settings_fields
    import directsdk_setup
    _discover(tmp_path, monkeypatch)
    fields = {field["key"]: field for field in plugin_settings_fields(PLUGIN_NAME, REPO_ROOT)}
    assert directsdk_setup.ACCOUNTS_SETTING in fields and directsdk_setup.STRATEGY_SETTING in fields
    assert fields["accounts"]["default"] == {}
    assert fields["strategy"]["default"] == "off"
    assert set(directsdk_setup.STRATEGIES) <= set(fields["strategy"]["choices"])


# --- per-account requests, /usage and setup ----------------------------------------------------------------------

def _recording_native(tmp_path, monkeypatch):
    capture = tmp_path / "spawns.jsonl"
    probe = ("open(os.environ['SPAWN_CAPTURE'], 'a').write(json.dumps([os.environ.get(%r), "
             "os.environ.get('CLAUDE_CONFIG_DIR')]) + '\\n')\n" % TOKEN)
    script = tmp_path / "native.py"
    script.write_text(FAKE.replace("rows=[]", probe + "rows=[]"), encoding="utf-8")
    monkeypatch.setenv("SPAWN_CAPTURE", str(capture))
    return [sys.executable, str(script)], lambda: [tuple(json.loads(l)) for l in capture.read_text().splitlines()]


def test_an_account_provider_runs_on_its_own_login_over_the_token_and_setting(tmp_path, monkeypatch):
    main, work = str(tmp_path / "main-login"), str(tmp_path / "work-login")
    _, providers = _discover(tmp_path, monkeypatch, accounts={"main": main, "work": work},
                             claude_config_dir=str(tmp_path / "single-login"))
    command, seen = _recording_native(tmp_path, monkeypatch)
    monkeypatch.setenv(TOKEN, "inherited-token")
    for name in ("work", "main"):
        client = providers.get_provider_profile(f"{PLUGIN_NAME}-{name}").create_client(command=command)
        assert client.create(**TOOL_REQUEST).choices[0].message.content == "hello\n"
    single = providers.get_provider_profile(PLUGIN_NAME).create_client(command=command)
    single.create(**TOOL_REQUEST)
    # The single-account form is unchanged: on a single-profile install the inherited token is the profile's own and
    # decides first beside claude_config_dir. An account provider always runs on its account's directory.
    assert seen() == [(None, work), (None, main), ("inherited-token", str(tmp_path / "single-login"))]


def test_without_accounts_the_single_account_form_is_unchanged(tmp_path, monkeypatch):
    _, providers = _discover(tmp_path, monkeypatch, strategy="ordered")
    command, seen = _recording_native(tmp_path, monkeypatch)
    monkeypatch.setenv(TOKEN, "inherited-token")
    providers.get_provider_profile(PLUGIN_NAME).create_client(command=command).create(**TOOL_REQUEST)
    assert seen() == [("inherited-token", None)]


def test_a_profile_that_does_not_declare_the_account_is_refused(tmp_path, monkeypatch):
    home, providers = _discover(tmp_path, monkeypatch, accounts={"work": str(tmp_path / "work-login")})
    command, seen = _recording_native(tmp_path, monkeypatch)
    work = providers.get_provider_profile(f"{PLUGIN_NAME}-work")
    _write_settings(home)  # the account was removed after the provider registered
    with pytest.raises(Exception) as raised:
        work.create_client(command=command).create(**TOOL_REQUEST)
    assert type(raised.value).__name__ == "ClaudeAccountMissing" and "'work'" in str(raised.value)
    assert work.classify_api_error(raised.value) == {"reason": "format_error", "retryable": False, "should_fallback": True}
    assert not (tmp_path / "spawns.jsonl").exists()


def test_usage_and_setup_ask_each_accounts_own_login(tmp_path, monkeypatch):
    main, work = str(tmp_path / "main-login"), str(tmp_path / "work-login")
    _, providers = _discover(tmp_path, monkeypatch, accounts={"main": main, "work": work})
    # The installed plugin's own copy of the setup module (core imports the plugin as a package).
    setup = sys.modules[type(providers.get_provider_profile(PLUGIN_NAME)).__module__ + ".directsdk_setup"]
    percent = {main: 10, work: 90}
    asked = []

    def ask(resolved, child, exported, upstream, timeout):
        asked.append(child.get("CLAUDE_CONFIG_DIR"))
        return {"limits": [{"kind": "session", "percent": percent[child["CLAUDE_CONFIG_DIR"]]}]}, None
    monkeypatch.setattr(setup, "_ask_usage", ask)
    monkeypatch.setattr(setup, "_resolve", lambda command, env: [sys.executable])
    snapshots = {name: providers.get_provider_profile(f"{PLUGIN_NAME}-{name}").fetch_account_usage()
                 for name in ("main", "work")}
    assert asked == [main, work]
    assert [snapshots[n].provider for n in ("main", "work")] == [f"{PLUGIN_NAME}-main", f"{PLUGIN_NAME}-work"]
    assert [snapshots[n].windows[0].used_percent for n in ("main", "work")] == [10, 90]

    statuses = []
    monkeypatch.setattr(setup.subprocess, "run", lambda argv, **kw: statuses.append(kw["env"].get("CLAUDE_CONFIG_DIR"))
                        or SimpleNamespace(stdout='{"loggedIn": true, "subscriptionType": "max"}', returncode=0))
    for name in ("work", "main"):
        assert providers.get_provider_profile(f"{PLUGIN_NAME}-{name}").setup_status()["logged_in"] is True
    assert statuses == [work, main]


# --- failover --------------------------------------------------------------------------------------------------

NATIVE = r'''
import json, os, sys, urllib.request, urllib.error
for line in sys.stdin:
    frame=json.loads(line)
    if frame.get('shouldQuery') is False:
        print(json.dumps({'type':'result','num_turns':0}),flush=True)
        continue
    break
open(os.environ['SPAWN_CAPTURE'],'a').write(json.dumps(os.environ.get('CLAUDE_CONFIG_DIR'))+'\n')
url=os.environ['ANTHROPIC_BASE_URL']+'/v1/messages'
try:
    urllib.request.urlopen(urllib.request.Request(url,data=b'{}',headers={'Content-Type':'application/json'}),timeout=5).read()
except urllib.error.HTTPError:
    pass
print(json.dumps({'type':'assistant','message':{'id':'first','role':'assistant','content':[{'type':'text','text':'FIRST'}]}}))
print(json.dumps({'type':'stream_event','event':{'type':'message_stop'}}))
print(json.dumps({'type':'result','subtype':'success','usage':{'input_tokens':0,'output_tokens':0}}))
'''


def _sse():
    usage = {"input_tokens": 0, "output_tokens": 0}
    events = [
        {"type": "message_start", "message": {"id": "first", "role": "assistant", "model": "sonnet", "content": [], "usage": usage}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "FIRST"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": usage},
        {"type": "message_stop"}]
    return "".join("data: " + json.dumps(e) + "\n\n" for e in events).encode()


@contextmanager
def _upstream(tmp_path, monkeypatch, answers):
    """A local Messages peer. ``answers[config dir]`` is ``("limit", reset)``, ``("throttle", None)`` or
    ``("overload", None)``; any other account gets a normal reply. Yields (client factory, spawned dirs)."""
    import directsdk
    capture = tmp_path / "dirs.jsonl"
    monkeypatch.setenv("SPAWN_CAPTURE", str(capture))

    class Peer(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            account = json.loads(capture.read_text().splitlines()[-1])
            kind, reset = answers.get(account, (None, None))
            if kind is None:
                body = _sse()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
            else:
                body = LIMIT_BODY
                self.send_response(529 if kind == "overload" else 429)
                if kind == "limit":
                    self.send_header("anthropic-ratelimit-unified-status", "rejected")
                    self.send_header("anthropic-ratelimit-unified-reset", str(reset))
                self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    peer = ThreadingHTTPServer(("127.0.0.1", 0), Peer)
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    native = tmp_path / "native_relay.py"
    native.write_text(NATIVE, encoding="utf-8")
    env = {"PATH": os.defpath, "HOME": str(tmp_path), "SPAWN_CAPTURE": str(capture),
           "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{peer.server_port}"}

    def client(account=None):
        return directsdk.Client(command=[sys.executable, str(native)], env=env, account=account)
    try:
        yield client, lambda: [json.loads(l) for l in capture.read_text().splitlines()] if capture.exists() else []
    finally:
        peer.shutdown()
        thread.join()
        peer.server_close()


def _accounts(tmp_path, monkeypatch, strategy, names=("main", "work")):
    dirs = {name: str(tmp_path / f"{name}-login") for name in names}
    _discover(tmp_path, monkeypatch, accounts=dirs, strategy=strategy)
    return dirs


def _reply(client):
    return client.create(**REQUEST).choices[0].message.content


def test_ordered_moves_a_plan_limited_request_on_and_skips_the_account_until_reset(tmp_path, monkeypatch):
    dirs = _accounts(tmp_path, monkeypatch, "ordered")
    reset = int(time.time()) + 3
    answers = {dirs["main"]: ("limit", reset)}
    with _upstream(tmp_path, monkeypatch, answers) as (client, spawned):
        assert _reply(client()) == "FIRST"
        assert spawned() == [dirs["main"], dirs["work"]]
        assert _reply(client()) == "FIRST"  # main is skipped: no spawn, no upstream 429
        assert spawned() == [dirs["main"], dirs["work"], dirs["work"]]
        answers.clear()  # the window reopened
        while time.time() <= reset:
            time.sleep(0.1)
        assert _reply(client()) == "FIRST"
        assert spawned()[-1] == dirs["main"]


def test_round_robin_spreads_requests_and_skips_a_limited_account(tmp_path, monkeypatch):
    dirs = _accounts(tmp_path, monkeypatch, "round-robin", names=("a", "b", "c"))
    answers = {}
    with _upstream(tmp_path, monkeypatch, answers) as (client, spawned):
        for _ in range(4):
            assert _reply(client()) == "FIRST"
        assert spawned() == [dirs["a"], dirs["b"], dirs["c"], dirs["a"]]
        answers[dirs["b"]] = ("limit", int(time.time()) + 3600)
        for _ in range(3):
            assert _reply(client()) == "FIRST"
        # turn 5 starts at b: limited, moves to c; turn 6 starts at c; turn 7 starts at a.
        assert spawned()[4:] == [dirs["b"], dirs["c"], dirs["c"], dirs["a"]]


def test_all_accounts_limited_returns_the_original_error(tmp_path, monkeypatch):
    import directsdk
    dirs = _accounts(tmp_path, monkeypatch, "ordered")
    now = int(time.time())
    answers = {dirs["main"]: ("limit", now + 7200), dirs["work"]: ("limit", now + 3600)}
    with _upstream(tmp_path, monkeypatch, answers) as (client, spawned):
        with pytest.raises(directsdk.ClaudeAPIError) as raised:
            client().create(**REQUEST)
        assert raised.value.status_code == 429 and raised.value.body == {"resets_at": now + 7200}
        assert spawned() == [dirs["main"], dirs["work"]]
        # Both benched: only the account that reopens first is asked, and Claude's own answer comes back.
        with pytest.raises(directsdk.ClaudeAPIError) as again:
            client().create(**REQUEST)
        assert again.value.body == {"resets_at": now + 3600}
        assert spawned()[2:] == [dirs["work"]]


@pytest.mark.parametrize("kind", ["throttle", "overload"])
def test_other_errors_do_not_fail_over_or_bench(tmp_path, monkeypatch, kind):
    import directsdk
    dirs = _accounts(tmp_path, monkeypatch, "ordered")
    answers = {dirs["main"]: (kind, None)}
    with _upstream(tmp_path, monkeypatch, answers) as (client, spawned):
        with pytest.raises(directsdk.ClaudeAPIError) as raised:
            client().create(**REQUEST)
        assert raised.value.status_code == (429 if kind == "throttle" else 529)
        assert raised.value.body is None
        assert spawned() == [dirs["main"]]
        answers.clear()
        assert _reply(client()) == "FIRST"
        assert spawned() == [dirs["main"], dirs["main"]]


def test_strategy_off_keeps_the_main_provider_on_one_login(tmp_path, monkeypatch):
    import directsdk
    dirs = _accounts(tmp_path, monkeypatch, "off")
    with _upstream(tmp_path, monkeypatch, {None: ("limit", int(time.time()) + 3600)}) as (client, spawned):
        with pytest.raises(directsdk.ClaudeAPIError):
            client().create(**REQUEST)
        assert spawned() == [None]
    assert dirs  # declared, but not used by the main provider without a strategy


def test_an_account_provider_never_switches_but_its_limit_is_remembered(tmp_path, monkeypatch):
    import directsdk
    dirs = _accounts(tmp_path, monkeypatch, "ordered")
    answers = {dirs["work"]: ("limit", int(time.time()) + 3600)}
    with _upstream(tmp_path, monkeypatch, answers) as (client, spawned):
        with pytest.raises(directsdk.ClaudeAPIError) as raised:
            client("work").create(**REQUEST)
        assert raised.value.status_code == 429
        assert spawned() == [dirs["work"]]
        answers[dirs["main"]] = ("limit", int(time.time()) + 600)
        with pytest.raises(directsdk.ClaudeAPIError) as pooled:
            client().create(**REQUEST)  # main limited now; work is still benched from the direct request
        assert spawned()[1:] == [dirs["main"]]
        assert pooled.value.body == {"resets_at": answers[dirs["main"]][1]}


def test_failover_streams_only_the_serving_accounts_reply(tmp_path, monkeypatch):
    dirs = _accounts(tmp_path, monkeypatch, "ordered")
    with _upstream(tmp_path, monkeypatch, {dirs["main"]: ("limit", int(time.time()) + 3600)}) as (client, spawned):
        chunks = list(client().create(stream=True, **REQUEST))
        text = "".join(c.choices[0].delta.content or "" for c in chunks)
        assert text == "FIRST" and chunks[-1].choices[0].finish_reason == "stop"
        assert spawned() == [dirs["main"], dirs["work"]]


def test_cancel_reaches_the_live_attempt(tmp_path, monkeypatch):
    import directsdk
    outer = directsdk.Request(directsdk.Client(command=["unused"]))
    attempt = directsdk.Request(outer.client)
    outer.attempt = attempt
    outer.cancel()
    assert attempt.cancelled.is_set()


# --- multi-profile hosts (composes with the shared-login default of #482) ---------------------------------------

@contextmanager
def _served(home):
    from agent.secret_scope import reset_multiplex_context, reset_secret_scope, set_multiplex_context, set_secret_scope
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    token = set_hermes_home_override(home)
    mode = set_multiplex_context(True)
    bound = set_secret_scope({}, profile_home=str(home))
    try:
        yield
    finally:
        reset_secret_scope(bound)
        reset_multiplex_context(mode)
        reset_hermes_home_override(token)


def test_a_served_profile_runs_on_its_own_declared_account_and_never_borrows_one(tmp_path, monkeypatch):
    import directsdk
    _, providers = _discover(tmp_path, monkeypatch, accounts={"work": str(tmp_path / "launch-work")})
    command, seen = _recording_native(tmp_path, monkeypatch)
    monkeypatch.setenv(TOKEN, "launch-process-token")
    declares = tmp_path / "profiles" / "declares"
    declares.mkdir(parents=True)
    _write_settings(declares, accounts={"work": str(tmp_path / "served-work")})
    silent = tmp_path / "profiles" / "silent"
    silent.mkdir(parents=True)
    _write_settings(silent)
    client = directsdk.Client(command=command, account="work")
    with _served(declares):
        assert client.create(**TOOL_REQUEST).choices[0].message.content == "hello\n"
    with _served(silent), pytest.raises(Exception) as raised:
        client.create(**TOOL_REQUEST)
    assert type(raised.value).__name__ == "ClaudeAccountMissing"
    # The served profile's own directory, never the launch token or the launch profile's account directory.
    assert seen() == [(None, str(tmp_path / "served-work"))]
