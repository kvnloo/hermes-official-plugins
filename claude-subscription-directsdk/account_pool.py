"""Several Claude logins in one profile (#441): which login a request runs on, and plan-limit failover.

A per-account provider runs every request on its own account. The plugin's main provider, with a ``strategy``
and at least two ``accounts``, picks an account per request and moves to the next one when Claude answers a
used-up plan window (a 429 that names its reset, ``ClaudeAPIError.body['resets_at']``). The reset is remembered
for the whole process, so every session, subagent and auxiliary call skips that account until it reopens.
Nothing else moves a request: a throttle 429 without a reset, an overload, an auth failure or any other error is
returned as it is, and so is a limit that arrives after the reply has started.
"""
import itertools
import logging
import math
import os
import threading
import time

try:
    from .directsdk_setup import account_dir, accounts, strategy
except ImportError:
    from directsdk_setup import account_dir, accounts, strategy

logger = logging.getLogger(__name__)

_limits = {}  # normalized config dir -> epoch seconds its used-up plan window reopens
_lock = threading.Lock()
_turns = itertools.count()


def _key(directory):
    return os.path.normcase(os.path.abspath(os.path.expanduser(directory)))


def limited_until(directory, now=None):
    """The epoch second ``directory``'s used-up plan window reopens, or ``None`` when it is not known to be limited."""
    now = time.time() if now is None else now
    key = _key(directory)
    with _lock:
        reset = _limits.get(key)
        if reset is not None and reset <= now:
            del _limits[key]
            reset = None
    return reset


def mark_limited(directory, reset):
    key = _key(directory)
    with _lock:
        _limits[key] = max(reset, _limits.get(key, reset))


def plan_limit_reset(error):
    """The future reset of a used-up plan window this error reports, else ``None``. Only the transport's 429 that
    carries ``body['resets_at']`` qualifies; the relay sets that only when Claude marked the window rejected."""
    if getattr(error, 'status_code', None) != 429:
        return None
    body = getattr(error, 'body', None)
    reset = body.get('resets_at') if isinstance(body, dict) else None
    if isinstance(reset, bool) or not isinstance(reset, (int, float)) or not math.isfinite(reset):
        return None
    return float(reset) if reset > time.time() else None


def plan(account=None):
    """``[(name, config dir), ...]`` to try in order, or ``None`` for the single-login path (no change from before).

    ``account``: a per-account provider, which always runs on its own account. Otherwise failover applies only with
    a ``strategy`` and at least two usable ``accounts``; accounts known to be limited are skipped, and when every
    one is, the one that reopens first is tried so the caller still gets Claude's own answer.
    """
    if account:
        return [(account, account_dir(account))]
    mode = strategy()
    rows = list(accounts().items()) if mode else []
    if len(rows) < 2:
        return None
    if mode == 'round-robin':
        start = next(_turns) % len(rows)
        rows = rows[start:] + rows[:start]
    now = time.time()
    resets = {name: limited_until(directory, now) for name, directory in rows}
    open_rows = [row for row in rows if resets[row[0]] is None]
    return open_rows or [min(rows, key=lambda row: resets[row[0]])]


def _shown(chunk):
    """Whether a streamed chunk carries anything the caller acts on (not a keep-alive)."""
    try:
        choice = chunk.choices[0]
    except (AttributeError, IndexError, TypeError):
        return True
    delta = getattr(choice, 'delta', None)
    return bool(getattr(choice, 'finish_reason', None)) or any(
        getattr(delta, key, None) for key in ('content', 'tool_calls', 'reasoning_content', 'reasoning_details', 'refusal'))


def serve(request, attempts, rows, run):
    """Yield ``run(attempt, config_dir)`` for each row until one is not a plan limit; then raise the first limit.

    ``request`` is the caller's handle (its ``cancel`` reaches the live attempt through ``request.attempt``);
    ``attempts()`` makes a fresh per-attempt handle, since a finished attempt is cancelled by its own cleanup.
    """
    first = None
    for name, directory in rows:
        attempt = attempts()
        with request.lock:
            request.attempt = attempt
        if request.cancelled.is_set():
            raise RuntimeError('Claude request cancelled')
        shown = False
        try:
            for chunk in run(attempt, directory):
                shown = shown or _shown(chunk)
                yield chunk
            return
        except Exception as error:
            reset = plan_limit_reset(error)
            if reset is None:
                raise
            mark_limited(directory, reset)
            if shown or request.cancelled.is_set():
                raise
            first = first or error
            if len(rows) > 1:
                logger.warning('claude-subscription-directsdk: account %r reached its plan limit (reopens %s); trying '
                               'the next account', name, time.strftime('%Y-%m-%d %H:%M %Z', time.localtime(reset)))
        finally:
            with request.lock:
                request.attempt = None
    raise first
