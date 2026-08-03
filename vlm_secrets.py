"""
vlm_secrets.py — single place the send chain reads credentials from.
=====================================================================
Credentials live in the gitignored `.env` beside this file, NOT in tracked source.

RESOLUTION ORDER (matches cot dashboard/vlm_secrets.py)
  1. os.environ  -- so CI/systemd/Railway can inject without touching disk
  2. .env beside this file
  3. the `default` argument (None unless given)

STDLIB ONLY, deliberately. python-dotenv is importable on this machine but is NOT
in requirements.txt, so depending on it would break a clean install and the CI
runner's stdlib-only guarantee. Do not "improve" this by importing dotenv.
"""

import os
from pathlib import Path

_ENV_PATH = Path(__file__).resolve().parent / ".env"
_cache = None


def _load_env_file():
    """Parse the .env once. Never raises -- an unreadable .env degrades to {} so a
    caller that also has real env vars set still works."""
    global _cache
    if _cache is not None:
        return _cache
    out = {}
    try:
        if _ENV_PATH.exists():
            for line in _ENV_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                v = v.strip()
                # tolerate KEY="value" / KEY='value'
                if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
                    v = v[1:-1]
                out[k.strip()] = v
    except Exception:
        out = {}
    _cache = out
    return out


def get(name, default=None):
    """Credential by name: real env var wins, then .env, then `default`."""
    v = os.environ.get(name)
    if v and v.strip():
        return v.strip()
    v = _load_env_file().get(name)
    if v and v.strip():
        return v.strip()
    return default


def require(name):
    """Like get(), but fail LOUDLY and immediately if the credential is absent.

    Used for values whose absence would otherwise surface as a confusing auth error
    deep inside Twilio/boto3/urllib at send time. A missing credential is a config
    problem the operator must fix, not something to paper over with a silent no-op.
    """
    v = get(name)
    if not v:
        raise RuntimeError(
            f"Missing credential {name!r}. Set it as an environment variable or add "
            f"'{name}=...' to {_ENV_PATH}. (Credentials are intentionally NOT stored "
            f"in tracked source.)"
        )
    return v


def status(names):
    """Which of `names` resolve, WITHOUT revealing values. For preflight/diagnostics."""
    return {n: bool(get(n)) for n in names}
