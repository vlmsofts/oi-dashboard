"""
gateway_expiry.py -- live ICE expiry rows from the VLM gateway, cache-first.
=============================================================================
Consumes the ecosystem's shared expiry authority
    GET https://vlmapi.vlmdata.com/v1/expiry/{CT|CC|KC|SB}/{futures|options}
(desk-side scheduled refresh from ICE, Fridays 07:30 + on-demand). Verified live
2026-09-02: CT 23 option rows, KC 10, CC 12, SB 17, all including Jan27; row
fields = commodity/contract/month_label/kind/ftd/ltd/fnd/lnd/fdd/ldd/fsd/
is_expired/refreshed_at; envelope = cached/stale/last_verified_at/content_stale/
content_age_days/row_count/data/refreshed_at.

Pattern copied from the sandbox repo's engine/gateway_expiry.py (cache-first,
fail-loud, CSV-floor merge) -- NOT imported across repos; this file is this
repo's own self-contained copy, trimmed to what contract_dates.py needs.

SCOPE: additive data source. contract_expiries.json remains the historical
floor -- the gateway board is LIVE-LISTED ONLY (confirmed empirically: even
?include_expired=true returns expired_row_count=0), so it can never resolve a
contract that has since rolled off. contract_dates.py merges JSON (floor) with
this module's rows (gateway wins overlaps).

Offline behaviour: a fetch failure serves the last cached payload of ANY age
(logged). No cache + no network -> [] so callers degrade to JSON-only, never
crash, never fabricate a date.

Stdlib-only (urllib) -- no new dependency.
"""
import os
import json
import time
import datetime as dt
import urllib.request
import urllib.error

import vlm_secrets

_HERE = os.path.dirname(os.path.abspath(__file__))
_CACHE_DIR = os.path.join(_HERE, "data", "gateway_expiry_cache")
_LOG = os.path.join(_HERE, "gateway_expiry.log")

BASE = os.environ.get("VLM_API_BASE", "https://vlmapi.vlmdata.com")
_TIMEOUT_SEC = 15
# Upstream refresh is weekly (Fri 07:30) + on-demand; 6h keeps us far fresher
# than the source can change while avoiding a network hit on every import.
_FRESH_TTL_SEC = 6 * 3600
# After a failed fetch, don't retry the network for this long within the same
# process -- avoids hammering a black-holed gateway on repeated lookups.
_FAIL_BACKOFF_SEC = 15 * 60
_last_fail = {}  # (CMD, kind) -> time.monotonic() of last failed fetch

_KINDS = ("options", "futures")


def _logline(m):
    try:
        with open(_LOG, "a", encoding="utf-8") as fh:
            fh.write(f"{dt.datetime.now().isoformat()}  {m}\n")
    except OSError:
        pass


def _cache_path(commodity, kind):
    return os.path.join(_CACHE_DIR, f"{commodity.upper()}_{kind}.json")


def _read_cache(commodity, kind):
    """-> (payload dict, age_seconds) or (None, None). Any parse failure = no cache."""
    path = _cache_path(commodity, kind)
    try:
        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
        fetched = dt.datetime.fromisoformat(blob["fetched_at"])
        age = (dt.datetime.now() - fetched).total_seconds()
        return blob["payload"], age
    except (OSError, ValueError, KeyError):
        return None, None


def _write_cache(commodity, kind, payload):
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = _cache_path(commodity, kind)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"fetched_at": dt.datetime.now().isoformat(), "payload": payload}, fh)
    os.replace(tmp, path)


def _fetch_gateway(commodity, kind):
    """One GET against the gateway. Returns the payload dict or raises."""
    key = vlm_secrets.get("VLM_API_KEY")
    if not key:
        raise RuntimeError("no VLM_API_KEY (env or .env)")
    url = f"{BASE}/v1/expiry/{commodity.upper()}/{kind}"
    req = urllib.request.Request(url, headers={"X-VLM-API-Key": key})
    with urllib.request.urlopen(req, timeout=_TIMEOUT_SEC) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_rows(commodity, kind="options", max_cache_age_sec=_FRESH_TTL_SEC):
    """Row dicts for {commodity}/{kind}, cache-first.

    Fresh cache (< max_cache_age_sec) -> no network. Otherwise fetch + recache;
    on ANY fetch failure serve the cached payload regardless of age (logged
    STALE_SERVE). Returns [] when neither network nor cache is available --
    callers must treat [] as 'gateway unavailable' and fall back to their JSON
    floor. Never raises.
    """
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}")
    cmd = commodity.upper()
    payload, age = _read_cache(cmd, kind)
    if payload is not None and age is not None and age < max_cache_age_sec:
        return payload.get("data") or []
    last = _last_fail.get((cmd, kind))
    if last is not None and (time.monotonic() - last) < _FAIL_BACKOFF_SEC:
        return (payload.get("data") or []) if payload is not None else []
    try:
        fresh = _fetch_gateway(cmd, kind)
    except Exception as e:
        _last_fail[(cmd, kind)] = time.monotonic()
        if payload is not None:
            _logline(f"STALE_SERVE {cmd}/{kind} age={int(age)}s fetch_err={e!r} "
                      f"(backoff {_FAIL_BACKOFF_SEC}s) -- serving last-known-good cache")
            return payload.get("data") or []
        _logline(f"GATEWAY_DOWN_NO_CACHE {cmd}/{kind} {e!r} (backoff {_FAIL_BACKOFF_SEC}s) "
                 f"-- degrading to contract_expiries.json floor only")
        return []
    _last_fail.pop((cmd, kind), None)
    if fresh.get("stale") or fresh.get("content_stale"):
        _logline(f"GATEWAY_STALE_FLAG {cmd}/{kind} refreshed_at={fresh.get('refreshed_at')} "
                 f"last_verified_at={fresh.get('last_verified_at')}")
    _write_cache(cmd, kind, fresh)
    _logline(f"FETCH_OK {cmd}/{kind} rows={fresh.get('row_count')} "
             f"refreshed_at={fresh.get('refreshed_at')}")
    return fresh.get("data") or []


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--commodity", default="CT")
    ap.add_argument("--kind", default="options")
    a = ap.parse_args()
    for row in fetch_rows(a.commodity, a.kind):
        print(row.get("contract"), "ltd=", row.get("ltd"), "fnd=", row.get("fnd"),
              "is_expired=", row.get("is_expired"))
