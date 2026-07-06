"""
vlm_auth.py — shared dashboard auth gate for VLM Flask/Dash dashboards.

Verifies a signed token issued by vlmdata.com and gates the dashboard UI's HTML
entry points. API routes, admin routes, and static assets stay reachable. Token-
less / invalid / expired requests to a gated path get a 302 to the login page.

Token contract (matches the vlmdata.com issuer — verified against official test
vectors V1/V2/V3, byte-for-byte token re-mint confirmed):
    payload  = base64url( JSON.stringify({ sub, exp }) )   # compact, no padding
    sig      = base64url( HMAC_SHA256(secret, payload_string) )
    token    = payload + "." + sig
    exp in MILLISECONDS; reject when now_ms >= exp.
    Signature verified BEFORE the payload is JSON-decoded.
    Comparison is constant-time (hmac.compare_digest).
    Split on the LAST '.' (lastIndexOf parity with the Node vlmAuth).

Gating model (parity with the deployed Node vlmAuth): an ALLOWLIST of HTML entry
points is gated (default: "/", "/index.html", "/dashboard.html"); every other
path — /api/*, /admin, /static/*, health — passes through untouched. Pass
`gated_paths` to match each app's real UI route(s).

Missing-secret behavior: FAIL OPEN (serve the request) but fire a one-shot,
non-blocking ops alert via the vlm-alert worker, so members are never disrupted
by a misconfigured deploy while the owner is notified to fix it. Never raises,
never blocks the response.

Env vars:
    DASHBOARD_TOKEN_SECRET  (required for gating to actually verify)
    VLM_ALERT_URL           (optional) vlm-alert worker URL
    VLM_ALERT_SECRET        (optional) X-VLM-Secret for the worker
    VLM_APP_NAME            (optional) label used in the alert; falls back to host

Stdlib only — no new dependencies.
"""

import os
import time
import hmac
import json
import base64
import hashlib
import logging
import threading
import urllib.request
from functools import partial

from flask import request, redirect, make_response

LOGIN_URL = "https://vlmdata.com/login"
COOKIE_NAME = "vlm_dash"
COOKIE_MAX_AGE = 4 * 60 * 60  # 4 hours, seconds
DEFAULT_GATED = ("/", "/index.html", "/dashboard.html")

_log = logging.getLogger("vlm_auth")
_alert_sent = False           # module-level dedupe: one alert per process
_alert_lock = threading.Lock()


def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _b64url_encode(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")


def verify_token(token: str, secret: str, now_ms: int | None = None) -> bool:
    """True iff token is a valid, unexpired, correctly-signed token."""
    if not token or not secret:
        return False
    if now_ms is None:
        now_ms = int(time.time() * 1000)
    dot = token.rfind(".")          # lastIndexOf parity
    if dot < 1:
        return False
    payload_str, sig = token[:dot], token[dot + 1:]
    expected = _b64url_encode(
        hmac.new(secret.encode("utf-8"), payload_str.encode("utf-8"),
                 hashlib.sha256).digest()
    )
    if not hmac.compare_digest(sig, expected):   # constant-time
        return False
    try:
        data = json.loads(_b64url_decode(payload_str))
    except Exception:
        return False
    exp = data.get("exp")
    if not isinstance(exp, (int, float)):
        return False
    if now_ms >= exp:               # reject when now >= exp
        return False
    return True


def _fire_alert_once(reason: str):
    """Send a deduped, non-blocking ops alert. Never raises."""
    global _alert_sent
    with _alert_lock:
        if _alert_sent:
            return
        _alert_sent = True

    app_name = os.environ.get("VLM_APP_NAME") or (request.host if request else "unknown")
    url = os.environ.get("VLM_ALERT_URL")
    secret = os.environ.get("VLM_ALERT_SECRET")

    # Always log — this is the guaranteed floor even if the worker isn't wired.
    _log.error("VLM_AUTH_ALERT app=%s reason=%s", app_name, reason)

    if not url or not secret:
        return  # no alert channel configured; the log line above is the record

    def _send():
        try:
            body = json.dumps({
                "app": app_name, "message": reason, "level": "critical",
            }).encode("utf-8")
            req = urllib.request.Request(
                url, data=body, method="POST",
                headers={"Content-Type": "application/json", "X-VLM-Secret": secret},
            )
            urllib.request.urlopen(req, timeout=5).read()
        except Exception as e:       # alerting must never affect the dashboard
            _log.error("VLM_AUTH_ALERT send failed: %s", e)

    threading.Thread(target=_send, daemon=True).start()


def _gate(gated_paths):
    """before_request handler. Returns None to allow, or a redirect response."""
    path = request.path

    # Only the HTML entry points are gated. Everything else (/api/*, /admin,
    # /static/*, health, Dash XHRs) passes through.
    if path not in gated_paths:
        return None

    secret = os.environ.get("DASHBOARD_TOKEN_SECRET")
    if not secret:
        # FAIL OPEN + alert: don't disrupt members over a config slip.
        _fire_alert_once("DASHBOARD_TOKEN_SECRET missing — auth gate failing open")
        return None

    token = request.args.get("token") or request.cookies.get(COOKIE_NAME)

    if verify_token(token, secret):
        if request.args.get("token"):
            # Arrived via ?token=: set the 4hr cookie, then redirect to the same
            # path with the token stripped but ALL OTHER query params preserved
            # (parity with the Node gate).
            args = request.args.to_dict(flat=True)
            args.pop("token", None)
            from urllib.parse import urlencode
            qs = urlencode(args)
            dest = path + ("?" + qs if qs else "")
            resp = make_response(redirect(dest, code=302))
            resp.set_cookie(
                COOKIE_NAME, token,
                max_age=COOKIE_MAX_AGE,
                httponly=True, secure=True, samesite="Lax",
            )
            return resp
        return None  # valid cookie already present

    return redirect(LOGIN_URL, code=302)


def install_gate(app, gated_paths=DEFAULT_GATED):
    """
    Attach the auth gate to a Flask app (or Dash app.server).

    gated_paths: exact HTML entry-point paths to protect. Default covers the
    common cases ("/", "/index.html", "/dashboard.html"). Pass the app's real
    UI route(s) if they differ (e.g. add "/mobile", "/history.html").

    Everything not in gated_paths is left untouched — /api/*, /admin, /static/*,
    health checks, and Dash /_dash-* XHRs all pass through. (Dash XHRs carry the
    cookie once the gated index page has loaded, so the app stays protected.)
    """
    app.before_request(partial(_gate, tuple(gated_paths)))
    return app
