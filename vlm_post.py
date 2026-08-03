"""
vlm_post.py — posts an HTML table to the VLM member site.
Place this file in the same folder as build_whatsapp_oi.py.
"""

import time
import urllib.request
import urllib.error
import json

import vlm_secrets

VLM_URL    = "https://test.vlmdata.com"
# Resolves from env var -> gitignored .env (see vlm_secrets). Deliberately get()
# and not require(): post_to_vlm() below already raises ValueError on an empty
# secret, and that existing failure path is what the send chain expects.
VLM_SECRET = vlm_secrets.get("VLM_PUSH_SECRET")

# Retry policy: the auto-send chain runs unattended, so a single transient blip
# (network drop, Cloudflare/origin 5xx, timeout) must not lose the daily post.
# Retry only TRANSIENT failures; a 4xx (bad secret, forbidden, bad payload) is a
# hard error that won't fix itself — fail fast so the alert fires immediately.
_MAX_ATTEMPTS = 3
_BACKOFF_SECS = (2, 5)  # waits before attempt 2 and attempt 3


def post_to_vlm(title, content, category="oi", excerpt="", status="published"):
    if not VLM_SECRET:
        raise ValueError("VLM_SECRET is not set in vlm_post.py")

    payload = json.dumps({
        "title":    title,
        "content":  content,
        "category": category,
        "excerpt":  excerpt,
        "status":   status,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{VLM_URL}/api/push/post",
        data    = payload,
        headers = {
            "Content-Type":  "application/json",
            "x-push-secret": VLM_SECRET,
            "User-Agent":    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        },
        method = "POST",
    )

    last_err = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read())
                print(f"[vlm_post] posted OK — slug: {result.get('slug')}"
                      + (f" (attempt {attempt})" if attempt > 1 else ""))
                return result
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            # 4xx = hard error (auth/payload) — do NOT retry. 5xx / 429 = transient.
            if e.code < 500 and e.code != 429:
                raise RuntimeError(f"[vlm_post] HTTP {e.code}: {body}") from e
            last_err = RuntimeError(f"[vlm_post] HTTP {e.code}: {body}")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_err = RuntimeError(f"[vlm_post] network error: {e}")

        if attempt < _MAX_ATTEMPTS:
            wait = _BACKOFF_SECS[attempt - 1]
            print(f"[vlm_post] attempt {attempt}/{_MAX_ATTEMPTS} failed "
                  f"({last_err}) — retrying in {wait}s")
            time.sleep(wait)

    raise last_err
