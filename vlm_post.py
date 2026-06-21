"""
vlm_post.py — posts an HTML table to the VLM member site.
Place this file in the same folder as build_whatsapp_oi.py.
"""

import os
import urllib.request
import urllib.error
import json

def _load_local_env(_p=os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')):
    if os.path.exists(_p):
        for _ln in open(_p, encoding='utf-8'):
            _ln=_ln.strip()
            if _ln and not _ln.startswith('#') and '=' in _ln:
                _k,_v=_ln.split('=',1); os.environ.setdefault(_k.strip(), _v.strip())
_load_local_env()

VLM_URL    = "https://test.vlmdata.com"
VLM_SECRET = os.environ.get("VLM_PUSH_SECRET")


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
        },
        method = "POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
            print(f"[vlm_post] posted OK — slug: {result.get('slug')}")
            return result
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"[vlm_post] HTTP {e.code}: {body}") from e
