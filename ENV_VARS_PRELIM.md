# Env vars — prelim OI Railway cron worker

Names only, no values, per house rule. Set these in the Railway service's
Variables tab before first deploy. Every one is `vlm_secrets.require(...)` or
a hard `os.environ` read somewhere in the chain -- the process exits loudly at
the point of use if one is missing (never a silent no-op), except where noted.

## Gmail (prelim_oi_watcher.py — connect())

| Var | Used by | Notes |
|---|---|---|
| `GMAIL_USER` | `connect()` | The Gmail address the watcher logs into via IMAP. |
| `GMAIL_COTTON_APP_PASSWORD` | `connect()` | Preferred name (matches the OS-env-var convention this repo already uses elsewhere, e.g. gmail_gain_watcher.py). |
| `GMAIL_APP_PASSWORD` | `connect()` | Fallback name if the above isn't set — `connect()` accepts either. Only one of these two needs to be set, not both. |

## Cloudflare R2 (send_oi_whatsapp.py — image upload, PNG delivery only)

| Var | Used by | Notes |
|---|---|---|
| `CF_ACCOUNT_ID` | `send_oi_whatsapp.upload_to_r2()` | Cloudflare account ID for the R2 endpoint URL. |
| `R2_ACCESS_KEY_ID` | `send_oi_whatsapp.upload_to_r2()` | R2 (S3-compatible) access key. |
| `R2_SECRET_KEY` | `send_oi_whatsapp.upload_to_r2()` | R2 secret key. |

`R2_BUCKET` (`crop-media`) and `R2_PUBLIC_BASE` are NOT secrets and are
hardcoded in `send_oi_whatsapp.py` — no env var needed for those.

## Twilio (send_oi_whatsapp.py send path AND prelim_oi_watcher.alert_failure())

| Var | Used by | Notes |
|---|---|---|
| `TWILIO_SID` | image send + failure alert | Same Twilio account for both the normal PNG delivery and the loud-failure text alert. |
| `TWILIO_TOKEN` | image send + failure alert | |
| `FROM_WA` | image send + failure alert | WhatsApp-enabled Twilio sender number. |
| `TO_WA` | image send + failure alert | Recipient (Lou's WhatsApp number). |

## VLM Data Gateway (build_prelim_oi.py — load_official(), gateway mode)

| Var | Used by | Notes |
|---|---|---|
| `VLM_API_KEY` | `_fetch_gateway_history()` | `X-VLM-API-Key` header value. Required only when `OI_DATA_SOURCE=gateway` (the container default — see below); not read at all in `local` mode. |
| `VLM_API_BASE` | `_fetch_gateway_history()` | Optional. Defaults to `https://vlmapi.vlmdata.com` if unset — only set this to point at a different gateway host (e.g. a staging instance). |
| `OI_DATA_SOURCE` | `load_official()` | Optional. `gateway` (default) or `local`. Set to `local` only for offline testing against `data/oi_data.csv` bundled/mounted in the container — normal production runs should leave this unset (defaults to `gateway`, keeping the service stateless per the architectural requirement). |

## Not required as env vars

- `DASHBOARD_TOKEN_SECRET`, `VLM_ALERT_URL`, `VLM_ALERT_SECRET`, `VLM_APP_NAME`
  — these belong to `vlm_auth.py` (the dashboard's UI auth gate), which this
  cron worker never imports.
- `GITHUB_TOKEN`, `SUPABASE_URL`, `VLM_PUSH_SECRET` — present in the repo's
  local `.env` for other scripts (bootstrap/master-fetch tooling) but not
  read anywhere in the prelim watcher/builder/sender chain. Do not carry
  these into the cron worker's Railway service unless something in this
  chain starts needing them.

## Source of this list

Enumerated by grepping every `os.environ.get/[...]` and `vlm_secrets.require/
get` call in `prelim_oi_watcher.py`, `build_prelim_oi.py`, and
`send_oi_whatsapp.py` (2026-08-26). Re-run that grep if any of those three
files change what they read.
