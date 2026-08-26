# prelim-oi-watcher — Railway deploy steps

Service ALREADY CREATED: project "Open interest dashboard"
(98ee73f8-e2f7-4522-ae2f-b1af9c830a12), service `prelim-oi-watcher`
(20fc255d-3c8d-465f-b494-2bcf406612bc), environment `production`.

Already set: OI_DATA_SOURCE=gateway, VLM_API_BASE=https://vlmapi.vlmdata.com
Railway auto-injects RAILWAY_ENVIRONMENT (confirmed present).

## 1. Add the 10 credentials (Railway -> prelim-oi-watcher -> Variables)

| Variable | Copy from |
|---|---|
| GMAIL_USER          | Desktop/VLM Data/.env   <-- ONLY one from this file |
| GMAIL_APP_PASSWORD  | Open interest dashboard/.env |
| CF_ACCOUNT_ID       | Open interest dashboard/.env |
| R2_ACCESS_KEY_ID    | Open interest dashboard/.env |
| R2_SECRET_KEY       | Open interest dashboard/.env |
| TWILIO_SID          | Open interest dashboard/.env |
| TWILIO_TOKEN        | Open interest dashboard/.env |
| FROM_WA             | Open interest dashboard/.env |
| TO_WA               | Open interest dashboard/.env |
| VLM_API_KEY         | Open interest dashboard/.env |

GMAIL_USER is the easy one to miss: it is in a DIFFERENT .env from the other
nine. Without it IMAP login fails while everything else looks correctly set.

## 2. Attach the source

Settings -> Source: GitHub repo `vlmsofts/oi-dashboard`, branch `main`.

## 3. Point it at the right config (REQUIRED)

The repo root already contains railway.json + Procfile for the EXISTING `web`
dashboard service (nixpacks/gunicorn). Railway reads railway.json/railway.toml
by DEFAULT, so without this step the prelim service would build as the
dashboard web app instead of the cron worker.

  Settings -> Config as Code -> Config Path: `railway.prelim.toml`
  Settings -> Build -> Dockerfile Path:      `Dockerfile.prelim`

## 4. Cron

railway.prelim.toml sets `cronSchedule = "*/5 7-13 * * 1-5"` = 07:00-13:55 UTC,
Mon-Fri. Covers 03:30-08:00 ET in BOTH EST (UTC-5) and EDT (UTC-4) -- one
static line, no DST logic, window widened 1h each side on purpose.

## 5. VERIFY THE EFFECT, NOT THE STATUS

Railway's deployment status and nextCronRunAt both lie (nextCronRunAt COMPUTES
FORWARD from the expression -- it looks correct even if the job never fired).
The only real proof is the WhatsApp PNG arriving, or the deploy logs showing
`no unread PRELIM OI messages`.

FIRST RUN IS THE REAL TEST: the image has NEVER been built. Docker and WSL are
both absent from the dev machine, so Linux font rendering is REASONED, not
observed. The report declares one stack, `Arial,sans-serif`; Arial does not
exist on Linux; Liberation Sans is metrically identical (same advance widths)
so the fixed-width table cannot reflow, and Dockerfile.prelim installs
fonts-liberation explicitly. Compare the first container PNG against
output/prelim/prelim_oi_2026-08-25.png before trusting it.

## 6. Do NOT disable the Windows task yet

`VLM Prelim OI Watcher` (ExecutionTimeLimit now PT20M) still runs and still
sends. build_prelim_oi.py defaults OI_DATA_SOURCE=local precisely so the
Windows job is unaffected by any of this. Disable it only after the container
has delivered a correct PNG.

## Open item

railway.json (live, `web` service) uses UPPERCASE `restartPolicyType:
"ON_FAILURE"`; vlm-data-gateway/railway.toml (live) uses lowercase
`"on_failure"`. Both are in production. railway.prelim.toml currently uses
lowercase `"never"` on that precedent. If the build rejects it, uppercase it.
