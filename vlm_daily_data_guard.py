"""
vlm_daily_data_guard.py — did the 09:30 master fetch actually land?
===================================================================
OBSERVATION ONLY. This script does not import, call, modify, or coordinate with
vlm_master_fetch.py in any way. It reads file mtimes and sends a WhatsApp if any
expected output is stale. Master fetch is a proven, working job; it is deliberately
left untouched. If this guard breaks, the pipeline is unaffected.

WHY THIS EXISTS (2026-08-26)
  Master fetch hung at 'JOB 2: OI dashboard data append' — a 17MB read of
  oi_data.csv wedged on a OneDrive file lock for 22+ minutes (the identical read
  took 1 SECOND after a reboot). It wrote nothing. Downstream, build_whatsapp_oi.py
  correctly refused to send stale cards. Nothing alerted. The failure was found only
  because the cards never arrived on Lou's phone.

WHY mtime AND NOT EXIT CODE / TASK STATUS
  The hung task reported `Last Result: 0` after the reboot killed it. Zero is the
  reboot-kill exit, not success. vlm_master_fetch.py also ends in an unconditional
  `return 0` even when its own summary['errors'] is populated, so its exit code
  cannot express failure at all. A scheduler kill (267014) surfaces as its own code
  and a hang surfaces as nothing whatsoever, because a hung task never reports.
  ==> The ONLY honest signal that today's fetch did work is that the output files
      were physically written today. Verify the EFFECT, never the status field.

WHY mtime AND NOT trade_date
  OI is T+1: on a good day oi_data.csv's max trade_date is legitimately YESTERDAY.
  Gating on trade_date would therefore false-alarm every single morning. mtime is
  what proves *today's run* touched the file. This is the same reasoning as
  build_whatsapp_oi.check_freshness(), deliberately mirrored.

RUN
  python vlm_daily_data_guard.py           # alert only if something is stale
  python vlm_daily_data_guard.py --dry-run # print verdict, never send
  python vlm_daily_data_guard.py --force   # send even if everything is fresh (test)

Schedule Mon-Fri ~09:45, after master fetch (09:30) has had time to finish.
Exit 0 = all fresh. Exit 1 = at least one file stale (alert attempted).
"""

import argparse
import base64
import pathlib
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime

import vlm_secrets

BASE_DIR = pathlib.Path(__file__).resolve().parent
DESKTOP = BASE_DIR.parent
LOG_FILE = BASE_DIR / 'vlm_daily_data_guard.log'

# The five files vlm_master_fetch.py writes, across THREE repo trees. Sourced by
# reading its own path constants (OI_CSV, OPT_FILE, SPREAD_CSV, SIGNALS_CSV,
# MACRO_CSV) — not guessed. macro_features.csv lives in the CTA MONITOR repo, so a
# guard scoped to this repo alone would miss a CTA-side failure entirely.
WATCHED = [
    ('oi_data.csv', BASE_DIR / 'data' / 'oi_data.csv'),
    ('options_oi.csv', BASE_DIR / 'data' / 'options_oi.csv'),
    ('spread_ohlc.csv', BASE_DIR / 'data' / 'spread_ohlc.csv'),
    ('macro_signals.csv', BASE_DIR / 'data' / 'signals' / 'macro_signals.csv'),
    ('macro_features.csv', DESKTOP / 'CTA MONITOR' / 'data' / 'macro_features.csv'),
]


def log(msg):
    stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{stamp}] {msg}'
    print(line)
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception:
        # A guard that dies on its own logging is worse than a guard with no log.
        pass


def check_files():
    """Return (stale, fresh, missing) as lists of (label, path, mtime_or_None)."""
    today = date.today()
    stale, fresh, missing = [], [], []
    for label, path in WATCHED:
        try:
            if not path.exists():
                missing.append((label, path, None))
                continue
            mtime = date.fromtimestamp(path.stat().st_mtime)
            if mtime == today:
                fresh.append((label, path, mtime))
            else:
                stale.append((label, path, mtime))
        except Exception as e:
            # An unreadable file is itself a problem worth alerting on — a
            # OneDrive lock can make stat() fail, which is exactly the class of
            # fault this guard was built for. Never swallow it into "fresh".
            log(f'  ERROR reading {label}: {e}')
            missing.append((label, path, None))
    return stale, fresh, missing


def send_whatsapp(body):
    """Text-only Twilio send. No R2/boto3 — that is only needed for images."""
    sid = vlm_secrets.require('TWILIO_SID')
    token = vlm_secrets.require('TWILIO_TOKEN')
    from_wa = vlm_secrets.require('FROM_WA')
    to_wa = vlm_secrets.require('TO_WA')

    api_url = f'https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json'
    auth = base64.b64encode((sid + ':' + token).encode('utf-8')).decode('utf-8')
    data = urllib.parse.urlencode({'From': from_wa, 'To': to_wa, 'Body': body}).encode('utf-8')
    req = urllib.request.Request(
        api_url,
        data=data,
        headers={
            'Authorization': 'Basic ' + auth,
            'Content-Type': 'application/x-www-form-urlencoded',
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true', help='Report verdict, never send.')
    ap.add_argument('--force', action='store_true',
                    help='Send even when everything is fresh (for testing the alert path).')
    args = ap.parse_args()

    log('--- daily data guard ---')
    stale, fresh, missing = check_files()

    for label, path, mtime in fresh:
        log(f'  OK    {label}  (mtime {mtime})')
    for label, path, mtime in stale:
        log(f'  STALE {label}  (mtime {mtime}, expected {date.today()})')
    for label, path, _ in missing:
        log(f'  GONE  {label}  ({path})')

    problems = stale + missing
    if not problems and not args.force:
        log(f'All {len(fresh)} watched files fresh. No alert.')
        return 0

    # Idempotency: one alert per calendar day. A stale file stays stale all day, so
    # without this a re-run (or a second scheduled fire) would re-alert repeatedly.
    # Deliberately NOT applied to --force, which exists to test the send path.
    marker = BASE_DIR / f'.guard_alerted_{date.today()}'
    if marker.exists() and not args.force:
        log(f'Already alerted today ({marker.name}) — not re-sending.')
        return 1

    lines = [f'VLM DATA GUARD — {date.today()}', '']
    if stale:
        lines.append('STALE (master fetch did not write these today):')
        lines += [f'  - {label} (last {mtime})' for label, _, mtime in stale]
    if missing:
        lines.append('MISSING / UNREADABLE:')
        lines += [f'  - {label}' for label, _, _ in missing]
    if fresh:
        lines.append('')
        lines.append(f'OK: {", ".join(label for label, _, _ in fresh)}')
    lines += [
        '',
        'Check: vlm_master_fetch.log tail, and whether the task is wedged.',
        'Task status/exit code is NOT proof — verify file mtime.',
    ]
    body = '\n'.join(lines)

    if args.dry_run:
        log('DRY RUN — message that WOULD be sent:')
        print('\n' + body + '\n')
        return 1 if problems else 0

    try:
        status = send_whatsapp(body)
        log(f'WhatsApp alert sent: HTTP {status}')
        # Marker written only on a SUCCESSFUL send, so a Twilio outage leaves the
        # alert retryable rather than silently suppressed for the rest of the day.
        if problems:
            marker.write_text(f'alerted {datetime.now().isoformat()}\n', encoding='utf-8')
    except Exception as e:
        log(f'WhatsApp alert FAILED: {e}')
        return 1

    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
