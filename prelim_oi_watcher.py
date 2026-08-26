"""
prelim_oi_watcher.py — VLM ICE Preliminary OI email trigger

The ICE preliminary OI report (ice.com/report/114) is reCAPTCHA-gated
("recaptchaRequired": true straight from ICE's own metadata API), so there is no
legitimate unattended download. This closes that gap from the other end: Lou
downloads the CSV on his phone at ~5am and emails it to himself; this watcher
picks it up, builds the report, and sends the PNG back via WhatsApp.

Trigger: an unread email whose SUBJECT contains "prelim", with a .csv attached.
Matching on subject (not sender) means it works from any address he happens to
send from.

Idempotency: the message is marked \\Seen only after a successful build, so a
crash or a Twilio outage leaves it unread and the next poll retries it.

LOUD FAILURE (added for the Railway port, 2026-08-26): the fatal flaw this
port fixes is that "built but not delivered" looked IDENTICAL to success in
the Windows Task Scheduler log -- on 2026-08-26 the scheduler killed the
process mid-WhatsApp-send at its ExecutionTimeLimit; the PNG/XLSX were built
fine but never reached WhatsApp, and the log just stopped with no error.
Retry-via-unread-message covers a crash on the NEXT poll, but says nothing
in the moment a send actually fails. alert_failure() below writes a loud,
banner-delimited DELIVERY FAILURE block naming the reason, so that case can
never again look like a normal log tail.

It is LOG-ONLY by design -- no WhatsApp, no email. Lou is the alert: if the
PNG has not arrived by ~07:00 he already knows something broke, and what he
needs from this file is the REASON, without reading a traceback. A second
WhatsApp would add nothing and, on a persistently failing send, would fire
once per 5-minute poll across the whole window.
A SIGTERM handler covers the same 2026-08-26 scenario on Railway: a graceful
container stop/restart mid-send now alerts before exiting, instead of just
stopping.

Usage:
  python prelim_oi_watcher.py              # poll once, build+send if found
  python prelim_oi_watcher.py --dry-run    # build but do NOT send or mark read
  python prelim_oi_watcher.py --no-send    # build + mark read, skip WhatsApp
"""

import argparse, email, imaplib, os, pathlib, signal, subprocess, sys
from datetime import datetime

BASE_DIR   = pathlib.Path(__file__).parent
INBOX_DIR  = BASE_DIR / 'data' / 'prelim_inbox'
OUT_DIR    = BASE_DIR / 'output' / 'prelim'
LOG_FILE   = BASE_DIR / 'prelim_oi_watcher.log'

# Just 'prelim' -- IMAP SUBJECT search is a case-insensitive substring match, so
# "prelim", "Prelim OI", "fwd: prelim" all trigger. Kept short so it is trivial to
# type on a phone. It still has to be present: without a subject filter, ANY unread
# mail with a CSV attached would trigger a build.
SUBJECT_TOKEN = 'prelim'
# Fallback trigger (added 2026-08-25 after a blank-subject send was skipped all
# morning): unread mail FROM this address also counts, but ONLY if it carries the
# ICE prelim CSV itself (filename check) — so ordinary self-sent mail with other
# CSVs attached can never trigger a build.
FALLBACK_SENDER = 'thecottonkid@gmail.com'
PRELIM_CSV_PREFIX = 'preliminaryopeninterest'
IMAP_HOST, IMAP_PORT = 'imap.gmail.com', 993

# No Eastern-time / DST gating logic here by design (see railway.prelim.toml):
# the cron schedule itself is a plain static UTC expression, widened by an
# hour on each side of the real 04:30-07:00 ET target so it never needs to
# know which side of DST the calendar is on. Extra firings outside the real
# window are cheap -- if there's no unread prelim email, main() logs that and
# returns immediately below (no Gmail/build work happens), and a message is
# only ever marked \\Seen after a successful send, so a wider firing window
# can never cause a double-send.


def _load_env():
    """Credentials live in VLM Data's .env (same file the GAIN watcher uses)."""
    for p in (BASE_DIR / '.env',
              BASE_DIR.parent / 'VLM Data' / '.env'):
        if p.exists():
            for ln in p.read_text(encoding='utf-8').splitlines():
                ln = ln.strip()
                if ln and not ln.startswith('#') and '=' in ln:
                    k, v = ln.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip())


def log(msg):
    line = f'[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}'
    print(line)
    with LOG_FILE.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def connect():
    # The app password lives under two different names across this machine:
    # GMAIL_COTTON_APP_PASSWORD (OS env var, what gmail_gain_watcher.py uses) and
    # GMAIL_APP_PASSWORD (repo .env). Accept either so the watcher does not depend
    # on which one happens to be present.
    user = os.environ.get('GMAIL_USER')
    pw   = (os.environ.get('GMAIL_COTTON_APP_PASSWORD')
            or os.environ.get('GMAIL_APP_PASSWORD'))
    if not (user and pw):
        missing = []
        if not user: missing.append('GMAIL_USER')
        if not pw:   missing.append('GMAIL_COTTON_APP_PASSWORD or GMAIL_APP_PASSWORD')
        log(f'ERROR: missing credential(s): {", ".join(missing)} — '
            f'checked OS env, {BASE_DIR / ".env"}, {BASE_DIR.parent / "VLM Data" / ".env"}')
        sys.exit(1)
    m = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
    m.login(user, pw)
    return m


def _has_prelim_csv(msg):
    """True if any attachment filename is the ICE prelim CSV (e.g.
    'PreliminaryOpenInterestFutures (6).csv')."""
    for part in msg.walk():
        fn = part.get_filename() or ''
        if fn.lower().startswith(PRELIM_CSV_PREFIX) and fn.lower().endswith('.csv'):
            return True
    return False


def find_messages(m):
    """Unread messages whose subject contains the trigger token, plus the
    blank-subject fallback: unread mail from FALLBACK_SENDER that actually
    carries the ICE prelim CSV."""
    m.select('INBOX')
    status, data = m.uid('search', None, f'(UNSEEN SUBJECT "{SUBJECT_TOKEN}")')
    uids = list(data[0].split()) if status == 'OK' and data[0] else []
    st_fb, data_fb = m.uid('search', None, f'(UNSEEN FROM "{FALLBACK_SENDER}")')
    fb = [u for u in (data_fb[0].split() if st_fb == 'OK' and data_fb[0] else [])
          if u not in uids]
    out = []
    for uid in uids + fb:
        st, raw = m.uid('fetch', uid, '(RFC822)')
        if st != 'OK' or not raw or not raw[0]:
            continue
        msg = email.message_from_bytes(raw[0][1])
        if uid in fb and not _has_prelim_csv(msg):
            continue  # self-sent mail without the prelim CSV — not a trigger
        out.append((uid, msg))
    return out


def extract_csv(msg):
    """First CSV attachment as (filename, bytes). Some phone mail clients send
    CSVs as application/octet-stream or text/plain, so match on extension
    rather than trusting the declared content-type."""
    found = []
    for part in msg.walk():
        fn = part.get_filename()
        if not fn or not fn.lower().endswith('.csv'):
            continue
        payload = part.get_payload(decode=True)
        if payload:
            found.append((fn, payload))
    if len(found) > 1:
        # Only the first is processed; say so rather than discarding silently.
        log(f'  NOTE: {len(found)} CSV attachments — using {found[0][0]!r}, '
            f'ignoring {", ".join(repr(f) for f, _ in found[1:])}')
    return found[0] if found else (None, None)


def build(csv_path):
    """Run the builder as a subprocess so its own validation/exit codes govern.
    Returns (ok, output).

    Timeout is kept safely under the Task Scheduler ExecutionTimeLimit (20 min,
    raised from 10 on 2026-08-26) so a slow build is caught HERE, with a log line,
    rather than being killed by the scheduler with no trace. A timeout returns
    ok=False like any other failure, so the email stays unread and the next
    5-minute poll retries it.

    The 10-min limit was NOT enough: on 2026-08-26 a build that normally takes 4s
    took 306s under machine load. The build itself finished and wrote PNG+XLSX,
    but the scheduler killed the process during the WhatsApp send (result 267014),
    so the report was built and never delivered. The budget must cover build+send,
    not just build.
    """
    try:
        r = subprocess.run(
            [sys.executable, str(BASE_DIR / 'build_prelim_oi.py'), '--csv', str(csv_path)],
            capture_output=True, text=True, cwd=str(BASE_DIR), timeout=480)
    except subprocess.TimeoutExpired as e:
        dec = lambda v: (v.decode('utf-8', 'replace') if isinstance(v, bytes)
                         else (v or ''))
        return False, f'BUILD TIMEOUT after 480s\n{dec(e.stdout)}{dec(e.stderr)}'
    except Exception as e:
        return False, f'BUILD FAILED to launch: {e!r}'
    return r.returncode == 0, (r.stdout or '') + (r.stderr or '')


def session_from(stdout):
    """Pull the session date the builder reported, to locate its output."""
    for ln in stdout.splitlines():
        if ln.startswith('Session'):
            return ln.split(':', 1)[1].strip().split()[0]
    return None


def send(png, session):
    """Reuse the proven R2 + Twilio primitives from the OI sender."""
    sys.path.insert(0, str(BASE_DIR))
    import send_oi_whatsapp as s
    urls = s.upload_to_r2([str(png)], session)
    return s.send_whatsapp_image(urls[0], f'ICE PRELIM OI — {session}', session)


# ── Loud failure ──────────────────────────────────────────────────────────────
# Currently-processing marker for the SIGTERM handler below -- set right before
# build() starts, cleared once the message is fully handled (sent+marked read,
# or a normal failure branch already logged/alerted its own reason). If the
# process is killed while this is non-None, the SIGTERM handler knows a build
# was in flight and alerts even though nothing else caught it.
_in_flight = None


def alert_failure(reason):
    """Record a 'built but not delivered' failure LOUDLY in the log.

    Deliberately LOG-ONLY. Lou is the alert: if the PNG does not arrive on
    WhatsApp by ~07:00 he already knows something broke, so a second WhatsApp
    message adds nothing and a retrying failure would have fired one on every
    5-minute poll across the window. What he needs from this file is the
    REASON, stated plainly, without reading a traceback.

    The failure this exists for (2026-08-26) produced a log whose last line was
    a normal-looking `XLSX : ...` — the process was killed mid-send, so the
    absence of a success line was the ONLY signal, and it read as a build
    failure that never happened. These markers make that case unmistakable.
    """
    log('  ' + '=' * 68)
    log(f'  DELIVERY FAILURE: {reason}')
    log('  The report may exist on disk but was NOT delivered. Check output/prelim/.')
    log('  ' + '=' * 68)


def _sigterm_handler(signum, frame):
    """Railway sends SIGTERM before SIGKILL on redeploys/restarts/OOM-adjacent
    stops. This is the exact 2026-08-26 failure mode (killed mid-send) —
    alert BEFORE exiting rather than just stopping silently."""
    if _in_flight:
        log(f'  SIGTERM received while processing {_in_flight!r} — alerting')
        alert_failure(f'process received SIGTERM mid-run (in flight: {_in_flight}) '
                       f'— build may be done but NOT confirmed delivered; message '
                       f'left unread, will retry next poll')
    else:
        log('  SIGTERM received (idle) — exiting')
    sys.exit(143)  # 128 + SIGTERM(15), conventional


def main():
    global _in_flight
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true',
                    help='build only; do not send or mark read')
    ap.add_argument('--no-send', action='store_true',
                    help='build and mark read, but skip WhatsApp')
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _sigterm_handler)

    _load_env()
    INBOX_DIR.mkdir(parents=True, exist_ok=True)

    m = connect()
    msgs = find_messages(m)
    if not msgs:
        log('no unread PRELIM OI messages')
        m.logout()
        return

    log(f'{len(msgs)} message(s) to process')
    for uid, msg in msgs:
        subj = str(msg.get('Subject', ''))[:80]
        fn, payload = extract_csv(msg)
        if not payload:
            log(f'  UID {uid.decode()}: no CSV attachment ({subj!r}) — leaving unread')
            continue

        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved = INBOX_DIR / f'{stamp}_{fn}'
        saved.write_bytes(payload)
        log(f'  saved {saved.name} ({len(payload):,} bytes)')
        _in_flight = saved.name

        ok, out = build(saved)
        for ln in out.splitlines():
            log(f'    | {ln}')
        if not ok:
            log('  BUILD FAILED — leaving message unread for retry')
            # Build failures are not alerted: they retry silently every 5 min
            # and (per the CSV-recency guard in build_prelim_oi.py) most are
            # a transient/expected condition, not the "silent success-looking
            # failure" this alert path exists for. A build that NEVER
            # recovers across the whole polling window still shows up in the
            # log for review; alerting here would just be noisy.
            _in_flight = None
            continue

        session = session_from(out)
        png = OUT_DIR / f'prelim_oi_{session}.png' if session else None
        if not (png and png.exists()):
            log(f'  built but PNG missing ({png}) — leaving unread')
            alert_failure(f'build reported success but PNG is missing '
                          f'(session={session}, expected {png}) — left unread for retry')
            _in_flight = None
            continue

        if args.dry_run:
            log(f'  DRY RUN — built {png.name}, not sending, left unread')
            _in_flight = None
            continue

        if not args.no_send:
            try:
                if send(png, session):
                    log(f'  WhatsApp sent for {session}')
                else:
                    log('  WhatsApp send FAILED — leaving unread for retry')
                    alert_failure(f'build SUCCEEDED (session={session}, {png.name}) '
                                  f'but WhatsApp send FAILED — report exists but was '
                                  f'NOT delivered; left unread, will retry next poll')
                    _in_flight = None
                    continue
            except Exception as e:
                log(f'  send error: {e} — leaving unread for retry')
                alert_failure(f'build SUCCEEDED (session={session}, {png.name}) but '
                              f'the send raised {e!r} — report exists but was NOT '
                              f'confirmed delivered; left unread, will retry next poll')
                _in_flight = None
                continue

        m.uid('store', uid, '+FLAGS', '\\Seen')
        log(f'  marked read (UID {uid.decode()})')
        _in_flight = None

    m.logout()


if __name__ == '__main__':
    main()
