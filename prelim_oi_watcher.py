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

Usage:
  python prelim_oi_watcher.py              # poll once, build+send if found
  python prelim_oi_watcher.py --dry-run    # build but do NOT send or mark read
  python prelim_oi_watcher.py --no-send    # build + mark read, skip WhatsApp
"""

import argparse, email, imaplib, os, pathlib, subprocess, sys
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
IMAP_HOST, IMAP_PORT = 'imap.gmail.com', 993


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


def find_messages(m):
    """Unread messages whose subject contains the trigger token."""
    m.select('INBOX')
    status, data = m.uid('search', None, f'(UNSEEN SUBJECT "{SUBJECT_TOKEN}")')
    if status != 'OK' or not data[0]:
        return []
    out = []
    for uid in data[0].split():
        st, raw = m.uid('fetch', uid, '(RFC822)')
        if st == 'OK' and raw and raw[0]:
            out.append((uid, email.message_from_bytes(raw[0][1])))
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

    Timeout is kept safely under the Task Scheduler ExecutionTimeLimit (10 min) so
    a slow build is caught HERE, with a log line, rather than being killed by the
    scheduler with no trace. A timeout returns ok=False like any other failure, so
    the email stays unread and the next 5-minute poll retries it.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true',
                    help='build only; do not send or mark read')
    ap.add_argument('--no-send', action='store_true',
                    help='build and mark read, but skip WhatsApp')
    args = ap.parse_args()

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

        ok, out = build(saved)
        for ln in out.splitlines():
            log(f'    | {ln}')
        if not ok:
            log('  BUILD FAILED — leaving message unread for retry')
            continue

        session = session_from(out)
        png = OUT_DIR / f'prelim_oi_{session}.png' if session else None
        if not (png and png.exists()):
            log(f'  built but PNG missing ({png}) — leaving unread')
            continue

        if args.dry_run:
            log(f'  DRY RUN — built {png.name}, not sending, left unread')
            continue

        if not args.no_send:
            try:
                if send(png, session):
                    log(f'  WhatsApp sent for {session}')
                else:
                    log('  WhatsApp send FAILED — leaving unread for retry')
                    continue
            except Exception as e:
                log(f'  send error: {e} — leaving unread for retry')
                continue

        m.uid('store', uid, '+FLAGS', '\\Seen')
        log(f'  marked read (UID {uid.decode()})')

    m.logout()


if __name__ == '__main__':
    main()
