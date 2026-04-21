"""
oi_fetcher.py — VLM Open Interest Monitor
Pulls settlement price + open interest for all generic rolling contract months
for CT, SB, KC, CC from Bloomberg Terminal via blpapi.
Appends today's rows to data/oi_data.csv, then git-pushes so Railway redeploys.

Generic tickers roll automatically — no ticker updates needed year to year.
  CT1 = front month Cotton, CT2 = second month, etc.

Schedule via Windows Task Scheduler:
  Program:   pythonw.exe
  Arguments: "C:/Users/Louis/OneDrive - VLM Commodities LTD/Desktop/Open interest dashboard/oi_fetcher.py"
  Trigger:   Daily at 9:35 AM EST, Monday-Friday

Bloomberg Terminal must be open and logged in when this runs.
"""

import sys, subprocess, pathlib, csv
from datetime import date, datetime

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_DIR  = pathlib.Path(__file__).parent
DATA_DIR  = BASE_DIR / 'data'
DATA_FILE = DATA_DIR / 'oi_data.csv'
LOG_FILE  = BASE_DIR / 'oi_fetcher.log'

# Generic rolling Bloomberg tickers.
# Bloomberg rolls these automatically — CT1 is always front month,
# CT2 is always second month, etc. No updates needed year to year.
# Format: 'BBG_TICKER': ('COMMODITY', 'GENERIC_LABEL')
TICKERS = {
    # ── Cotton No.2 ────────────────────────────────
    'CT1 Comdty':     ('CT', 'CT1'),
    'CT2 Comdty':     ('CT', 'CT2'),
    'CTDEC1 Comdty':  ('CT', 'CT DEC1'),   # Dec generic (crop year anchor)
    'CTMAR1 Comdty':  ('CT', 'CT MAR1'),
    'CTMAY1 Comdty':  ('CT', 'CT MAY1'),
    'CTJUL1 Comdty':  ('CT', 'CT JUL1'),
    'CTOCT1 Comdty':  ('CT', 'CT OCT1'),

    # ── Sugar No.11 ────────────────────────────────
    'SB1 Comdty':     ('SB', 'SB1'),
    'SB2 Comdty':     ('SB', 'SB2'),
    'SBMAR1 Comdty':  ('SB', 'SB MAR1'),
    'SBMAY1 Comdty':  ('SB', 'SB MAY1'),
    'SBJUL1 Comdty':  ('SB', 'SB JUL1'),
    'SBOCT1 Comdty':  ('SB', 'SB OCT1'),

    # ── Coffee C ───────────────────────────────────
    'KC1 Comdty':     ('KC', 'KC1'),
    'KC2 Comdty':     ('KC', 'KC2'),
    'KCMAY1 Comdty':  ('KC', 'KC MAY1'),
    'KCJUL1 Comdty':  ('KC', 'KC JUL1'),
    'KCSEP1 Comdty':  ('KC', 'KC SEP1'),
    'KCDEC1 Comdty':  ('KC', 'KC DEC1'),

    # ── Cocoa ──────────────────────────────────────
    'CC1 Comdty':     ('CC', 'CC1'),
    'CC2 Comdty':     ('CC', 'CC2'),
    'CCMAY1 Comdty':  ('CC', 'CC MAY1'),
    'CCJUL1 Comdty':  ('CC', 'CC JUL1'),
    'CCSEP1 Comdty':  ('CC', 'CC SEP1'),
    'CCDEC1 Comdty':  ('CC', 'CC DEC1'),
}

CSV_COLUMNS = ['date', 'commodity', 'contract', 'bbg_ticker', 'settle', 'open_int', 'oi_chg']


# ── Logging ─────────────────────────────────────────────────────────────────────
def log(msg):
    ts   = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


# ── Bloomberg fetch ──────────────────────────────────────────────────────────────
def fetch_bloomberg():
    """Pull PX_LAST + OPEN_INT for all tickers. Returns dict keyed by BBG ticker."""
    try:
        import blpapi
    except ImportError:
        log('ERROR: blpapi not installed.  Run: pip install blpapi')
        return None

    try:
        opts = blpapi.SessionOptions()
        opts.setServerHost('localhost')
        opts.setServerPort(8194)
        session = blpapi.Session(opts)

        if not session.start():
            log('ERROR: Could not start Bloomberg session. Is Terminal running?')
            return None
        if not session.openService('//blp/refdata'):
            session.stop()
            log('ERROR: Could not open //blp/refdata service.')
            return None

        svc = session.getService('//blp/refdata')
        req = svc.createRequest('ReferenceDataRequest')
        for ticker in TICKERS:
            req.getElement('securities').appendValue(ticker)
        req.getElement('fields').appendValue('PX_LAST')
        req.getElement('fields').appendValue('OPEN_INT')
        # Also pull the actual expiry month so we know what contract CT1 is today
        req.getElement('fields').appendValue('FUT_CUR_GEN_TICKER')
        session.sendRequest(req)

        results = {}
        done = False
        while not done:
            ev = session.nextEvent(8000)
            for msg in ev:
                if msg.hasElement('securityData'):
                    sd = msg.getElement('securityData')
                    for i in range(sd.numValues()):
                        row  = sd.getValue(i)
                        tick = row.getElementAsString('security')
                        if tick not in TICKERS:
                            continue
                        fd = row.getElement('fieldData')
                        px        = None
                        oi        = None
                        cur_tick  = None
                        if fd.hasElement('PX_LAST') and not fd.getElement('PX_LAST').isNull():
                            px = round(float(fd.getElementAsFloat('PX_LAST')), 2)
                        if fd.hasElement('OPEN_INT') and not fd.getElement('OPEN_INT').isNull():
                            oi = int(fd.getElementAsFloat('OPEN_INT'))
                        if fd.hasElement('FUT_CUR_GEN_TICKER') and not fd.getElement('FUT_CUR_GEN_TICKER').isNull():
                            cur_tick = fd.getElementAsString('FUT_CUR_GEN_TICKER')
                        if px is not None or oi is not None:
                            results[tick] = {
                                'settle':      px,
                                'open_int':    oi,
                                'actual_contract': cur_tick or '',
                            }
                            log(f'  {tick:18s}  [{cur_tick or "?":10s}]  settle={px}  OI={oi}')
            if ev.eventType() == blpapi.Event.RESPONSE:
                done = True

        session.stop()
        return results if results else None

    except Exception as e:
        log(f'ERROR: Bloomberg exception: {e}')
        return None


# ── CSV helpers ──────────────────────────────────────────────────────────────────
def load_yesterday_oi():
    """Return dict of {bbg_ticker: open_int} for the most recent date in the CSV."""
    if not DATA_FILE.exists():
        return {}
    prev = {}
    try:
        with open(DATA_FILE, 'r', encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return {}
        last_date = max(r['date'] for r in rows)
        for r in rows:
            if r['date'] == last_date and r['open_int']:
                prev[r['bbg_ticker']] = int(r['open_int'])
    except Exception as e:
        log(f'WARNING: Could not read previous OI: {e}')
    return prev


def append_to_csv(rows):
    """Append list of row dicts to DATA_FILE, creating file + header if needed."""
    DATA_DIR.mkdir(exist_ok=True)
    write_header = not DATA_FILE.exists()
    with open(DATA_FILE, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    log(f'CSV: appended {len(rows)} rows to {DATA_FILE.name}')


# ── Git push ─────────────────────────────────────────────────────────────────────
def git_push(today_str):
    try:
        cwd = str(BASE_DIR)
        subprocess.run(
            ['git', 'add', 'data/oi_data.csv'],
            cwd=cwd, check=True, capture_output=True, text=True)
        result = subprocess.run(
            ['git', 'commit', '-m', f'auto: OI data {today_str}'],
            cwd=cwd, capture_output=True, text=True)
        if result.returncode != 0:
            if 'nothing to commit' in result.stdout:
                log('Git: nothing to commit (OI unchanged)')
                return True
            log(f'Git commit failed: {result.stderr.strip()}')
            return False
        result = subprocess.run(
            ['git', 'push'], cwd=cwd, capture_output=True, text=True)
        if result.returncode == 0:
            log('Git: pushed to GitHub — Railway will redeploy.')
            return True
        else:
            log(f'Git push failed: {result.stderr.strip()}')
            return False
    except Exception as e:
        log(f'Git error: {e}')
        return False


# ── Main ─────────────────────────────────────────────────────────────────────────
def main():
    log('--- oi_fetcher.py started ---')

    today = date.today()
    if today.weekday() >= 5:                        # Sat=5, Sun=6
        log(f'Weekend ({today.strftime("%A")}) — skipping.')
        return 0

    today_str = today.strftime('%Y-%m-%d')

    # Load previous day OI so we can calculate oi_chg
    prev_oi = load_yesterday_oi()
    log(f'Previous OI loaded for {len(prev_oi)} tickers.')

    # Fetch from Bloomberg
    raw = fetch_bloomberg()
    if not raw:
        log('Bloomberg returned no data. CSV unchanged.')
        return 1

    # Build rows for today
    csv_rows = []
    for ticker, (commodity, contract) in TICKERS.items():
        data     = raw.get(ticker, {})
        settle   = data.get('settle')
        open_int = data.get('open_int')
        actual   = data.get('actual_contract', '')    # e.g. "CTN26" — what CT1 is today
        if settle is None and open_int is None:
            log(f'  SKIP {ticker} — no data returned')
            continue
        prev   = prev_oi.get(ticker)
        oi_chg = (open_int - prev) if (open_int is not None and prev is not None) else ''
        csv_rows.append({
            'date':       today_str,
            'commodity':  commodity,
            'contract':   actual if actual else contract,   # store actual e.g. "CTN26" when known
            'bbg_ticker': ticker,
            'settle':     settle   if settle   is not None else '',
            'open_int':   open_int if open_int is not None else '',
            'oi_chg':     oi_chg,
        })

    if not csv_rows:
        log('No rows to write. CSV unchanged.')
        return 1

    append_to_csv(csv_rows)
    git_push(today_str)
    log(f'--- oi_fetcher.py complete: {len(csv_rows)} contracts written ---')
    return 0


if __name__ == '__main__':
    sys.exit(main())
