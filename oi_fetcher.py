"""
oi_fetcher.py — VLM Open Interest Monitor
Smart incremental updater: detects every missing trading day since the last
CSV entry, backfills via Bloomberg BDH, appends new rows, and git-pushes
so Railway redeploys the dashboard automatically.

Run at any time Bloomberg Terminal is open and logged in:
  - Weekday after close → appends today's EOD
  - After a gap (weekend / holiday / missed day) → backfills all missing days
    then appends today
  - Weekend / off-hours with no new Bloomberg data → exits cleanly

Task Scheduler (recommended):
  Program:   pythonw.exe
  Arguments: "<full path>/oi_fetcher.py"
  Trigger:   Daily at 9:35 AM EST, Monday–Friday
  (Bloomberg Terminal must be open and logged in when it fires)
"""

import sys, subprocess, pathlib, csv
from datetime import date, timedelta, datetime

# ── Paths ───────────────────────────────────────────────────────────────────────
BASE_DIR  = pathlib.Path(__file__).parent
DATA_DIR  = BASE_DIR / 'data'
DATA_FILE = DATA_DIR / 'oi_data.csv'
LOG_FILE  = BASE_DIR / 'oi_fetcher.log'

# ── Tickers (month-specific generics — matches oi_bootstrap.py) ─────────────────
# Bloomberg rolls these automatically each crop year.
# CTJUL1 = nearest July Cotton contract (JUL 26 now, JUL 27 after expiry, etc.)
# No numbered slots (CT1/CT2) — those distort history on roll dates.
TICKERS = {
    # Cotton No.2  (Mar/May/Jul/Oct/Dec × 2)
    'CTMAR1 Comdty': ('CT', 'CTMAR1'),
    'CTMAY1 Comdty': ('CT', 'CTMAY1'),
    'CTJUL1 Comdty': ('CT', 'CTJUL1'),
    'CTOCT1 Comdty': ('CT', 'CTOCT1'),
    'CTDEC1 Comdty': ('CT', 'CTDEC1'),
    'CTMAR2 Comdty': ('CT', 'CTMAR2'),
    'CTMAY2 Comdty': ('CT', 'CTMAY2'),
    'CTJUL2 Comdty': ('CT', 'CTJUL2'),
    'CTOCT2 Comdty': ('CT', 'CTOCT2'),
    'CTDEC2 Comdty': ('CT', 'CTDEC2'),
    # Sugar No.11  (Mar/May/Jul/Oct × 2)
    'SBMAR1 Comdty': ('SB', 'SBMAR1'),
    'SBMAY1 Comdty': ('SB', 'SBMAY1'),
    'SBJUL1 Comdty': ('SB', 'SBJUL1'),
    'SBOCT1 Comdty': ('SB', 'SBOCT1'),
    'SBMAR2 Comdty': ('SB', 'SBMAR2'),
    'SBMAY2 Comdty': ('SB', 'SBMAY2'),
    'SBJUL2 Comdty': ('SB', 'SBJUL2'),
    'SBOCT2 Comdty': ('SB', 'SBOCT2'),
    # Coffee C  (Mar/May/Jul/Sep/Dec × 2)
    'KCMAR1 Comdty': ('KC', 'KCMAR1'),
    'KCMAY1 Comdty': ('KC', 'KCMAY1'),
    'KCJUL1 Comdty': ('KC', 'KCJUL1'),
    'KCSEP1 Comdty': ('KC', 'KCSEP1'),
    'KCDEC1 Comdty': ('KC', 'KCDEC1'),
    'KCMAR2 Comdty': ('KC', 'KCMAR2'),
    'KCMAY2 Comdty': ('KC', 'KCMAY2'),
    'KCJUL2 Comdty': ('KC', 'KCJUL2'),
    'KCSEP2 Comdty': ('KC', 'KCSEP2'),
    'KCDEC2 Comdty': ('KC', 'KCDEC2'),
    # Cocoa  (Mar/May/Jul/Sep/Dec × 2)
    'CCMAR1 Comdty': ('CC', 'CCMAR1'),
    'CCMAY1 Comdty': ('CC', 'CCMAY1'),
    'CCJUL1 Comdty': ('CC', 'CCJUL1'),
    'CCSEP1 Comdty': ('CC', 'CCSEP1'),
    'CCDEC1 Comdty': ('CC', 'CCDEC1'),
    'CCMAR2 Comdty': ('CC', 'CCMAR2'),
    'CCMAY2 Comdty': ('CC', 'CCMAY2'),
    'CCJUL2 Comdty': ('CC', 'CCJUL2'),
    'CCSEP2 Comdty': ('CC', 'CCSEP2'),
    'CCDEC2 Comdty': ('CC', 'CCDEC2'),
}

CSV_COLUMNS = ['date', 'commodity', 'contract', 'bbg_ticker',
               'settle', 'open_int', 'oi_chg', 'first_notice', 'last_trade']


# ── Logging ──────────────────────────────────────────────────────────────────────
def log(msg):
    ts   = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    try:
        print(line)
    except UnicodeEncodeError:
        print(line.encode('ascii', errors='replace').decode('ascii'))
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


# ── CSV helpers ───────────────────────────────────────────────────────────────────
def get_last_date():
    """Return the most recent date string in the CSV, or None if file absent/empty."""
    if not DATA_FILE.exists():
        return None
    try:
        with open(DATA_FILE, 'r', encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
        if rows:
            return max(r['date'] for r in rows)
    except Exception as e:
        log(f'WARNING: could not read CSV last date: {e}')
    return None


def get_last_oi():
    """Return {bbg_ticker: open_int} for the most recent date in the CSV."""
    if not DATA_FILE.exists():
        return {}
    try:
        with open(DATA_FILE, 'r', encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return {}
        last_date = max(r['date'] for r in rows)
        return {r['bbg_ticker']: int(r['open_int'])
                for r in rows
                if r['date'] == last_date and r.get('open_int')}
    except Exception as e:
        log(f'WARNING: could not load last OI: {e}')
    return {}


def append_rows(new_rows):
    """Append rows to the CSV (creates header if file does not exist)."""
    DATA_DIR.mkdir(exist_ok=True)
    write_header = not DATA_FILE.exists()
    with open(DATA_FILE, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(new_rows)
    log(f'Appended {len(new_rows)} rows to {DATA_FILE.name}')


# ── Bloomberg BDH fetch ───────────────────────────────────────────────────────────
def fetch_bdh(start_str, end_str):
    """
    Pull daily PX_LAST + OPEN_INT + FUT_NOTICE_FIRST + LAST_TRADEABLE_DT
    for every ticker via Bloomberg BDH (historical data request).
    Returns { bbg_ticker: [ {date, settle, open_int, first_notice, last_trade}, ... ] }
    ACTIVE_DAYS_ONLY means Bloomberg only returns actual trading days —
    weekends and holidays are silently excluded.
    """
    try:
        import blpapi
    except ImportError:
        log('ERROR: blpapi not installed.')
        return None

    try:
        opts = blpapi.SessionOptions()
        opts.setServerHost('localhost')
        opts.setServerPort(8194)
        session = blpapi.Session(opts)

        if not session.start():
            log('ERROR: Bloomberg session failed to start. Is Terminal running?')
            return None
        if not session.openService('//blp/refdata'):
            session.stop()
            log('ERROR: Could not open //blp/refdata.')
            return None

        svc     = session.getService('//blp/refdata')
        results = {}
        total   = len(TICKERS)

        for idx, (ticker, (commodity, contract)) in enumerate(TICKERS.items(), 1):
            log(f'  [{idx:02d}/{total}] {ticker} ...')
            req = svc.createRequest('HistoricalDataRequest')
            req.getElement('securities').appendValue(ticker)
            req.getElement('fields').appendValue('PX_LAST')
            req.getElement('fields').appendValue('OPEN_INT')
            req.getElement('fields').appendValue('FUT_NOTICE_FIRST')
            req.getElement('fields').appendValue('LAST_TRADEABLE_DT')
            req.set('startDate', start_str.replace('-', ''))
            req.set('endDate',   end_str.replace('-', ''))
            req.set('periodicitySelection', 'DAILY')
            req.set('nonTradingDayFillOption', 'ACTIVE_DAYS_ONLY')
            session.sendRequest(req)

            ticker_rows = []
            done = False
            while not done:
                ev = session.nextEvent(8000)
                for msg in ev:
                    if msg.hasElement('securityData'):
                        sd       = msg.getElement('securityData')
                        fd_array = sd.getElement('fieldData')
                        for j in range(fd_array.numValues()):
                            fd    = fd_array.getValue(j)
                            d_val = fd.getElementAsDatetime('date')
                            dt    = f'{d_val.year:04d}-{d_val.month:02d}-{d_val.day:02d}'
                            px = oi = fn = lt = None
                            if fd.hasElement('PX_LAST') and not fd.getElement('PX_LAST').isNull():
                                raw = fd.getElementAsFloat('PX_LAST')
                                if raw and raw > 0:
                                    px = round(float(raw), 2)
                            if fd.hasElement('OPEN_INT') and not fd.getElement('OPEN_INT').isNull():
                                raw = fd.getElementAsFloat('OPEN_INT')
                                if raw and raw > 0:
                                    oi = int(raw)
                            if fd.hasElement('FUT_NOTICE_FIRST') and not fd.getElement('FUT_NOTICE_FIRST').isNull():
                                try:
                                    v  = fd.getElementAsDatetime('FUT_NOTICE_FIRST')
                                    fn = f'{v.year:04d}-{v.month:02d}-{v.day:02d}'
                                except Exception:
                                    pass
                            if fd.hasElement('LAST_TRADEABLE_DT') and not fd.getElement('LAST_TRADEABLE_DT').isNull():
                                try:
                                    v  = fd.getElementAsDatetime('LAST_TRADEABLE_DT')
                                    lt = f'{v.year:04d}-{v.month:02d}-{v.day:02d}'
                                except Exception:
                                    pass
                            if px is not None or oi is not None:
                                ticker_rows.append({
                                    'date': dt, 'settle': px, 'open_int': oi,
                                    'first_notice': fn or '', 'last_trade': lt or '',
                                })
                if ev.eventType() == blpapi.Event.RESPONSE:
                    done = True

            if ticker_rows:
                log(f'       -> {len(ticker_rows)} trading day(s)')
                results[ticker] = ticker_rows
            else:
                log(f'       -> no data returned')

        session.stop()
        return results if results else None

    except Exception as e:
        log(f'ERROR: Bloomberg exception: {e}')
        return None


# ── Build CSV rows from BDH results ──────────────────────────────────────────────
def build_rows(raw, prev_oi):
    """
    Convert fetch_bdh() results into flat CSV rows sorted by date.
    prev_oi: {bbg_ticker: int} — OI values for the day before the fetch window,
             used to calculate oi_chg for the first fetched day per ticker.
    """
    all_rows = []

    for ticker, (commodity, contract) in TICKERS.items():
        ticker_data = raw.get(ticker, [])
        if not ticker_data:
            continue
        ticker_data.sort(key=lambda r: r['date'])

        # Start oi_chg chain from the last known OI before this window
        running_prev = prev_oi.get(ticker)

        for r in ticker_data:
            oi     = r['open_int']
            oi_chg = (oi - running_prev) if (oi is not None and running_prev is not None) else ''
            all_rows.append({
                'date':         r['date'],
                'commodity':    commodity,
                'contract':     contract,
                'bbg_ticker':   ticker,
                'settle':       r['settle']       if r['settle']       is not None else '',
                'open_int':     oi                if oi                is not None else '',
                'oi_chg':       oi_chg,
                'first_notice': r['first_notice'],
                'last_trade':   r['last_trade'],
            })
            if oi is not None:
                running_prev = oi

    # Sort: date asc, then commodity, then contract label (matches bootstrap sort)
    all_rows.sort(key=lambda r: (r['date'], r['commodity'], r['contract']))
    return all_rows


# ── Git push ──────────────────────────────────────────────────────────────────────
def git_push(new_dates):
    dates_str = ', '.join(sorted(set(new_dates)))
    msg = f'auto: OI data {dates_str}'
    try:
        cwd = str(BASE_DIR)
        subprocess.run(['git', 'add', 'data/oi_data.csv'],
                       cwd=cwd, check=True, capture_output=True, text=True)
        result = subprocess.run(['git', 'commit', '-m', msg],
                                cwd=cwd, capture_output=True, text=True)
        if result.returncode != 0:
            if 'nothing to commit' in result.stdout:
                log('Git: nothing to commit (data unchanged — Bloomberg may not have new EOD yet)')
                return True
            log(f'Git commit failed: {result.stderr.strip()}')
            return False
        result = subprocess.run(['git', 'push'],
                                cwd=cwd, capture_output=True, text=True)
        if result.returncode == 0:
            log(f'Git: pushed — Railway will redeploy. Dates: {dates_str}')
            return True
        log(f'Git push failed: {result.stderr.strip()}')
        return False
    except Exception as e:
        log(f'Git error: {e}')
        return False


# ── Main ──────────────────────────────────────────────────────────────────────────
def main():
    log('--- oi_fetcher.py started ---')

    # Determine the fetch window
    last_date_str = get_last_date()
    if last_date_str is None:
        log('ERROR: oi_data.csv not found — run oi_bootstrap.py first.')
        return 1

    last_date  = date.fromisoformat(last_date_str)
    today      = date.today()
    fetch_from = last_date + timedelta(days=1)
    fetch_to   = today

    if fetch_from > fetch_to:
        log(f'Data is already current through {last_date_str}. Nothing to fetch.')
        return 0

    log(f'Last date in CSV : {last_date_str}')
    log(f'Fetch window     : {fetch_from} -> {fetch_to}')
    log(f'Tickers          : {len(TICKERS)}')

    # Load prev OI for oi_chg calculation on the first new day
    prev_oi = get_last_oi()
    log(f'Previous OI loaded for {len(prev_oi)} tickers.')

    # Bloomberg BDH fetch
    raw = fetch_bdh(fetch_from.strftime('%Y-%m-%d'), fetch_to.strftime('%Y-%m-%d'))
    if not raw:
        log('Bloomberg returned no data — EOD may not be published yet. Try again later.')
        return 1

    # Build and append rows
    new_rows = build_rows(raw, prev_oi)
    if not new_rows:
        log('No new rows to append.')
        return 1

    new_dates = sorted(set(r['date'] for r in new_rows))
    log(f'New trading dates: {new_dates}')

    append_rows(new_rows)
    git_push(new_dates)

    log(f'--- oi_fetcher.py complete: {len(new_rows)} rows across {len(new_dates)} date(s) ---')
    return 0


if __name__ == '__main__':
    sys.exit(main())
