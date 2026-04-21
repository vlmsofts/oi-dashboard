"""
oi_bootstrap.py — VLM Open Interest Monitor
ONE-TIME script to pull full historical daily OI + settle from Bloomberg
using BDH (historical data) with generic rolling tickers.
Writes data/oi_data.csv from scratch — run this ONCE before starting
the daily oi_fetcher.py Task Scheduler job.

Bloomberg Terminal must be open and logged in.

Usage (in PowerShell):
  python "C:/Users/Louis/OneDrive - VLM Commodities LTD/Desktop/Open interest dashboard/oi_bootstrap.py"
  python oi_bootstrap.py --start 2015-01-01
  python oi_bootstrap.py --start 2010-01-01 --end 2025-04-20
"""

import sys, csv, pathlib, argparse, shutil
from datetime import date, datetime

BASE_DIR  = pathlib.Path(__file__).parent
DATA_DIR  = BASE_DIR / 'data'
DATA_FILE = DATA_DIR / 'oi_data.csv'
LOG_FILE  = BASE_DIR / 'oi_bootstrap.log'

CSV_COLUMNS = ['date', 'commodity', 'contract', 'bbg_ticker', 'settle', 'open_int', 'oi_chg']

# Generic rolling Bloomberg tickers — same as oi_fetcher.py.
# BDH on these returns the historical time series of each generic slot.
# CT1 on 2015-01-01 returns the front month that was active on that date, etc.
TICKERS = {
    # ── Cotton No.2 ────────────────────────────────
    'CT1 Comdty':     ('CT', 'CT1'),
    'CT2 Comdty':     ('CT', 'CT2'),
    'CTDEC1 Comdty':  ('CT', 'CT DEC1'),
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


def log(msg):
    ts   = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def fetch_bdh(start_str, end_str):
    """
    Pull daily PX_LAST + OPEN_INT for every generic ticker via Bloomberg BDH.
    Returns dict: { bbg_ticker: [ {'date': 'YYYY-MM-DD', 'settle': x, 'open_int': y}, ... ] }
    """
    try:
        import blpapi
    except ImportError:
        log('ERROR: blpapi not installed. Run: pip install blpapi')
        return None

    try:
        opts = blpapi.SessionOptions()
        opts.setServerHost('localhost')
        opts.setServerPort(8194)
        session = blpapi.Session(opts)

        if not session.start():
            log('ERROR: Could not start Bloomberg session.')
            return None
        if not session.openService('//blp/refdata'):
            session.stop()
            log('ERROR: Could not open //blp/refdata.')
            return None

        svc     = session.getService('//blp/refdata')
        results = {}

        # BDH must be sent one ticker at a time
        total = len(TICKERS)
        for idx, (ticker, (commodity, contract)) in enumerate(TICKERS.items(), 1):
            log(f'  [{idx:02d}/{total}] Fetching {ticker} ...')
            req = svc.createRequest('HistoricalDataRequest')
            req.getElement('securities').appendValue(ticker)
            req.getElement('fields').appendValue('PX_LAST')
            req.getElement('fields').appendValue('OPEN_INT')
            req.set('startDate', start_str.replace('-', ''))   # YYYYMMDD
            req.set('endDate',   end_str.replace('-', ''))
            req.set('periodicitySelection', 'DAILY')
            # Only return actual trading days — cleaner data, no fill needed
            req.set('nonTradingDayFillOption', 'ACTIVE_DAYS_ONLY')
            session.sendRequest(req)

            ticker_rows = []
            done = False
            while not done:
                ev = session.nextEvent(12000)
                for msg in ev:
                    if msg.hasElement('securityData'):
                        sd       = msg.getElement('securityData')
                        fd_array = sd.getElement('fieldData')
                        for j in range(fd_array.numValues()):
                            fd    = fd_array.getValue(j)
                            d_val = fd.getElementAsDatetime('date')
                            dt    = f'{d_val.year:04d}-{d_val.month:02d}-{d_val.day:02d}'
                            px    = None
                            oi    = None
                            if fd.hasElement('PX_LAST') and not fd.getElement('PX_LAST').isNull():
                                raw_px = fd.getElementAsFloat('PX_LAST')
                                if raw_px and raw_px > 0:
                                    px = round(float(raw_px), 2)
                            if fd.hasElement('OPEN_INT') and not fd.getElement('OPEN_INT').isNull():
                                raw_oi = fd.getElementAsFloat('OPEN_INT')
                                if raw_oi and raw_oi > 0:
                                    oi = int(raw_oi)
                            if px is not None or oi is not None:
                                ticker_rows.append({'date': dt, 'settle': px, 'open_int': oi})
                if ev.eventType() == blpapi.Event.RESPONSE:
                    done = True

            results[ticker] = ticker_rows
            log(f'        -> {len(ticker_rows)} trading days')

        session.stop()
        return results

    except Exception as e:
        log(f'ERROR: Bloomberg BDH exception: {e}')
        return None


def write_csv(raw):
    """Convert raw BDH results to flat CSV with oi_chg per ticker series."""
    DATA_DIR.mkdir(exist_ok=True)
    all_rows = []

    for ticker, (commodity, contract) in TICKERS.items():
        ticker_rows = raw.get(ticker, [])
        if not ticker_rows:
            log(f'  WARNING: no rows returned for {ticker}')
            continue
        ticker_rows.sort(key=lambda r: r['date'])   # ascending so oi_chg works
        prev_oi = None
        for r in ticker_rows:
            oi     = r['open_int']
            oi_chg = (oi - prev_oi) if (oi is not None and prev_oi is not None) else ''
            all_rows.append({
                'date':       r['date'],
                'commodity':  commodity,
                'contract':   contract,
                'bbg_ticker': ticker,
                'settle':     r['settle']  if r['settle']   is not None else '',
                'open_int':   oi           if oi            is not None else '',
                'oi_chg':     oi_chg,
            })
            if oi is not None:
                prev_oi = oi

    # Sort by date asc, then commodity, then contract label
    all_rows.sort(key=lambda r: (r['date'], r['commodity'], r['contract']))

    with open(DATA_FILE, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    log(f'Wrote {len(all_rows):,} total rows to {DATA_FILE}')
    log(f'Date range in file: {all_rows[0]["date"]} to {all_rows[-1]["date"]}')


def main():
    parser = argparse.ArgumentParser(description='Bootstrap historical OI data from Bloomberg.')
    parser.add_argument('--start', default='2008-01-01',
                        help='Start date YYYY-MM-DD  (default: 2008-01-01)')
    parser.add_argument('--end',   default=date.today().strftime('%Y-%m-%d'),
                        help='End date YYYY-MM-DD  (default: today)')
    args = parser.parse_args()

    log('--- oi_bootstrap.py started ---')
    log(f'Date range : {args.start} to {args.end}')
    log(f'Tickers    : {len(TICKERS)}')
    log(f'Output     : {DATA_FILE}')

    # Back up existing file if there is one
    if DATA_FILE.exists():
        backup = DATA_FILE.with_suffix('.bak')
        shutil.copy(DATA_FILE, backup)
        log(f'Existing CSV backed up to {backup.name}')

    raw = fetch_bdh(args.start, args.end)
    if not raw:
        log('No data returned from Bloomberg. Aborting.')
        return 1

    write_csv(raw)
    log('--- oi_bootstrap.py complete ---')
    log('Next step: start the daily oi_fetcher.py Task Scheduler job.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
