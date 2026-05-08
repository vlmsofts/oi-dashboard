"""
oi_bootstrap.py — VLM Open Interest Monitor
ONE-TIME script to pull full historical daily OI + settle from Bloomberg
using BDH (historical data) with month-specific generic rolling tickers.
Writes data/oi_data.csv from scratch — run this ONCE before starting
the daily vlm_master_fetch.py Task Scheduler job.

Bloomberg Terminal must be open and logged in.

Usage (in PowerShell):
  python oi_bootstrap.py
  python oi_bootstrap.py --start 2015-01-01
  python oi_bootstrap.py --start 2010-01-01 --end 2025-04-20
"""

import sys, csv, pathlib, argparse, shutil
from datetime import date, datetime

BASE_DIR  = pathlib.Path(__file__).parent
DATA_DIR  = BASE_DIR / 'data'
DATA_FILE = DATA_DIR / 'oi_data.csv'
LOG_FILE  = BASE_DIR / 'oi_bootstrap.log'

CSV_COLUMNS = ['date', 'commodity', 'contract', 'bbg_ticker',
               'settle', 'open_int', 'oi_chg', 'first_notice', 'last_trade']

# Month-specific generic tickers — Bloomberg rolls these automatically.
# CTJUL1 is always the front July contract, CTJUL2 always the second July, etc.
# No numbered slot generics (CT1-CT7) — those overlap and distort history.
TICKERS = {
    # ── Cotton No.2  (Mar/May/Jul/Oct/Dec × 2) ────────────────────────────────
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
    # ── Sugar No.11  (Mar/May/Jul/Oct × 2) ────────────────────────────────────
    'SBMAR1 Comdty': ('SB', 'SBMAR1'),
    'SBMAY1 Comdty': ('SB', 'SBMAY1'),
    'SBJUL1 Comdty': ('SB', 'SBJUL1'),
    'SBOCT1 Comdty': ('SB', 'SBOCT1'),
    'SBMAR2 Comdty': ('SB', 'SBMAR2'),
    'SBMAY2 Comdty': ('SB', 'SBMAY2'),
    'SBJUL2 Comdty': ('SB', 'SBJUL2'),
    'SBOCT2 Comdty': ('SB', 'SBOCT2'),
    # ── Coffee C  (Mar/May/Jul/Sep/Dec × 2) ───────────────────────────────────
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
    # ── Cocoa  (Mar/May/Jul/Sep/Dec × 2) ──────────────────────────────────────
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


def log(msg):
    ts   = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def fetch_bdh(start_str, end_str):
    """
    Pull daily PX_LAST + OPEN_INT for every ticker via Bloomberg BDH.
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
        total   = len(TICKERS)

        for idx, (ticker, (commodity, contract)) in enumerate(TICKERS.items(), 1):
            log(f'  [{idx:02d}/{total}] Fetching {ticker} ...')
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
                ev = session.nextEvent(12000)
                for msg in ev:
                    if msg.hasElement('securityData'):
                        sd       = msg.getElement('securityData')
                        fd_array = sd.getElement('fieldData')
                        for j in range(fd_array.numValues()):
                            fd = fd_array.getValue(j)
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
                                    v = fd.getElementAsDatetime('FUT_NOTICE_FIRST')
                                    fn = f'{v.year:04d}-{v.month:02d}-{v.day:02d}'
                                except Exception:
                                    pass
                            if fd.hasElement('LAST_TRADEABLE_DT') and not fd.getElement('LAST_TRADEABLE_DT').isNull():
                                try:
                                    v = fd.getElementAsDatetime('LAST_TRADEABLE_DT')
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

            results[ticker] = ticker_rows
            log(f'        -> {len(ticker_rows)} trading days')

        session.stop()
        return results

    except Exception as e:
        log(f'ERROR: Bloomberg BDH exception: {e}')
        return None


def write_csv(raw):
    """Convert raw BDH results to flat CSV with oi_chg calculated per ticker series."""
    DATA_DIR.mkdir(exist_ok=True)
    all_rows = []

    for ticker, (commodity, contract) in TICKERS.items():
        ticker_rows = raw.get(ticker, [])
        if not ticker_rows:
            log(f'  WARNING: no rows for {ticker}')
            continue
        ticker_rows.sort(key=lambda r: r['date'])
        prev_oi    = None
        prev_contract = None  # track contract code to null oi_chg on roll
        for r in ticker_rows:
            oi     = r['open_int']
            oi_chg = (oi - prev_oi) if (oi is not None and prev_oi is not None) else ''
            all_rows.append({
                'date':         r['date'],
                'commodity':    commodity,
                'contract':     contract,   # e.g. 'CTJUL1' — the Bloomberg ticker label
                'bbg_ticker':   ticker,
                'settle':       r['settle']       if r['settle']       is not None else '',
                'open_int':     oi                if oi                is not None else '',
                'oi_chg':       oi_chg,
                'first_notice': r['first_notice'],
                'last_trade':   r['last_trade'],
            })
            if oi is not None:
                prev_oi = oi

    all_rows.sort(key=lambda r: (r['date'], r['commodity'], r['contract']))

    with open(DATA_FILE, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    log(f'Wrote {len(all_rows):,} total rows to {DATA_FILE}')
    if all_rows:
        log(f'Date range in file: {all_rows[0]["date"]} to {all_rows[-1]["date"]}')


def main():
    parser = argparse.ArgumentParser(description='Bootstrap historical OI data from Bloomberg.')
    parser.add_argument('--start', default='2008-01-01')
    parser.add_argument('--end',   default=date.today().strftime('%Y-%m-%d'))
    args = parser.parse_args()

    log('--- oi_bootstrap.py started ---')
    log(f'Date range : {args.start} to {args.end}')
    log(f'Tickers    : {len(TICKERS)}')
    log(f'Output     : {DATA_FILE}')

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
    log('Next step: run vlm_master_fetch.py daily via Task Scheduler.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
