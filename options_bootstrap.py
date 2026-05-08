"""
options_bootstrap.py — VLM Options OI Historical Backfill
ONE-TIME script to pull full historical daily OI for all Cotton options
via Bloomberg BDH. Merges into data/options_oi.csv.

Run once from the Open interest dashboard folder:
  python options_bootstrap.py
  python options_bootstrap.py --start 2025-01-01
  python options_bootstrap.py --start 2024-01-01 --end 2026-04-24

Bloomberg Terminal must be open and logged in.
Takes approximately 5-15 minutes depending on date range.
"""

import sys, csv, pathlib, argparse, shutil
from datetime import date, datetime
from collections import defaultdict
try:
    import blpapi
except ImportError:
    blpapi = None

BASE_DIR    = pathlib.Path(__file__).parent
DATA_DIR    = BASE_DIR / 'data'
OPT_FILE    = DATA_DIR / 'options_oi.csv'
LOG_FILE    = BASE_DIR / 'options_bootstrap.log'

CSV_COLUMNS = ['date','security_des','contract_month','put_call',
               'strike_px','open_int','oi_chg','px_settle','px_volume']

_MC = {'F':'Jan','G':'Feb','H':'Mar','J':'Apr','K':'May','M':'Jun',
       'N':'Jul','Q':'Aug','U':'Sep','V':'Oct','X':'Nov','Z':'Dec'}

def parse_contract_month(sec):
    try:
        s = str(sec).strip()
        return f"{_MC.get(s[2],'?')} {2020 + int(s[3])}"
    except Exception:
        return '?'

def log(msg):
    ts   = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def get_option_chain(session):
    """Get all active option tickers via OPT_FUTURES_CHAIN_DATES + OPT_CHAIN."""
    svc = session.getService('//blp/refdata')

    # Step 1: underlying contracts
    req1 = svc.createRequest('ReferenceDataRequest')
    req1.getElement('securities').appendValue('CT1 Comdty')
    req1.getElement('fields').appendValue('OPT_FUTURES_CHAIN_DATES')
    session.sendRequest(req1)
    underlying = []
    done = False
    while not done:
        ev = session.nextEvent(10000)
        for msg in ev:
            if not msg.hasElement('securityData'): continue
            sd = msg.getElement('securityData')
            for i in range(sd.numValues()):
                fd = sd.getValue(i).getElement('fieldData')
                if not fd.hasElement('OPT_FUTURES_CHAIN_DATES'): continue
                chain = fd.getElement('OPT_FUTURES_CHAIN_DATES')
                for j in range(chain.numValues()):
                    try:
                        uc = chain.getValue(j).getElementAsString('Underlying Contract').strip()
                        if uc and uc + ' Comdty' not in underlying:
                            underlying.append(uc + ' Comdty')
                    except: continue
        if ev.eventType() == blpapi.Event.RESPONSE:
            done = True
    log(f'  Underlying contracts: {len(underlying)} — {underlying}')

    # Step 2: option tickers via OPT_CHAIN
    all_tickers = []
    for uc in underlying:
        req2 = svc.createRequest('ReferenceDataRequest')
        req2.getElement('securities').appendValue(uc)
        req2.getElement('fields').appendValue('OPT_CHAIN')
        session.sendRequest(req2)
        done2 = False
        while not done2:
            ev2 = session.nextEvent(10000)
            for msg2 in ev2:
                if not msg2.hasElement('securityData'): continue
                sd2 = msg2.getElement('securityData')
                for i in range(sd2.numValues()):
                    fd2 = sd2.getValue(i).getElement('fieldData')
                    if not fd2.hasElement('OPT_CHAIN'): continue
                    chain2 = fd2.getElement('OPT_CHAIN')
                    for k in range(chain2.numValues()):
                        try:
                            row = chain2.getValue(k)
                            try:
                                t = row.getElementAsString('Security Description').strip()
                            except Exception:
                                t = row.getValueAsString(0).strip()
                            if t:
                                tok = t if 'Comdty' in t else t + ' Comdty'
                                if tok not in all_tickers:
                                    all_tickers.append(tok)
                        except: continue
            if ev2.eventType() == blpapi.Event.RESPONSE:
                done2 = True
        log(f'  {uc}: {len(all_tickers)} tickers cumulative')

    log(f'  Total option tickers: {len(all_tickers)}')
    return all_tickers


def fetch_bdh_options(session, tickers, start_str, end_str):
    """
    Pull daily OPEN_INT + PX_SETTLE + PX_VOLUME for all option tickers via BDH.
    Batches tickers one at a time (BDH requirement).
    Returns dict: {ticker: [{date, open_int, px_settle, px_volume}, ...]}
    """
    svc = session.getService('//blp/refdata')
    results = {}
    total = len(tickers)

    for idx, ticker in enumerate(tickers, 1):
        if idx % 100 == 0 or idx == 1:
            log(f'  BDH [{idx}/{total}] {ticker}')
        req = svc.createRequest('HistoricalDataRequest')
        req.getElement('securities').appendValue(ticker)
        req.getElement('fields').appendValue('OPEN_INT')
        req.getElement('fields').appendValue('PX_SETTLE')
        req.getElement('fields').appendValue('VOLUME')
        req.set('startDate', start_str.replace('-',''))
        req.set('endDate',   end_str.replace('-',''))
        req.set('periodicityAdjustment', 'ACTUAL')
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
                        fd    = fd_array.getValue(j)
                        d_val = fd.getElementAsDatetime('date')
                        dt    = f'{d_val.year:04d}-{d_val.month:02d}-{d_val.day:02d}'
                        oi    = None; sett = None; vol = None
                        if fd.hasElement('OPEN_INT') and not fd.getElement('OPEN_INT').isNull():
                            raw = fd.getElementAsFloat('OPEN_INT')
                            if raw and raw > 0: oi = int(raw)
                        if fd.hasElement('PX_SETTLE') and not fd.getElement('PX_SETTLE').isNull():
                            raw = fd.getElementAsFloat('PX_SETTLE')
                            if raw: sett = round(float(raw), 4)
                        if fd.hasElement('VOLUME') and not fd.getElement('VOLUME').isNull():
                            raw = fd.getElementAsFloat('VOLUME')
                            if raw: vol = int(raw)
                        if oi is not None:
                            ticker_rows.append({'date':dt,'open_int':oi,'px_settle':sett or '','px_volume':vol or 0})
            if ev.eventType() == blpapi.Event.RESPONSE:
                done = True

        if ticker_rows:
            results[ticker] = ticker_rows

    log(f'  BDH complete: {len(results)} tickers with data')
    return results


def build_csv_rows(bdh_results):
    """Convert BDH results to flat CSV rows with oi_chg calculated per ticker."""
    all_rows = []
    for ticker, ticker_rows in bdh_results.items():
        # Parse security_des from ticker e.g. 'CTN6C    80 Comdty' -> 'CTN6C    80'
        sec = ticker.replace(' Comdty','').strip()
        cm  = parse_contract_month(sec)
        # Parse put/call from ticker
        pc = ''
        for ch in sec:
            if ch in ('C','P') and sec.index(ch) > 3:
                pc = ch; break

        ticker_rows.sort(key=lambda r: r['date'])
        prev_oi = None
        for r in ticker_rows:
            oi     = r['open_int']
            oi_chg = (oi - prev_oi) if prev_oi is not None else ''
            all_rows.append({
                'date':           r['date'],
                'security_des':   sec,
                'contract_month': cm,
                'put_call':       pc,
                'strike_px':      '',   # will be blank — strike in security_des
                'open_int':       oi,
                'oi_chg':         oi_chg,
                'px_settle':      r['px_settle'],
                'px_volume':      r['px_volume'],
            })
            prev_oi = oi

    all_rows.sort(key=lambda r: (r['date'], r['security_des']))
    return all_rows


def merge_and_write(new_rows):
    """Merge new rows with existing CSV, dedup by (date, security_des), write."""
    existing = {}
    if OPT_FILE.exists():
        # Backup
        shutil.copy(OPT_FILE, OPT_FILE.with_suffix('.bak'))
        log(f'  Backed up existing file to options_oi.bak')
        with open(OPT_FILE, 'r', encoding='utf-8', newline='') as f:
            for r in csv.DictReader(f):
                key = (r['date'], r['security_des'].strip())
                existing[key] = r

    # Add new rows or update volume on existing rows
    added = 0
    updated = 0
    for r in new_rows:
        key = (r['date'], r['security_des'].strip())
        if key not in existing:
            existing[key] = r
            added += 1
        else:
            # Update volume if new data has it
            new_vol = r.get('px_volume', '')
            if new_vol and str(new_vol) != '0':
                existing[key]['px_volume'] = new_vol
                updated += 1

    all_rows = sorted(existing.values(), key=lambda r: (r['date'], r['security_des']))
    DATA_DIR.mkdir(exist_ok=True)
    with open(OPT_FILE, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(all_rows)

    log(f'  Written {len(all_rows):,} total rows ({added:,} new, {updated:,} volume updated) to {OPT_FILE.name}')
    if all_rows:
        dates = sorted({r['date'] for r in all_rows})
        log(f'  Date range: {dates[0]} to {dates[-1]}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2025-01-01')
    parser.add_argument('--end',   default=date.today().strftime('%Y-%m-%d'))
    args = parser.parse_args()

    log('--- options_bootstrap.py started ---')
    log(f'Date range: {args.start} to {args.end}')

    if not blpapi:
        log('ERROR: blpapi not installed'); return 1

    opts = blpapi.SessionOptions()
    opts.setServerHost('localhost'); opts.setServerPort(8194)
    session = blpapi.Session(opts)
    if not session.start():
        log('ERROR: Bloomberg session failed'); return 1
    if not session.openService('//blp/refdata'):
        session.stop(); log('ERROR: //blp/refdata failed'); return 1

    try:
        tickers = get_option_chain(session)
        if not tickers:
            log('ERROR: No option tickers found'); return 1

        bdh = fetch_bdh_options(session, tickers, args.start, args.end)
        rows = build_csv_rows(bdh)
        merge_and_write(rows)

    finally:
        session.stop()
        log('Bloomberg session closed.')

    log('--- options_bootstrap.py complete ---')
    return 0

if __name__ == '__main__':
    sys.exit(main())
