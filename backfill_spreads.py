"""
backfill_spreads.py — one-time 3-year historical fill for spread_ohlc.csv

Pulls PX_LAST / PX_HIGH / PX_LOW / PX_VOLUME for all 20 generic calendar
spread tickers via Bloomberg HistoricalDataRequest and writes
data/spread_ohlc.csv.  Safe to re-run — deduplicates by date+commodity+label.

Run once with Bloomberg Terminal open and logged in:
    python backfill_spreads.py

Output: data/spread_ohlc.csv  (same schema used by vlm_master_fetch.py Job 5)
"""

import csv, pathlib, blpapi
from datetime import date, timedelta

OUT_CSV = pathlib.Path(__file__).parent / 'data' / 'spread_ohlc.csv'
COLUMNS = ['date', 'commodity', 'spread_label', 'settle', 'high', 'low', 'volume']

# 3-year lookback from today
END_DATE   = date.today()
START_DATE = END_DATE - timedelta(days=3 * 365)

# Same 20 tickers as SPREAD_TICKERS in vlm_master_fetch.py
SPREAD_TICKERS = {
    'S:CTCT 1-2 Comdty': ('CT', '1-2'),
    'S:CTCT 2-3 Comdty': ('CT', '2-3'),
    'S:CTCT 3-4 Comdty': ('CT', '3-4'),
    'S:CTCT 4-5 Comdty': ('CT', '4-5'),
    'S:CTCT 5-6 Comdty': ('CT', '5-6'),
    'S:KCKC 1-2 Comdty': ('KC', '1-2'),
    'S:KCKC 2-3 Comdty': ('KC', '2-3'),
    'S:KCKC 3-4 Comdty': ('KC', '3-4'),
    'S:KCKC 4-5 Comdty': ('KC', '4-5'),
    'S:KCKC 5-6 Comdty': ('KC', '5-6'),
    'S:CCCC 1-2 Comdty': ('CC', '1-2'),
    'S:CCCC 2-3 Comdty': ('CC', '2-3'),
    'S:CCCC 3-4 Comdty': ('CC', '3-4'),
    'S:CCCC 4-5 Comdty': ('CC', '4-5'),
    'S:CCCC 5-6 Comdty': ('CC', '5-6'),
    'S:SBSB 1-2 Comdty': ('SB', '1-2'),
    'S:SBSB 2-3 Comdty': ('SB', '2-3'),
    'S:SBSB 3-4 Comdty': ('SB', '3-4'),
    'S:SBSB 4-5 Comdty': ('SB', '4-5'),
    'S:SBSB 5-6 Comdty': ('SB', '5-6'),
}

FIELDS = ['PX_LAST', 'PX_HIGH', 'PX_LOW', 'PX_VOLUME']


def fetch_history(session):
    svc = session.getService('//blp/refdata')
    req = svc.createRequest('HistoricalDataRequest')

    for ticker in SPREAD_TICKERS:
        req.getElement('securities').appendValue(ticker)
    for f in FIELDS:
        req.getElement('fields').appendValue(f)

    req.set('startDate', START_DATE.strftime('%Y%m%d'))
    req.set('endDate',   END_DATE.strftime('%Y%m%d'))
    req.set('periodicitySelection', 'DAILY')

    session.sendRequest(req)

    # results[ticker] = [(date_str, settle, high, low, vol), ...]
    results = {t: [] for t in SPREAD_TICKERS}
    done = False
    while not done:
        ev = session.nextEvent(30000)
        for msg in ev:
            if not msg.hasElement('securityData'):
                continue
            sec_data = msg.getElement('securityData')
            ticker   = sec_data.getElementAsString('security')
            if ticker not in results:
                continue
            if sec_data.hasElement('securityError'):
                print(f'  ERROR: {ticker} — {sec_data.getElement("securityError").getElementAsString("message")}')
                continue
            field_data = sec_data.getElement('fieldData')
            for i in range(field_data.numValues()):
                row = field_data.getValue(i)
                try:
                    d = row.getElementAsDatetime('date')
                    date_str = f'{d.year:04d}-{d.month:02d}-{d.day:02d}'
                except Exception:
                    continue
                def _flt(f):
                    if not row.hasElement(f): return None
                    el = row.getElement(f)
                    return round(float(el.getValueAsFloat()), 4) if not el.isNull() else None
                settle = _flt('PX_LAST')
                if settle is None:
                    continue
                results[ticker].append({
                    'date_str': date_str,
                    'settle':   settle,
                    'high':     _flt('PX_HIGH') or '',
                    'low':      _flt('PX_LOW')  or '',
                    'volume':   int(_flt('PX_VOLUME')) if _flt('PX_VOLUME') else '',
                })
        if ev.eventType() == blpapi.Event.RESPONSE:
            done = True

    return results


def main():
    opts = blpapi.SessionOptions()
    opts.setServerHost('localhost')
    opts.setServerPort(8194)
    session = blpapi.Session(opts)

    if not session.start():
        print('ERROR: Could not start Bloomberg session.')
        return
    if not session.openService('//blp/refdata'):
        print('ERROR: Could not open //blp/refdata.')
        session.stop()
        return

    print(f'Fetching {START_DATE} to {END_DATE} for {len(SPREAD_TICKERS)} spread tickers...')
    results = fetch_history(session)
    session.stop()

    # Build flat rows
    rows = []
    for ticker, (commodity, spread_label) in SPREAD_TICKERS.items():
        ticker_rows = results.get(ticker, [])
        print(f'  {ticker:<26} {len(ticker_rows):>4} rows')
        for r in ticker_rows:
            rows.append({
                'date':         r['date_str'],
                'commodity':    commodity,
                'spread_label': spread_label,
                'settle':       r['settle'],
                'high':         r['high'],
                'low':          r['low'],
                'volume':       r['volume'],
            })

    if not rows:
        print('No data returned.')
        return

    # Dedup if CSV already exists
    existing_keys = set()
    if OUT_CSV.exists():
        with open(OUT_CSV, 'r', encoding='utf-8', newline='') as f:
            for r in csv.DictReader(f):
                existing_keys.add((r['date'], r['commodity'], r['spread_label']))
        rows = [r for r in rows
                if (r['date'], r['commodity'], r['spread_label']) not in existing_keys]
        print(f'  {len(existing_keys)} existing rows; {len(rows)} new rows to append.')

    rows.sort(key=lambda r: (r['date'], r['commodity'], r['spread_label']))

    OUT_CSV.parent.mkdir(exist_ok=True)
    write_header = not OUT_CSV.exists()
    with open(OUT_CSV, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    print(f'\nWrote {len(rows)} rows to {OUT_CSV}')
    print('Next: git add data/spread_ohlc.csv && git commit -m "backfill: 3yr spread OHLC" && git push')


if __name__ == '__main__':
    main()
