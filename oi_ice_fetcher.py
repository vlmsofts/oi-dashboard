"""
oi_ice_fetcher.py — ICE RTD workbook reader for OI dashboard shadow pipeline.

Reads ICE RTD Excel workbooks (via xlwings COM) and returns data in the same
format used by the Bloomberg CSV pipeline:
  - futures rows : list of dicts matching oi_data.csv schema
  - options rows : list of dicts matching options_oi.csv schema

Workbook location:
  C:\\Users\\Louis\\OneDrive - VLM Commodities LTD\\Site Sync\\ICE RTD FEED {COMM}.xlsx

The workbook must be open in Excel with ICE RTD feed active.
Returns (None, None) if unavailable — caller falls back to Bloomberg.

Column layout of ICE option sheets (0-based, confirmed from workbook):
  Header: Qty Bid Offer Qty Last Volume OptBlock Settlement OI Strike
          Qty Bid Offer Qty Last Volume OptBlock Settlement OI
  Index:   0   1   2    3   4     5       6          7       8   9
           10  11  12  13  14    15      16         17      18
"""

import math
import sys
from datetime import date
from pathlib import Path

try:
    import xlwings as xw
    _XW_OK = True
except ImportError:
    _XW_OK = False

from contract_dates import (
    get_fnd, get_ltd, get_bbg_slot, contract_month_label, ice_to_bbg, _D
)

SITE_SYNC = Path(r'C:\Users\Louis\OneDrive - VLM Commodities LTD\Site Sync')

WB_NAME = {
    'CT': 'ICE RTD FEED CT.xlsx',
    'KC': 'ICE RTD FEED KC.xlsx',
    'SB': 'ICE RTD FEED SB.xlsx',
    'CC': 'ICE RTD FEED CC.xlsx',
}

# Month code helpers
_MON_NUM  = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
             'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
_MON_CODE = {1:'F',2:'G',3:'H',4:'J',5:'K',6:'M',
             7:'N',8:'Q',9:'U',10:'V',11:'X',12:'Z'}
_CODE_MON = {v: k for k, v in _MON_CODE.items()}

# Option sheet column indices
_C_BID    =  1
_C_OFFER  =  2
_C_LAST   =  4
_C_VOL    =  5
_C_SETTLE =  7
_C_OI     =  8
_STRIKE   =  9
_P_BID    = 11
_P_OFFER  = 12
_P_LAST   = 14
_P_VOL    = 15
_P_SETTLE = 17
_P_OI     = 18

# Futures sheet fallback column positions
_FUT_COLS = {
    'Strip': 2, 'bid': 6, 'offer': 7,
    'Last Price': 9, 'Settle': 17, 'Market State': 19, 'OI': 46,
}


# ── Utilities ────────────────────────────────────────────────────────────────

def _safe_float(val):
    if val is None:
        return None
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _safe_int(val):
    f = _safe_float(val)
    return int(f) if f is not None else None


def _strip_to_ice(strip_str, prefix):
    """'Jul26' → 'CTN6',  'Dec27' → 'CTZ7'"""
    if not strip_str or not isinstance(strip_str, str):
        return None
    s = strip_str.strip()
    if len(s) < 4:
        return None
    mon_str  = s[:3]
    yr_digit = s[-1]
    month_num = _MON_NUM.get(mon_str)
    if month_num is None:
        return None
    mc = _MON_CODE[month_num]
    return f'{prefix.upper()}{mc}{yr_digit}'


def _contract_sort_key(ice_code, prefix):
    code = ice_code[len(prefix):]
    if len(code) < 2:
        return (99, 99)
    yr  = int(code[1]) if code[1].isdigit() else 99
    mon = _CODE_MON.get(code[0], 99)
    return (yr, mon)


# ── Workbook discovery ───────────────────────────────────────────────────────

def _find_workbook(wb_filename):
    if not _XW_OK:
        return None
    try:
        for app in xw.apps:
            for wb in app.books:
                if wb.name.lower() == wb_filename.lower():
                    return wb
    except Exception:
        pass
    return None


def _open_wb(commodity):
    name = WB_NAME.get(commodity.upper())
    return _find_workbook(name) if name else None


# ── Futures reader ───────────────────────────────────────────────────────────

def _read_futures(wb, prefix):
    sheet_name = f'{prefix.upper()} Futures'
    sh = None
    try:
        sh = wb.sheets[sheet_name]
    except Exception:
        for s in wb.sheets:
            if s.name.strip().lower() == sheet_name.lower():
                sh = s
                break
    if sh is None:
        return {}

    data = sh.used_range.value
    if not data or len(data) < 2:
        return {}

    header = [str(h).strip() if h is not None else '' for h in data[0]]

    def _col(name, fallback):
        try:
            return header.index(name)
        except ValueError:
            return fallback

    col_strip  = _col('Strip',        _FUT_COLS['Strip'])
    col_last   = _col('Last Price',   _FUT_COLS['Last Price'])
    col_settle = _col('Settle',       _FUT_COLS['Settle'])
    col_mstate = _col('Market State', _FUT_COLS['Market State'])
    col_oi     = _col('OI',           _FUT_COLS['OI'])
    col_bid    = _FUT_COLS['bid']
    col_offer  = _FUT_COLS['offer']

    result = {}
    for row in data[1:]:
        if not row or len(row) <= col_settle:
            continue
        product = str(row[0]).strip() if row[0] else ''
        if 'Futures' not in product:
            continue
        strip_val = row[col_strip] if len(row) > col_strip else None
        if strip_val and '/' in str(strip_val):
            continue

        ice_code = _strip_to_ice(strip_val, prefix)
        if not ice_code or ice_code in result:
            continue

        result[ice_code] = {
            'settle': _safe_float(row[col_settle]) if len(row) > col_settle else None,
            'last':   _safe_float(row[col_last])   if len(row) > col_last   else None,
            'bid':    _safe_float(row[col_bid])    if len(row) > col_bid    else None,
            'offer':  _safe_float(row[col_offer])  if len(row) > col_offer  else None,
            'oi':     _safe_float(row[col_oi])     if len(row) > col_oi     else None,
        }
    return result


# ── Options reader ───────────────────────────────────────────────────────────

def _option_sheets(wb, prefix):
    result = []
    plen = len(prefix)
    for sh in wb.sheets:
        name = sh.name.strip().upper()
        if (name.startswith(prefix.upper())
                and len(name) == plen + 2
                and name[plen:].isalnum()):
            result.append(name)
    result.sort(key=lambda c: _contract_sort_key(c, prefix))
    return result


def _read_options(wb, sheet_name):
    try:
        sh = wb.sheets[sheet_name]
    except Exception:
        return []

    data = sh.used_range.value
    if not data or len(data) < 2:
        return []

    rows = []
    for row in data[1:]:
        if not row or len(row) <= _P_OI:
            continue
        strike = _safe_float(row[_STRIKE])
        if strike is None:
            continue
        rows.append({
            'strike':      strike,
            'call_settle': _safe_float(row[_C_SETTLE]),
            'call_oi':     _safe_int(row[_C_OI]),
            'call_vol':    _safe_int(row[_C_VOL]),
            'put_settle':  _safe_float(row[_P_SETTLE]),
            'put_oi':      _safe_int(row[_P_OI]),
            'put_vol':     _safe_int(row[_P_VOL]),
        })
    rows.sort(key=lambda r: r['strike'])
    return rows


# ── Build CSV rows ───────────────────────────────────────────────────────────

def _build_futures_rows(prefix, futures_raw, today_str, prev_oi):
    """
    Returns list of dicts matching oi_data.csv schema.
    prev_oi : {ice_code: open_int} from yesterday's shadow CSV for oi_chg.
    """
    rows = []
    for ice_code, data in futures_raw.items():
        bbg_slot = get_bbg_slot(ice_code)
        if bbg_slot is None:
            continue  # contract not in current BBG slot mapping — skip

        settle = data.get('settle')
        oi     = _safe_int(data.get('oi'))
        if settle is None or oi is None:
            continue

        fnd = get_fnd(ice_code)
        ltd = get_ltd(ice_code)
        prev = prev_oi.get(ice_code)
        oi_chg = (oi - prev) if (prev is not None) else 0

        rows.append({
            'date':         today_str,
            'commodity':    prefix.upper(),
            'contract':     bbg_slot,
            'bbg_ticker':   f'{bbg_slot} Comdty',
            'settle':       f'{settle:.4f}',
            'open_int':     str(oi),
            'oi_chg':       str(oi_chg),
            'first_notice': str(fnd) if fnd else '',
            'last_trade':   str(ltd) if ltd else '',
        })
    return rows


def _build_options_rows(prefix, options_raw, today_str, prev_opt_oi):
    """
    Returns list of dicts matching options_oi.csv schema.
    options_raw : {ice_code: [strike_row, ...]}
    prev_opt_oi : {(ice_code, pc, strike): open_int} from yesterday's shadow CSV.
    """
    rows = []
    for ice_code, strikes in options_raw.items():
        month_label = contract_month_label(ice_code)
        for sr in strikes:
            strike = sr['strike']
            for pc, oi_key, settle_key, vol_key in [
                ('C', 'call_oi', 'call_settle', 'call_vol'),
                ('P', 'put_oi',  'put_settle',  'put_vol'),
            ]:
                oi  = sr.get(oi_key)
                if oi is None:
                    continue
                settle = sr.get(settle_key)
                vol    = sr.get(vol_key) or 0
                prev   = prev_opt_oi.get((ice_code, pc, strike))
                oi_chg = (oi - prev) if (prev is not None) else 0

                # Format strike: drop .0 for whole numbers
                strike_str = str(int(strike)) if strike == int(strike) else str(strike)
                sec_des    = f'{ice_code}{pc}    {strike_str}'

                rows.append({
                    'date':           today_str,
                    'commodity':      prefix.upper(),
                    'security_des':   sec_des,
                    'contract_month': month_label,
                    'put_call':       pc,
                    'strike_px':      strike_str,
                    'open_int':       str(oi),
                    'oi_chg':         str(oi_chg),
                    'px_settle':      f'{settle:.4f}' if settle is not None else '',
                    'px_volume':      str(vol),
                })
    return rows


# ── Previous OI loaders ──────────────────────────────────────────────────────

def _load_prev_futures_oi(shadow_csv: Path, prefix: str) -> dict:
    """Returns {ice_code: open_int} for the latest date in shadow CSV."""
    if not shadow_csv.exists():
        return {}
    import csv as _csv
    rows = list(_csv.DictReader(shadow_csv.open(encoding='utf-8')))
    if not rows:
        return {}
    comm_rows = [r for r in rows if r.get('commodity') == prefix.upper()]
    if not comm_rows:
        return {}
    latest = max(r['date'] for r in comm_rows)
    result = {}
    for r in comm_rows:
        if r['date'] != latest:
            continue
        bbg_slot = r.get('contract', '')
        # Convert BBG slot back to ICE code
        from contract_dates import _BBG_TO_ICE
        ice_code = _BBG_TO_ICE.get(bbg_slot.upper())
        if ice_code:
            try:
                result[ice_code] = int(r['open_int'])
            except (ValueError, TypeError):
                pass
    return result


def _load_prev_options_oi(shadow_csv: Path, prefix: str) -> dict:
    """Returns {(ice_code, pc, strike_float): open_int} for latest date."""
    if not shadow_csv.exists():
        return {}
    import csv as _csv
    rows = list(_csv.DictReader(shadow_csv.open(encoding='utf-8')))
    if not rows:
        return {}
    comm_rows = [r for r in rows if r.get('commodity') == prefix.upper()]
    if not comm_rows:
        return {}
    latest = max(r['date'] for r in comm_rows)
    result = {}
    for r in comm_rows:
        if r['date'] != latest:
            continue
        sec = r.get('security_des', '')
        pc  = r.get('put_call', '')
        try:
            strike = float(r.get('strike_px', ''))
            oi     = int(r['open_int'])
            # ICE code is the part before C or P in security_des (e.g. KCN6 from 'KCN6P    212.5')
            ice_code = sec.split(pc)[0] if pc in sec else ''
            if ice_code:
                result[(ice_code.upper(), pc, strike)] = oi
        except (ValueError, TypeError):
            pass
    return result


# ── Main entry point ─────────────────────────────────────────────────────────

def fetch_ice_oi(commodity: str, today_str: str,
                 shadow_oi_csv: Path, shadow_opt_csv: Path):
    """
    Reads ICE RTD workbook for one commodity.
    Returns (futures_rows, options_rows) — lists of CSV-ready dicts.
    Returns (None, None) if workbook unavailable.

    commodity     : 'KC' or 'CC' (or 'CT'/'SB')
    today_str     : 'YYYY-MM-DD'
    shadow_oi_csv : path to shadow oi_data CSV (for prev-day oi_chg calc)
    shadow_opt_csv: path to shadow options CSV  (for prev-day oi_chg calc)
    """
    try:
        import pythoncom
        pythoncom.CoInitialize()
        try:
            return _fetch_ice_oi_inner(commodity, today_str, shadow_oi_csv, shadow_opt_csv)
        finally:
            pythoncom.CoUninitialize()
    except ImportError:
        return _fetch_ice_oi_inner(commodity, today_str, shadow_oi_csv, shadow_opt_csv)


def _fetch_ice_oi_inner(commodity, today_str, shadow_oi_csv, shadow_opt_csv):
    prefix = commodity.upper()
    wb = _open_wb(prefix)
    if wb is None:
        return None, None

    futures_raw = _read_futures(wb, prefix)
    if not futures_raw:
        return None, None

    opt_sheets  = _option_sheets(wb, prefix)
    options_raw = {}
    for sheet in opt_sheets:
        rows = _read_options(wb, sheet)
        if rows:
            options_raw[sheet.upper()] = rows

    prev_fut = _load_prev_futures_oi(shadow_oi_csv, prefix)
    prev_opt = _load_prev_options_oi(shadow_opt_csv, prefix)

    futures_rows = _build_futures_rows(prefix, futures_raw, today_str, prev_fut)
    options_rows = _build_options_rows(prefix, options_raw, today_str, prev_opt)

    return futures_rows, options_rows


if __name__ == '__main__':
    # Quick diagnostic — print what ICE RTD would write for KC and CC today
    from datetime import date as _date
    today = _date.today().strftime('%Y-%m-%d')
    base  = Path(__file__).parent
    shadow_oi  = base / 'data' / 'oi_data_shadow.csv'
    shadow_opt = base / 'data' / 'options_oi_shadow.csv'

    for comm in ['KC', 'CC']:
        print(f'\n{"="*60}')
        print(f'  ICE RTD fetch — {comm}  ({today})')
        print('='*60)
        fut, opts = fetch_ice_oi(comm, today, shadow_oi, shadow_opt)
        if fut is None:
            print(f'  UNAVAILABLE — workbook not open or xlwings missing')
            continue
        print(f'  Futures rows : {len(fut)}')
        for r in fut:
            print(f"    {r['contract']:12}  settle={r['settle']:>8}  oi={r['open_int']:>8}  chg={r['oi_chg']:>7}  fnd={r['first_notice']}")
        print(f'  Options rows : {len(opts)}')
        if opts:
            sample = opts[:3]
            for r in sample:
                print(f"    {r['security_des']:25}  oi={r['open_int']:>6}  chg={r['oi_chg']:>6}  settle={r['px_settle']}")
