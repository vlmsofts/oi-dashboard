"""
contract_dates.py — ICE-verified contract expiry dates for CT, KC, CC, SB.

LTD : Last Trading Date — confirmed from ICE.com expiry pages (2026-05-22)
FND : First Notice Date  — confirmed from Bloomberg FUT_NOTICE_FIRST (2026-05-22)

ICE expiry page URLs:
  Futures  CT /products/254  KC /products/15   CC /products/7    SB /products/23
  Options  CT /products/1027 KC /products/14   CC /products/8    SB /products/22

Usage:
    from contract_dates import get_fnd, get_ltd, get_bbg_slot, ice_to_bbg

    get_fnd('CTN6')      -> date(2026, 6, 24)
    get_ltd('CTJUL1')    -> date(2026, 7,  9)   # BBG generic slot accepted
    get_bbg_slot('CTN6') -> 'CTJUL1'
    ice_to_bbg           # full reverse mapping dict
"""

from datetime import date

# ── ICE contract code → {fnd, ltd} ───────────────────────────────────────────
# Key: {COMM}{MONTH_CODE}{YEAR_DIGIT}
#   Month codes: F=Jan G=Feb H=Mar J=Apr K=May M=Jun N=Jul Q=Aug U=Sep V=Oct X=Nov Z=Dec
#   Year digit : last digit of year  (6=2026, 7=2027, 8=2028, 9=2029)
#
# Sugar (SB): LTD falls in the month BEFORE contract month — this is correct.
#   SBN6 trades until Jun 30, first notice Jul 1 (delivery month = Jul).

_D = {
    # ── Cotton No. 2 ─────────────────────────────────────────────────────────
    # opt_exp: last Friday >= 5 business days before FND (ICE spec, ProductSpec_1027)
    #          CONFIRMED vs ICE calendar (2026-06-03)
    'CTN6': {'fnd': date(2026,  6, 24), 'ltd': date(2026,  7,  9), 'opt_exp': date(2026,  6, 12)},
    'CTV6': {'fnd': date(2026,  9, 24), 'ltd': date(2026, 10,  8), 'opt_exp': date(2026,  9, 11)},
    'CTZ6': {'fnd': date(2026, 11, 23), 'ltd': date(2026, 12,  8), 'opt_exp': date(2026, 11, 13)},
    'CTH7': {'fnd': date(2027,  2, 22), 'ltd': date(2027,  3,  8), 'opt_exp': date(2027,  2, 12)},
    'CTK7': {'fnd': date(2027,  4, 26), 'ltd': date(2027,  5,  6), 'opt_exp': date(2027,  4, 16)},
    'CTN7': {'fnd': date(2027,  6, 24), 'ltd': date(2027,  7,  8), 'opt_exp': date(2027,  6, 11)},
    'CTV7': {'fnd': date(2027,  9, 24), 'ltd': date(2027, 10,  7), 'opt_exp': date(2027,  9, 17)},
    'CTZ7': {'fnd': date(2027, 11, 23), 'ltd': date(2027, 12,  8), 'opt_exp': date(2027, 11, 12)},
    'CTH8': {'fnd': date(2028,  2, 23), 'ltd': date(2028,  3,  9), 'opt_exp': date(2028,  2, 11)},
    'CTK8': {'fnd': date(2028,  4, 24), 'ltd': date(2028,  5,  8), 'opt_exp': date(2028,  4,  7)},

    # ── Coffee C ─────────────────────────────────────────────────────────────
    # opt_exp: 2nd Friday of month preceding delivery, min 4 biz days to FND
    #          holidays excluded from biz day count — CONFIRMED (2026-06-03)
    #          KCH7: 2nd Fri (Feb 12) unusable due to Presidents Day, -> Feb 11 (Thu)
    'KCN6': {'fnd': date(2026,  6, 22), 'ltd': date(2026,  7, 21), 'opt_exp': date(2026,  6, 12)},
    'KCU6': {'fnd': date(2026,  8, 21), 'ltd': date(2026,  9, 18), 'opt_exp': date(2026,  8, 14)},
    'KCZ6': {'fnd': date(2026, 11, 19), 'ltd': date(2026, 12, 18), 'opt_exp': date(2026, 11, 13)},
    'KCH7': {'fnd': date(2027,  2, 18), 'ltd': date(2027,  3, 18), 'opt_exp': date(2027,  2, 11)},
    'KCK7': {'fnd': date(2027,  4, 22), 'ltd': date(2027,  5, 18), 'opt_exp': date(2027,  4,  9)},
    'KCN7': {'fnd': date(2027,  6, 22), 'ltd': date(2027,  7, 20), 'opt_exp': date(2027,  6, 11)},
    'KCU7': {'fnd': date(2027,  8, 23), 'ltd': date(2027,  9, 20), 'opt_exp': date(2027,  8, 13)},
    'KCZ7': {'fnd': date(2027, 11, 19), 'ltd': date(2027, 12, 20), 'opt_exp': date(2027, 11, 12)},
    'KCH8': {'fnd': date(2028,  2, 18), 'ltd': date(2028,  3, 21), 'opt_exp': date(2028,  2, 11)},

    # ── Cocoa ─────────────────────────────────────────────────────────────────
    # opt_exp: 2nd Friday of month preceding delivery — CONFIRMED vs ICE /products/8 (2026-05-22)
    'CCN6': {'fnd': date(2026,  6, 24), 'ltd': date(2026,  7, 16), 'opt_exp': date(2026,  6, 12)},
    'CCU6': {'fnd': date(2026,  8, 25), 'ltd': date(2026,  9, 15), 'opt_exp': date(2026,  8, 14)},
    'CCZ6': {'fnd': date(2026, 11, 23), 'ltd': date(2026, 12, 15), 'opt_exp': date(2026, 11, 13)},
    'CCH7': {'fnd': date(2027,  2, 22), 'ltd': date(2027,  3, 15), 'opt_exp': date(2027,  2, 12)},
    'CCK7': {'fnd': date(2027,  4, 26), 'ltd': date(2027,  5, 13), 'opt_exp': date(2027,  4,  9)},
    'CCN7': {'fnd': date(2027,  6, 24), 'ltd': date(2027,  7, 15), 'opt_exp': date(2027,  6, 11)},
    'CCU7': {'fnd': date(2027,  8, 25), 'ltd': date(2027,  9, 15), 'opt_exp': date(2027,  8, 13)},
    'CCZ7': {'fnd': date(2027, 11, 23), 'ltd': date(2027, 12, 15), 'opt_exp': date(2027, 11, 12)},
    'CCH8': {'fnd': date(2028,  2, 23), 'ltd': date(2028,  3, 16), 'opt_exp': date(2028,  2, 11)},

    # ── Sugar No. 11 ─────────────────────────────────────────────────────────
    # opt_exp: 15th of month preceding delivery (adjusted for weekends/holidays)
    #          CONFIRMED vs ICE /products/22 (2026-05-22)
    # Note: SB FND is in the delivery month; LTD is last day of prior month.
    'SBN6': {'fnd': date(2026,  7,  1), 'ltd': date(2026,  6, 30), 'opt_exp': date(2026,  6, 15)},
    'SBV6': {'fnd': date(2026, 10,  1), 'ltd': date(2026,  9, 30), 'opt_exp': date(2026,  9, 15)},
    'SBH7': {'fnd': date(2027,  3,  1), 'ltd': date(2027,  2, 26), 'opt_exp': date(2027,  2, 16)},
    'SBK7': {'fnd': date(2027,  5,  3), 'ltd': date(2027,  4, 30), 'opt_exp': date(2027,  4, 15)},
    'SBN7': {'fnd': date(2027,  7,  1), 'ltd': date(2027,  6, 30), 'opt_exp': date(2027,  6, 15)},
    'SBV7': {'fnd': date(2027, 10,  1), 'ltd': date(2027,  9, 30), 'opt_exp': date(2027,  9, 15)},
    'SBH8': {'fnd': date(2028,  3,  1), 'ltd': date(2028,  2, 29), 'opt_exp': date(2028,  2, 15)},
    'SBK8': {'fnd': date(2028,  5,  1), 'ltd': date(2028,  4, 28), 'opt_exp': date(2028,  4, 17)},
}

# ── Bloomberg generic slot → ICE code ────────────────────────────────────────
# Valid as of 2026-05-22.  Update when a contract rolls off (nearest slot
# advances to the next year's expiry — typically 2-4 weeks before LTD).

_BBG_TO_ICE = {
    # Cotton
    'CTJUL1': 'CTN6',  'CTJUL2': 'CTN7',
    'CTOCT1': 'CTV6',  'CTOCT2': 'CTV7',
    'CTDEC1': 'CTZ6',  'CTDEC2': 'CTZ7',
    'CTMAR1': 'CTH7',  'CTMAR2': 'CTH8',
    'CTMAY1': 'CTK7',  'CTMAY2': 'CTK8',
    # Coffee
    'KCJUL1': 'KCN6',  'KCJUL2': 'KCN7',
    'KCSEP1': 'KCU6',  'KCSEP2': 'KCU7',
    'KCDEC1': 'KCZ6',  'KCDEC2': 'KCZ7',
    'KCMAR1': 'KCH7',  'KCMAR2': 'KCH8',
    'KCMAY1': 'KCK7',
    # Cocoa
    'CCJUL1': 'CCN6',  'CCJUL2': 'CCN7',
    'CCSEP1': 'CCU6',  'CCSEP2': 'CCU7',
    'CCDEC1': 'CCZ6',  'CCDEC2': 'CCZ7',
    'CCMAR1': 'CCH7',  'CCMAR2': 'CCH8',
    'CCMAY1': 'CCK7',
    # Sugar
    'SBJUL1': 'SBN6',  'SBJUL2': 'SBN7',
    'SBOCT1': 'SBV6',  'SBOCT2': 'SBV7',
    'SBMAR1': 'SBH7',  'SBMAR2': 'SBH8',
    'SBMAY1': 'SBK7',  'SBMAY2': 'SBK8',
}

# Reverse map: ICE code → BBG generic slot
ice_to_bbg = {v: k for k, v in _BBG_TO_ICE.items()}


# ── Contract month label  (KCN6 → "Jul 2026") ───────────────────────────────
_MON_CODE_TO_NAME = {
    'F': 'Jan', 'G': 'Feb', 'H': 'Mar', 'J': 'Apr', 'K': 'May', 'M': 'Jun',
    'N': 'Jul', 'Q': 'Aug', 'U': 'Sep', 'V': 'Oct', 'X': 'Nov', 'Z': 'Dec',
}

def contract_month_label(ice_code: str) -> str:
    """'KCN6' → 'Jul 2026',  'CTZ7' → 'Dec 2027'"""
    try:
        mc  = ice_code[-2]
        yr  = int(ice_code[-1])
        mon = _MON_CODE_TO_NAME.get(mc.upper(), '???')
        return f'{mon} {2020 + yr}'
    except Exception:
        return ''


# ── Public lookup functions ──────────────────────────────────────────────────

def _resolve(code: str):
    """Accept ICE contract code or BBG generic slot (with/without ' Comdty')."""
    upper = code.upper().split()[0]
    if upper in _D:
        return _D[upper]
    ice = _BBG_TO_ICE.get(upper)
    if ice:
        return _D.get(ice)
    return None


def get_dates(code: str) -> dict | None:
    """Return {'fnd': date, 'ltd': date} or None."""
    return _resolve(code)


def get_fnd(code: str) -> date | None:
    """First Notice Date — accepts ICE code or BBG slot."""
    d = _resolve(code)
    return d['fnd'] if d else None


def get_ltd(code: str) -> date | None:
    """Last Trading Date — accepts ICE code or BBG slot."""
    d = _resolve(code)
    return d['ltd'] if d else None


def get_opt_exp(code: str) -> date | None:
    """Option expiry date — accepts ICE code or BBG slot.
    All four commodities CONFIRMED vs ICE expiry calendar pages (2026-06-03).
    CT: last Friday >= 5 biz days before FND (ICE ProductSpec_1027).
    KC/CC: 2nd Friday of month preceding delivery.
    SB: 15th of month preceding delivery (adjusted for weekends/holidays).
    """
    d = _resolve(code)
    return d.get('opt_exp') if d else None


def get_bbg_slot(ice_code: str) -> str | None:
    """Translate ICE contract code to Bloomberg generic slot.
    'CTN6' -> 'CTJUL1',  'KCZ6' -> 'KCDEC1'
    """
    return ice_to_bbg.get(ice_code.upper())


if __name__ == '__main__':
    # Quick smoke-test
    tests = [
        ('CTN6',   '2026-06-24', '2026-07-09', '2026-06-12', 'CTJUL1'),
        ('CTJUL1', '2026-06-24', '2026-07-09', '2026-06-12', None),
        ('KCZ6',   '2026-11-19', '2026-12-18', '2026-11-13', 'KCDEC1'),
        ('CCN6',   '2026-06-24', '2026-07-16', '2026-06-12', 'CCJUL1'),
        ('SBN6',   '2026-07-01', '2026-06-30', '2026-06-15', 'SBJUL1'),
    ]
    all_pass = True
    for code, exp_fnd, exp_ltd, exp_opt, exp_bbg in tests:
        fnd = get_fnd(code)
        ltd = get_ltd(code)
        opt = get_opt_exp(code)
        bbg = get_bbg_slot(code) if len(code) <= 5 else None
        ok  = (str(fnd) == exp_fnd and str(ltd) == exp_ltd
               and str(opt) == exp_opt
               and (exp_bbg is None or bbg == exp_bbg))
        status = 'PASS' if ok else 'FAIL'
        if not ok:
            all_pass = False
        print(f'  {status}  {code:10}  fnd={fnd}  ltd={ltd}  opt_exp={opt}  bbg={bbg}')
    print('\nAll tests passed.' if all_pass else '\nSOME TESTS FAILED.')
