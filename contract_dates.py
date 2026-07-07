"""
contract_dates.py -- ICE-verified contract expiry dates for CT, KC, CC, SB.

DATA SOURCE (since 2026-06-21): the authoritative ICE expiry dataset in
contract_expiries.json (sibling file). That JSON is the SINGLE SOURCE OF TRUTH.
The annual refresh updates ONLY that file -- no code change is needed yearly.

What this module builds from the JSON, once, at import time:
  _D          : {ICE_CODE_4CHAR: {'fnd': date, 'ltd': date, 'opt_exp': date}}
                fnd / ltd come from the FUTURES record; opt_exp is the OPTION
                LTD (OPT_LTD) of the option whose contract code matches the
                futures code (commodity + month-code + year). opt_exp drives the
                Black-76 time-to-expiry in vlm_master_fetch.py.
  _BBG_TO_ICE : {BBG_GENERIC_SLOT: ICE_CODE_4CHAR} resolved DYNAMICALLY from
                today's date using the proven nearest-unexpired rule (folded in
                from contract_resolver_proposed.py). Replaces the old static,
                hand-stamped table that went stale and blanked Greeks.
  ice_to_bbg  : reverse of _BBG_TO_ICE (ICE code -> BBG generic slot).

ICE code form used throughout this module and by both consumers: 4 chars,
{COMM}{MONTH_CODE}{YEAR_DIGIT} e.g. CTN6, CTZ7, KCN7.
  Month codes: F=Jan G=Feb H=Mar J=Apr K=May M=Jun N=Jul Q=Aug U=Sep V=Oct X=Nov Z=Dec
  Year digit : last digit of year (6=2026, 7=2027, 8=2028, 9=2029)

PUBLIC API -- UNCHANGED. Consumer (VLM Data/vlm_master_fetch.py) calls these
with a SPECIFIC 4-char ICE code and must keep working with zero changes:

    from contract_dates import get_fnd, get_ltd, get_opt_exp, get_bbg_slot
    from contract_dates import contract_month_label, ice_to_bbg, _D, _BBG_TO_ICE

    get_fnd('CTN7')      -> date(2027, 6, 24)
    get_ltd('CTN7')      -> date(2027, 7,  8)
    get_opt_exp('CTN7')  -> date(2027, 6, 11)   # OPTION expiry (Black-76 T)
    get_bbg_slot('CTN7') -> 'CTJUL1'            # dynamic nearest-unexpired slot
    ice_to_bbg           -> full reverse mapping dict

SAFE FALLBACK: if contract_expiries.json is missing/unreadable, or a contract
is not found, a clear warning is logged and the getters return None. The module
NEVER crashes on import and NEVER returns an expired/garbage contract.

ICE expiry page URLs (provenance of the JSON):
  Futures  CT /products/254  KC /products/15   CC /products/7    SB /products/23
  Options  CT /products/1027 KC /products/14   CC /products/8    SB /products/22
"""

import json
import os
from datetime import date

# -- Month code <-> name/token tables -----------------------------------------
_MON_CODE_TO_NAME = {
    'F': 'Jan', 'G': 'Feb', 'H': 'Mar', 'J': 'Apr', 'K': 'May', 'M': 'Jun',
    'N': 'Jul', 'Q': 'Aug', 'U': 'Sep', 'V': 'Oct', 'X': 'Nov', 'Z': 'Dec',
}
# Numeric month -> ICE month code (used to derive the code from the JSON).
_NUM_TO_CODE = {
    1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
    7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z',
}
# 3-letter generic month token (Bloomberg) <-> ICE month code.
_CODE_TO_TOKEN = {
    'F': 'JAN', 'G': 'FEB', 'H': 'MAR', 'J': 'APR', 'K': 'MAY', 'M': 'JUN',
    'N': 'JUL', 'Q': 'AUG', 'U': 'SEP', 'V': 'OCT', 'X': 'NOV', 'Z': 'DEC',
}
_TOKEN_TO_CODE = {v: k for k, v in _CODE_TO_TOKEN.items()}

_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'contract_expiries.json')


def _warn(msg):
    """Minimal, dependency-free warning to stdout. Never raises."""
    try:
        print('[contract_dates] WARNING: ' + str(msg))
    except Exception:
        pass


def _to_4char(json_code):
    """
    Normalize a JSON contract code (e.g. 'CTN26', 'CTN27') to the 4-char ICE
    form the consumers use ('CTN6', 'CTN7'): keep commodity + month code, take
    only the LAST year digit.
    """
    if not json_code or len(json_code) < 4:
        return None
    comm = json_code[:2]
    mc = json_code[2]
    yr_digit = json_code[-1]
    if mc not in _MON_CODE_TO_NAME or not yr_digit.isdigit():
        return None
    return comm + mc + yr_digit


def _parse_iso(s):
    """'YYYY-MM-DD' -> date, or None."""
    if not s or not isinstance(s, str):
        return None
    try:
        y, m, d = s.split('-')
        return date(int(y), int(m), int(d))
    except Exception:
        return None


def _build_calendar_from_json(path):
    """
    Load contract_expiries.json and build _D:
      {4char_ice_code: {'fnd': date, 'ltd': date, 'opt_exp': date or None}}

    fnd / ltd : from the FUTURES record (FND / LTD).
    opt_exp   : OPT_LTD of the OPTION record whose code matches the futures code
                (commodity + month code + year). Verified 1:1 against the prior
                ICE-hand-verified table for every live contract on 2026-06-21.

    Returns {} (and logs) on any failure -- never raises.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
    except FileNotFoundError:
        _warn('contract_expiries.json not found at ' + repr(path) +
              ' -- all lookups will return None.')
        return {}
    except Exception as e:
        _warn('could not read/parse contract_expiries.json (' + str(e) +
              ') -- all lookups will return None.')
        return {}

    products = (raw or {}).get('products', {})
    if not isinstance(products, dict) or not products:
        _warn('contract_expiries.json has no "products" -- '
              'all lookups will return None.')
        return {}

    out = {}
    for comm, pdata in products.items():
        if not isinstance(pdata, dict):
            continue

        # First index option OPT_LTD by 4-char code for this product.
        # COMPLETENESS GUARD: a listed option with NO parseable OPT_LTD is a data
        # gap in contract_expiries.json (seen with SBX6). Without an expiry the
        # option's rows get no DTE and blank Black-76 IV downstream, silently.
        # Warn at build time so the monthly JSON refresh catches it -- the code
        # cannot invent a date that isn't in the source.
        opt_ltd_by_code = {}
        for ocode, oinfo in (pdata.get('options', {}) or {}).items():
            key = _to_4char(ocode)
            if not key:
                continue
            d = _parse_iso((oinfo or {}).get('OPT_LTD'))
            if d:
                opt_ltd_by_code[key] = d
            else:
                _warn('option ' + str(ocode) + ' (' + str(comm) +
                      ') has no parseable OPT_LTD in contract_expiries.json -- '
                      'its option rows will get no expiry/DTE/IV. Fix at refresh.')

        # Then walk the futures and attach the matching option expiry.
        for fcode, finfo in (pdata.get('futures', {}) or {}).items():
            key = _to_4char(fcode)
            if not key:
                continue
            fnd = _parse_iso((finfo or {}).get('FND'))
            ltd = _parse_iso((finfo or {}).get('LTD'))
            opt_exp = opt_ltd_by_code.get(key)  # may be None (no live option)
            if fnd is None and ltd is None and opt_exp is None:
                continue
            entry = {'fnd': fnd, 'ltd': ltd}
            # Match the prior _D shape: only include 'opt_exp' when known, so
            # callers that test membership ('opt_exp' in info) behave as before.
            if opt_exp is not None:
                entry['opt_exp'] = opt_exp
            out[key] = entry

        # SERIAL-MONTH FIX (structural, not a per-symbol patch). Serial option
        # months (CT: Jan=F, Sep=U, Nov=X; also CC/KC Q,V and SB F,Q,U) have NO
        # listed FUTURE, so the futures walk above never creates a _D entry for
        # them and their expiry is lost -- blanking DTE and Black-76 IV for every
        # serial-tenor option row across ALL FOUR commodities. The option's own
        # OPT_LTD already lives in the JSON (it is what Black-76 T needs); the old
        # loop just never read it for future-less codes. Carry any option-only
        # code (not already added from a future) in as an OPTION-ONLY entry:
        #   fnd/ltd = None  (a serial has no futures FND/LTD -- correct, not a gap)
        #   opt_exp = the option's own OPT_LTD  (drives time-to-expiry)
        # ADDITIVE by construction: it only fills codes absent from `out`, so it
        # can never alter a quarterly entry. Proven (2026-07-07) to leave the
        # dynamic BBG generic<->ICE slot maps byte-identical: no consumer generic
        # requests a serial month token, so these entries stay out of slot
        # resolution and only surface via direct get_opt_exp(<serial code>).
        for ocode, oe in opt_ltd_by_code.items():
            if ocode not in out:
                out[ocode] = {'fnd': None, 'ltd': None, 'opt_exp': oe}

    if not out:
        _warn('contract_expiries.json produced an empty calendar -- '
              'all lookups will return None.')
    return out


# -- Ground-truth expiry calendar (built once at import) ----------------------
_D = _build_calendar_from_json(_JSON_PATH)


# -- Dynamic Bloomberg-generic resolver (folded in from
#    contract_resolver_proposed.py -- same proven logic, same semantics) ------

def generic_month_name(month_code):
    """'N' -> 'JUL'."""
    return _CODE_TO_TOKEN.get(month_code.upper(), '???')


def _parse_generic(generic):
    """
    Split a Bloomberg generic slot into (commodity, month_token, slot_n).
    Accepts 'CTJUL1', 'ctjul2', 'CTJUL1 Comdty'. Returns (comm, token, n) or
    None if it does not look like <2-letter comm><3-letter month><digit>.
    """
    if not generic:
        return None
    s = generic.upper().split()[0]            # drop ' Comdty'
    if len(s) < 6:
        return None
    comm = s[:2]
    token = s[2:5]
    tail = s[5:]
    if token not in _TOKEN_TO_CODE:
        return None
    if not tail.isdigit():
        return None
    n = int(tail)
    if n < 1:
        return None
    return comm, token, n


def _calendar_for(comm, month_code):
    """
    All contracts in _D for this commodity + delivery month with a known
    opt_exp, sorted chronologically by option expiry.
    """
    out = []
    for ice, info in _D.items():
        if ice[:2] != comm:
            continue
        if ice[-2] != month_code:          # month code char (second-to-last)
            continue
        if info.get('opt_exp') is None:
            continue
        out.append((ice, info))
    out.sort(key=lambda kv: kv[1]['opt_exp'])
    return out


def resolve_generic_to_ice(generic, as_of):
    """
    DYNAMIC resolver. Given a Bloomberg delivery-month generic and a date,
    return the specific ICE contract Bloomberg would point that generic at.

    Selection rule: slot N (1-based) = the Nth contract of the requested
    delivery month whose OPTION EXPIRY (opt_exp) is STRICTLY AFTER as_of.
    Selecting on opt_exp guarantees positive time-to-expiry downstream, so a
    resolved slot can never blank the Greeks.

    Returns dict {generic, ice, opt_exp, fnd, ltd} or None when the generic is
    unparseable or _D lacks enough out-year contracts to satisfy slot N
    (a SAFE exhaustion signal -- never a wrong/expired contract).
    """
    parsed = _parse_generic(generic)
    if parsed is None:
        return None
    comm, token, n = parsed
    month_code = _TOKEN_TO_CODE[token]

    cal = _calendar_for(comm, month_code)
    live = [(ice, info) for ice, info in cal if info['opt_exp'] > as_of]
    if len(live) < n:
        return None

    ice, info = live[n - 1]
    return {
        'generic': comm + token + str(n),
        'ice':     ice,
        'opt_exp': info['opt_exp'],
        'fnd':     info['fnd'],
        'ltd':     info['ltd'],
    }


# -- Dynamic BBG generic <-> ICE maps (built once from today's date) ----------
# These replace the old static, hand-stamped _BBG_TO_ICE table. They are built
# from _D using the proven nearest-unexpired resolver, anchored at today's date,
# so they never go stale: as a front contract's OPTION expires, the slot rolls
# forward to the next live contract automatically.

# The generic universe the live consumers request (slots 1 and 2 per delivery
# month) -- matches OI_TICKERS in vlm_master_fetch.py and the proof harness.
_CONSUMER_GENERICS = [
    # Cotton (CT delivery months H,K,N,V,Z)
    'CTMAR1', 'CTMAY1', 'CTJUL1', 'CTOCT1', 'CTDEC1',
    'CTMAR2', 'CTMAY2', 'CTJUL2', 'CTOCT2', 'CTDEC2',
    # Sugar (H,K,N,V)
    'SBMAR1', 'SBMAY1', 'SBJUL1', 'SBOCT1',
    'SBMAR2', 'SBMAY2', 'SBJUL2', 'SBOCT2',
    # Coffee (H,K,N,U,Z)
    'KCMAR1', 'KCMAY1', 'KCJUL1', 'KCSEP1', 'KCDEC1',
    'KCMAR2', 'KCMAY2', 'KCJUL2', 'KCSEP2', 'KCDEC2',
    # Cocoa (H,K,N,U,Z)
    'CCMAR1', 'CCMAY1', 'CCJUL1', 'CCSEP1', 'CCDEC1',
    'CCMAR2', 'CCMAY2', 'CCJUL2', 'CCSEP2', 'CCDEC2',
]


def _build_bbg_maps(as_of):
    """
    Build {generic_slot: ice_code} and its reverse {ice_code: generic_slot}
    dynamically as of as_of. Slot 1 (nearest unexpired) is preferred when an
    ICE code is reachable from both slot 1 and slot 2 of a generic, so that
    get_bbg_slot(ice) returns the lowest-numbered slot Bloomberg would use.
    """
    fwd = {}
    for g in _CONSUMER_GENERICS:
        res = resolve_generic_to_ice(g, as_of)
        if res is not None:
            fwd[g] = res['ice']

    def _slot_num(slot):
        tail = slot[5:]
        return int(tail) if tail.isdigit() else 99

    rev = {}
    # Prefer the lowest slot number for each ICE code (...1 over ...2).
    for g, ice in fwd.items():
        if ice not in rev or _slot_num(g) < _slot_num(rev[ice]):
            rev[ice] = g
    return fwd, rev


_BBG_TO_ICE, ice_to_bbg = _build_bbg_maps(date.today())


# -- Contract month label  (KCN6 -> "Jul 2026") ------------------------------

def contract_month_label(ice_code):
    """'KCN6' -> 'Jul 2026',  'CTZ7' -> 'Dec 2027'."""
    try:
        mc = ice_code[-2]
        yr = int(ice_code[-1])
        mon = _MON_CODE_TO_NAME.get(mc.upper(), '???')
        return mon + ' ' + str(2020 + yr)
    except Exception:
        return ''


# -- Public lookup functions (SIGNATURES UNCHANGED) ---------------------------

def _resolve(code):
    """
    Accept a specific ICE contract code (4-char, e.g. 'CTN7', with/without
    ' Comdty') OR a BBG generic slot ('CTJUL1'). Returns the _D entry dict or
    None. Generic slots are resolved dynamically as of today.
    """
    if not code:
        return None
    upper = code.upper().split()[0]
    if upper in _D:
        return _D[upper]
    # Maybe it's a BBG generic slot -- resolve it dynamically.
    res = resolve_generic_to_ice(upper, date.today())
    if res is not None:
        return _D.get(res['ice'])
    # Last resort: the prebuilt forward map (covers the consumer slots).
    ice = _BBG_TO_ICE.get(upper)
    if ice:
        return _D.get(ice)
    return None


def get_dates(code):
    """Return {'fnd': date, 'ltd': date, 'opt_exp': date} or None."""
    return _resolve(code)


def get_fnd(code):
    """First Notice Date -- accepts ICE code or BBG slot. None if unknown."""
    d = _resolve(code)
    return d.get('fnd') if d else None


def get_ltd(code):
    """Last Trading Date -- accepts ICE code or BBG slot. None if unknown."""
    d = _resolve(code)
    return d.get('ltd') if d else None


def get_opt_exp(code):
    """
    Option expiry date (OPTIONS LTD) -- accepts ICE code or BBG slot.
    Sourced from contract_expiries.json. Drives Black-76 time-to-expiry.
    Returns None if unknown (caller treats None as 'no Greeks' -- never crashes).
    """
    d = _resolve(code)
    return d.get('opt_exp') if d else None


def get_bbg_slot(ice_code):
    """
    Translate a specific ICE contract code to its Bloomberg generic slot.
    'CTN7' -> 'CTJUL1' (as of today). Returns None if the contract is not a
    currently-live consumer slot.
    """
    if not ice_code:
        return None
    return ice_to_bbg.get(ice_code.upper().split()[0])


# -- Last-resort reference: the previous STATIC tables (NOT used) --------------
# Retained only as a commented historical reference. The live behaviour above
# is sourced entirely from contract_expiries.json. Do NOT re-enable these --
# they go stale and were the cause of the blank-Greeks regression.
#
# _D_STATIC = {
#     'CTN6': {'fnd': date(2026, 6, 24), 'ltd': date(2026, 7, 9), 'opt_exp': date(2026, 6, 12)},
#     ... (see git history for the full hand-maintained table) ...
# }
# _BBG_TO_ICE_STATIC = {
#     'CTJUL1': 'CTN6', 'CTJUL2': 'CTN7', 'CTOCT1': 'CTV6', ...
# }


if __name__ == '__main__':
    today = date.today()
    print('contract_dates self-test  (as_of=' + str(today) + ')')
    print('  _D entries loaded from JSON : ' + str(len(_D)))
    print('  _BBG_TO_ICE slots resolved  : ' + str(len(_BBG_TO_ICE)))
    for g in ('CTJUL1', 'CCJUL1', 'KCJUL1', 'SBJUL1'):
        ice = _BBG_TO_ICE.get(g)
        opt = get_opt_exp(ice) if ice else None
        slot = get_bbg_slot(ice) if ice else None
        T = (opt - today).days if opt else None
        print('  ' + g + ' -> ' + str(ice) + '  opt_exp=' + str(opt) +
              '  T=' + str(T) + 'd  get_bbg_slot(' + str(ice) + ')=' + str(slot))
