#!/usr/bin/env python3
"""
backfill_options_iv.py -- retro-fill BLANK expire_dt / days_to_exp / iv_pct / greeks
in options_oi.csv, reproducing the daily job's Black-76 pipeline EXACTLY.

READ-ONLY on the live file: reads data/options_oi.csv, writes
  data/options_oi_IVFILL.csv        (corrected copy)
  data/options_ivfill_PROVENANCE.csv (every cell change logged)
NEVER writes the live file. Promotion is a separate, gated step.

Guarantees (see OPTIONS_IV_BACKFILL_PLAN.md):
  - fill-empty-only: only writes a cell that is currently blank; never overwrites.
  - string-preserving: every non-target field is emitted byte-identical to input
    (csv.DictReader/DictWriter, no numeric coercion). Row count unchanged.
  - two independent write-units: (expire_dt+days_to_exp) and (iv_pct+greeks).
  - matches the job: trade_dt = release-date - 1 day, weekend-skip only;
    strike/put_call parsed from security_des when the column is blank;
    F priority = F_parity(>=4 shared strikes) -> [no historical px_by_ice/slot];
    rfr = 0.0365 base SOFR; MIN_DTE_DAYS=2; MIN_PARITY_PAIRS=4; signed delta.
  - no sane-band clamp (the job has none); extreme-corner solves are flagged in
    provenance, not withheld.
"""
import csv, re, math, sys, os
from collections import defaultdict
from datetime import date, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(HERE, 'data', 'options_oi.csv')
OUT  = os.path.join(HERE, 'data', 'options_oi_IVFILL.csv')
PROV = os.path.join(HERE, 'data', 'options_ivfill_PROVENANCE.csv')

sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'VLM Data'))
import contract_dates as cd
from black76_validator import implied_vol, black76_greeks

RFR = 0.0365
MIN_DTE_DAYS = 2
MIN_PARITY_PAIRS = 4

# Validated rolled-off expiries (ICE-rule-derived + data-validated; see
# ICE_OPTIONS_SERIAL_EXPIRY_AUTHORITY.md).
# NOTE: SBX6 (Nov sugar) IS a valid SB serial (X -> next regular Jan; LTD = 15th of
# the preceding month = 15-Oct-2026, a Thursday, no adjustment). Earlier "phantom"
# call was wrong; adversarial review caught its omission here. Included below.
HIST_OPT_EXP = {
    'CTN6': date(2026, 6, 12), 'KCN6': date(2026, 6, 12), 'CCN6': date(2026, 6, 12),
    'SBN6': date(2026, 6, 15), 'KCM6': date(2026, 5, 8),  'CCM6': date(2026, 5, 8),
    'SBM6': date(2026, 5, 15), 'SBX6': date(2026, 10, 15),
}

FIELDS = ['date', 'commodity', 'security_des', 'contract_month', 'put_call',
          'strike_px', 'open_int', 'oi_chg', 'px_settle', 'px_volume',
          'expire_dt', 'days_to_exp', 'iv_pct', 'delta', 'gamma', 'vega', 'theta']

STRIKE_RE = re.compile(r'^[A-Z]{2}[A-Z0-9]{2}[CP]\s+([0-9]+(?:\.[0-9]+)?)$')


def _f(x):
    try: return float(x)
    except (TypeError, ValueError): return None


def trade_dt_of(release_iso):
    """Job's exact shift: release_date - 1 day, skip Sat/Sun only (no holidays)."""
    d = date.fromisoformat(release_iso) - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def strike_of(row):
    """strike_px if present, else parse from security_des (blank pre-2026-04-25)."""
    s = _f(row['strike_px'])
    if s and s > 0:
        return s
    m = STRIKE_RE.match((row['security_des'] or '').strip())
    return float(m.group(1)) if m else None


def pc_of(row):
    """put_call column if C/P, else security_des[4] (blank in column pre-cutover)."""
    pc = row['put_call']
    if pc in ('C', 'P'):
        return pc
    sd = row['security_des'] or ''
    return sd[4] if len(sd) > 4 and sd[4] in ('C', 'P') else None


def opt_exp_of(ice):
    """Forward-calendar expiry, else validated historical expiry for rolled-off codes."""
    return cd.get_opt_exp(ice) or HIST_OPT_EXP.get(ice)


def main():
    with open(SRC, newline='') as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames
        rows = list(reader)

    # Schema guard: refuse to run if the file is not the exact 17-col shape.
    if header != FIELDS:
        sys.exit('SCHEMA MISMATCH: expected %r, got %r -- aborting.' % (FIELDS, header))

    # F_parity per (date, commodity, contract_month) using parsed strike/pc.
    grp = defaultdict(lambda: {'C': {}, 'P': {}})
    for r in rows:
        pc = pc_of(r); k = strike_of(r); px = _f(r['px_settle'])
        if pc in ('C', 'P') and k and k > 0 and px and px > 0:
            grp[(r['date'], r['commodity'], r['contract_month'])][pc][k] = px

    def fwd_of(keytuple, td, oe):
        d = grp[keytuple]
        shared = sorted(set(d['C']) & set(d['P']))
        if len(shared) < MIN_PARITY_PAIRS:
            return None
        T = max((oe - td).days, 0) / 365.0
        fwds = sorted(k + (d['C'][k] - d['P'][k]) * math.exp(RFR * T) for k in shared)
        n = len(fwds)
        return fwds[n // 2] if n % 2 else 0.5 * (fwds[n // 2 - 1] + fwds[n // 2])

    prov = []   # (date, security_des, field, old, new, note)
    n_expire = n_iv = n_greek_rows = 0

    # ===================================================================
    # GATED OVERWRITE PASS: correct the 2026-06-03 single-run mis-stamp.
    # Lou-approved 2026-07-07. EVERY option row on 2026-06-03 (and ONLY that
    # date, verified) carries an expiry ~1wk LATER than the same contract's
    # expiry on every other date -- 23 contracts, and IVs solved off the wrong
    # (late) expiry are themselves wrong. This is NOT fill-empty-only: it
    # OVERWRITES populated expire_dt/days_to_exp/iv_pct/greeks on 06-03, then
    # recomputes IV/greeks off the corrected expiry. Correct expiry = the
    # per-contract MAJORITY expiry across all OTHER dates (matches ICE authority).
    # ===================================================================
    MISSTAMP_DATE = '2026-06-03'
    # Build per-ice-code majority expiry from every date EXCEPT the mis-stamp date.
    _exp_dates = defaultdict(lambda: defaultdict(set))  # ice -> expiry -> {dates}
    for r in rows:
        if r['date'] == MISSTAMP_DATE:
            continue
        ice = (r['security_des'] or '')[:4]
        if r['expire_dt'].strip():
            _exp_dates[ice][r['expire_dt']].add(r['date'])
    majority_exp = {}
    for ice, exps in _exp_dates.items():
        majority_exp[ice] = max(exps, key=lambda e: len(exps[e]))

    n_0603_exp = n_0603_iv = 0
    for r in rows:
        if r['date'] != MISSTAMP_DATE:
            continue
        ice = (r['security_des'] or '')[:4]
        correct = majority_exp.get(ice)
        if correct is None or r['expire_dt'].strip() == correct:
            continue  # nothing to correct (or already right)
        td = trade_dt_of(r['date'])
        pc = pc_of(r)
        oe = date.fromisoformat(correct)
        old_exp = r['expire_dt']; old_dte = r['days_to_exp']; old_iv = r['iv_pct']
        dte = (oe - td).days
        # overwrite expiry + dte
        r['expire_dt'] = correct
        r['days_to_exp'] = str(dte)
        prov.append((r['date'], r['security_des'], 'expire_dt', old_exp, correct, 'MISSTAMP_0603_fix'))
        prov.append((r['date'], r['security_des'], 'days_to_exp', old_dte, str(dte), 'MISSTAMP_0603_fix'))
        n_0603_exp += 1
        # recompute IV/greeks off corrected expiry IF the row had an IV (else leave blank)
        if old_iv.strip() == '' or pc not in ('C', 'P'):
            continue
        K = strike_of(r); settle = _f(r['px_settle'])
        F = fwd_of((r['date'], r['commodity'], r['contract_month']), td, oe) if (K and settle) else None
        new_iv = None
        if F and dte >= MIN_DTE_DAYS and K and K > 0 and settle and settle > 0:
            iv = implied_vol(F, K, dte / 365.0, RFR, settle, pc == 'C')
            if iv:
                new_iv = str(round(iv * 100, 4))
        # overwrite iv + greeks with recomputed (or blank if unsolvable on corrected expiry)
        r['iv_pct'] = new_iv if new_iv is not None else ''
        prov.append((r['date'], r['security_des'], 'iv_pct', old_iv, r['iv_pct'], 'MISSTAMP_0603_recompute'))
        if new_iv is not None:
            g = black76_greeks(F, K, dte / 365.0, RFR, iv, pc == 'C')
            for fld, prec in (('delta', 4), ('gamma', 6), ('vega', 6), ('theta', 6)):
                old_g = r[fld]
                r[fld] = str(round(g[fld], prec)) if g else ''
                prov.append((r['date'], r['security_des'], fld, old_g, r[fld], 'MISSTAMP_0603_recompute'))
        else:
            for fld in ('delta', 'gamma', 'vega', 'theta'):
                old_g = r[fld]
                if old_g.strip() != '':
                    r[fld] = ''
                    prov.append((r['date'], r['security_des'], fld, old_g, '', 'MISSTAMP_0603_recompute'))
        n_0603_iv += 1

    for r in rows:
        ice = (r['security_des'] or '')[:4]
        td = trade_dt_of(r['date'])
        oe = opt_exp_of(ice)
        pc = pc_of(r)

        # --- write-unit 1: expire_dt + days_to_exp (blank-only, together) ---
        if r['expire_dt'].strip() == '' and oe is not None and pc in ('C', 'P'):
            dte = (oe - td).days
            r['expire_dt'] = str(oe)
            r['days_to_exp'] = str(dte)
            prov.append((r['date'], r['security_des'], 'expire_dt', '', str(oe), 'opt_exp'))
            prov.append((r['date'], r['security_des'], 'days_to_exp', '', str(dte), 'trade_dt=%s' % td))
            n_expire += 1

        # --- write-unit 2: iv_pct + greeks (blank-only; needs solvable IV) ---
        if r['iv_pct'].strip() != '':
            continue
        if oe is None or pc not in ('C', 'P'):
            continue
        dte = (oe - td).days
        if dte < MIN_DTE_DAYS:
            continue
        K = strike_of(r); settle = _f(r['px_settle'])
        if not K or K <= 0 or not settle or settle <= 0:
            continue
        F = fwd_of((r['date'], r['commodity'], r['contract_month']), td, oe)
        if F is None:
            continue
        T = dte / 365.0
        iv = implied_vol(F, K, T, RFR, settle, pc == 'C')
        if not iv:
            continue
        r['iv_pct'] = str(round(iv * 100, 4))
        note = 'F_parity=%.4f dte=%d' % (F, dte)
        # extreme deep-ITM/near-expiry corner: flag, do not withhold (job doesn't).
        try:
            if abs(math.log(F / K)) > 0.25 and dte < 10:
                note += ' EXTREME_CORNER'
        except (ValueError, ZeroDivisionError):
            pass
        prov.append((r['date'], r['security_des'], 'iv_pct', '', r['iv_pct'], note))
        n_iv += 1

        g = black76_greeks(F, K, T, RFR, iv, pc == 'C')
        if g:
            for fld, val, prec in (('delta', g['delta'], 4), ('gamma', g['gamma'], 6),
                                    ('vega', g['vega'], 6), ('theta', g['theta'], 6)):
                if r[fld].strip() == '':
                    r[fld] = str(round(val, prec))
                    prov.append((r['date'], r['security_des'], fld, '', r[fld], 'b76'))
            n_greek_rows += 1

    # write outputs (string-preserving: values already strings; blanks stay '')
    with open(OUT, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    with open(PROV, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['date', 'security_des', 'field', 'old_value', 'new_value', 'note'])
        w.writerows(prov)

    print('rows: %d (unchanged)' % len(rows))
    print('--- additive fill-empty-only ---')
    print('expire_dt+days_to_exp filled : %d' % n_expire)
    print('iv_pct filled                : %d' % n_iv)
    print('greek-rows filled            : %d' % n_greek_rows)
    print('--- gated 2026-06-03 mis-stamp correction (OVERWRITES) ---')
    print('expire/dte corrected         : %d' % n_0603_exp)
    print('iv/greeks recomputed         : %d' % n_0603_iv)
    print('provenance entries           : %d' % len(prov))
    print('wrote %s' % OUT)
    print('wrote %s' % PROV)


if __name__ == '__main__':
    main()
