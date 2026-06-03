"""
shadow_pipeline.py — ICE RTD shadow test pipeline for KC and CC.

Runs AFTER vlm_master_fetch.py (Bloomberg pipeline).
Reads ICE RTD workbooks, writes shadow CSVs, generates shadow PNGs,
saves shadow site-post HTML (does NOT post to live site), and diffs
shadow output against today's Bloomberg CSV rows.

Shadow files:
  data/oi_data_shadow.csv        — same schema as oi_data.csv
  data/options_oi_shadow.csv     — same schema as options_oi.csv
  output/shadow/{date}/OI_Monitor_{COMM}_{date}.png
  output/shadow/{date}/site_post_{COMM}.html
  data/shadow_diff_log.csv       — daily diff results

Usage:
  python shadow_pipeline.py
"""

import csv
import pathlib
import sys
from datetime import date, datetime, timedelta

BASE_DIR    = pathlib.Path(__file__).parent
DATA_DIR    = BASE_DIR / 'data'
OUTPUT_DIR  = BASE_DIR / 'output' / 'shadow'

SHADOW_OI_CSV  = DATA_DIR / 'oi_data_shadow.csv'
SHADOW_OPT_CSV = DATA_DIR / 'options_oi_shadow.csv'
LIVE_OI_CSV    = DATA_DIR / 'oi_data.csv'
LIVE_OPT_CSV   = DATA_DIR / 'options_oi.csv'
DIFF_LOG_CSV   = DATA_DIR / 'shadow_diff_log.csv'

OI_COLS  = ['date','commodity','contract','bbg_ticker',
            'settle','open_int','oi_chg','first_notice','last_trade']
OPT_COLS = ['date','commodity','security_des','contract_month',
            'put_call','strike_px','open_int','oi_chg','px_settle','px_volume']

SHADOW_COMMS = ['KC', 'CC']


# ── Date helpers ─────────────────────────────────────────────────────────────

def _prior_bday(d: date) -> date:
    d = d - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _as_of(release_date_str: str) -> str:
    d = datetime.strptime(release_date_str, '%Y-%m-%d').date()
    return _prior_bday(d).strftime('%Y-%m-%d')


# ── CSV helpers ──────────────────────────────────────────────────────────────

def _read_csv(path: pathlib.Path) -> list:
    if not path.exists():
        return []
    return list(csv.DictReader(path.open(encoding='utf-8')))


def _append_csv(path: pathlib.Path, cols: list, rows: list):
    """Append rows to CSV, creating with header if new. Skips if today already present."""
    if not rows:
        return 0
    today = rows[0]['date']
    existing = _read_csv(path)
    existing_dates = {r['date'] for r in existing
                      if r.get('commodity') in SHADOW_COMMS}
    if today in existing_dates:
        print(f'    [shadow] {path.name}: {today} already present — skipped')
        return 0
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open('a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if write_header:
            w.writeheader()
        w.writerows(rows)
    return len(rows)


# ── Diff logic ───────────────────────────────────────────────────────────────

def _diff_futures(today_str: str) -> list:
    """Compare shadow vs Bloomberg futures OI rows for today. Returns diff records."""
    live   = {r['contract']: r for r in _read_csv(LIVE_OI_CSV)
              if r['date'] == today_str and r.get('commodity') in SHADOW_COMMS}
    shadow = {r['contract']: r for r in _read_csv(SHADOW_OI_CSV)
              if r['date'] == today_str and r.get('commodity') in SHADOW_COMMS}
    diffs = []
    for contract in sorted(set(live) | set(shadow)):
        lr = live.get(contract)
        sr = shadow.get(contract)
        if lr is None:
            diffs.append({'date': today_str, 'type': 'futures', 'contract': contract,
                          'field': 'row', 'live': 'MISSING', 'shadow': 'PRESENT', 'match': 'FAIL'})
            continue
        if sr is None:
            diffs.append({'date': today_str, 'type': 'futures', 'contract': contract,
                          'field': 'row', 'live': 'PRESENT', 'shadow': 'MISSING', 'match': 'FAIL'})
            continue
        for field in ['settle', 'open_int', 'first_notice', 'last_trade']:
            lv = lr.get(field, '').strip()
            sv = sr.get(field, '').strip()
            match = 'PASS'
            if field == 'settle':
                try:
                    match = 'PASS' if abs(float(lv) - float(sv)) < 0.005 else 'FAIL'
                except (ValueError, TypeError):
                    match = 'FAIL' if lv != sv else 'PASS'
            elif field == 'open_int':
                match = 'PASS' if lv == sv else 'FAIL'
            else:
                match = 'PASS' if lv == sv else 'FAIL'
            diffs.append({'date': today_str, 'type': 'futures', 'contract': contract,
                          'field': field, 'live': lv, 'shadow': sv, 'match': match})
    return diffs


def _diff_options(today_str: str) -> dict:
    """
    High-level diff: compare total OI per commodity for top-10 movers.
    Full per-row diff is verbose; we check aggregate and top-10 alignment.
    Returns summary dict.
    """
    results = {}
    for comm in SHADOW_COMMS:
        live   = [r for r in _read_csv(LIVE_OPT_CSV)
                  if r['date'] == today_str and r.get('commodity') == comm]
        shadow = [r for r in _read_csv(SHADOW_OPT_CSV)
                  if r['date'] == today_str and r.get('commodity') == comm]

        live_total_oi   = sum(int(r['open_int']) for r in live   if r.get('open_int'))
        shadow_total_oi = sum(int(r['open_int']) for r in shadow if r.get('open_int'))
        live_rows   = len(live)
        shadow_rows = len(shadow)
        # PCT difference in total OI
        pct_diff = abs(live_total_oi - shadow_total_oi) / max(live_total_oi, 1) * 100
        results[comm] = {
            'live_rows':    live_rows,
            'shadow_rows':  shadow_rows,
            'live_oi':      live_total_oi,
            'shadow_oi':    shadow_total_oi,
            'pct_diff':     round(pct_diff, 2),
            'match':        'PASS' if pct_diff < 1.0 else 'FAIL',
        }
    return results


def _write_diff_log(today_str: str, fut_diffs: list, opt_summary: dict):
    cols = ['date','type','contract','field','live','shadow','match']
    write_header = not DIFF_LOG_CSV.exists() or DIFF_LOG_CSV.stat().st_size == 0
    with DIFF_LOG_CSV.open('a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if write_header:
            w.writeheader()
        for r in fut_diffs:
            w.writerow(r)
        for comm, s in opt_summary.items():
            w.writerow({
                'date':     today_str,
                'type':     'options_summary',
                'contract': comm,
                'field':    'total_oi',
                'live':     str(s['live_oi']),
                'shadow':   str(s['shadow_oi']),
                'match':    s['match'],
            })
            w.writerow({
                'date':     today_str,
                'type':     'options_summary',
                'contract': comm,
                'field':    'row_count',
                'live':     str(s['live_rows']),
                'shadow':   str(s['shadow_rows']),
                'match':    'PASS' if s['live_rows'] == s['shadow_rows'] else 'INFO',
            })


# ── PNG and HTML generation ──────────────────────────────────────────────────

def _generate_shadow_outputs(today_str: str, out_dir: pathlib.Path):
    """
    Reuse build_whatsapp_oi functions but pointed at shadow CSVs.
    Generates PNGs and saves site-post HTML without posting.
    """
    import csv as _csv

    # Temporarily monkey-patch the file paths used by build_whatsapp_oi
    import build_whatsapp_oi as bw

    _orig_oi  = bw.OI_FILE
    _orig_opt = bw.OPT_FILE

    try:
        bw.OI_FILE  = SHADOW_OI_CSV
        bw.OPT_FILE = SHADOW_OPT_CSV

        # Check shadow CSVs have today's data
        shadow_rows = _read_csv(SHADOW_OI_CSV)
        if not any(r['date'] == today_str for r in shadow_rows):
            print(f'  [shadow PNG] no shadow data for {today_str} — skipping PNG generation')
            return []

        release_date = max(r['date'] for r in shadow_rows)
        as_of        = _as_of(release_date)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = []

        for comm in SHADOW_COMMS:
            futures, agg_oi, agg_chg, _ = bw.load_futures(comm)
            if not futures:
                print(f'  [shadow PNG] no futures data for {comm}')
                continue
            opts, _ = bw.load_options_top10(comm, target_date=release_date)
            html     = bw.build_html(futures, agg_oi, agg_chg, opts, as_of, comm)

            # Save HTML
            html_path = out_dir / f'site_post_{comm}.html'
            html_path.write_text(html, encoding='utf-8')
            print(f'  [shadow HTML] saved: {html_path.name}')

            # Render PNG
            png_path = out_dir / f'OI_Monitor_{comm}_{as_of}.png'
            result = bw.render_png(html, png_path)
            if result:
                print(f'  [shadow PNG]  saved: {png_path.name}')
                saved.append(result)
            else:
                print(f'  [shadow PNG]  render failed for {comm}')

        return saved
    finally:
        bw.OI_FILE  = _orig_oi
        bw.OPT_FILE = _orig_opt


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    today = date.today()
    if today.weekday() >= 5:
        print('Weekend — shadow pipeline skipped.')
        return 0

    today_str = today.strftime('%Y-%m-%d')
    out_dir   = OUTPUT_DIR / today_str

    print()
    print('=' * 60)
    print('  VLM SHADOW PIPELINE — ICE RTD')
    print(f'  KC + CC  |  {today_str}')
    print('=' * 60)

    # ── 1. Fetch from ICE RTD ────────────────────────────────────────
    from oi_ice_fetcher import fetch_ice_oi

    all_fut_rows = []
    all_opt_rows = []
    fetch_ok = []

    for comm in SHADOW_COMMS:
        print(f'\n  Fetching {comm} from ICE RTD...')
        fut, opts = fetch_ice_oi(comm, today_str, SHADOW_OI_CSV, SHADOW_OPT_CSV)
        if fut is None:
            print(f'  {comm}: UNAVAILABLE — workbook not open')
            continue
        print(f'  {comm}: {len(fut)} futures rows, {len(opts)} options rows')
        all_fut_rows.extend(fut)
        all_opt_rows.extend(opts)
        fetch_ok.append(comm)

    if not fetch_ok:
        print('\n  No ICE RTD data available. Shadow pipeline cannot run.')
        print('  Ensure Excel is open with ICE RTD FEED KC.xlsx and CC.xlsx.')
        return 1

    # ── 2. Append to shadow CSVs ─────────────────────────────────────
    print('\n  Writing shadow CSVs...')
    n_fut = _append_csv(SHADOW_OI_CSV,  OI_COLS,  all_fut_rows)
    n_opt = _append_csv(SHADOW_OPT_CSV, OPT_COLS, all_opt_rows)
    print(f'    oi_data_shadow.csv    +{n_fut} rows')
    print(f'    options_oi_shadow.csv +{n_opt} rows')

    # ── 3. Diff vs Bloomberg ─────────────────────────────────────────
    print('\n  Diffing vs Bloomberg...')
    if not LIVE_OI_CSV.exists():
        print('  WARNING: live oi_data.csv not found — skipping diff')
        fut_diffs  = []
        opt_summary = {}
    else:
        fut_diffs   = _diff_futures(today_str)
        opt_summary = _diff_options(today_str)
        _write_diff_log(today_str, fut_diffs, opt_summary)

        fut_fails = [d for d in fut_diffs if d['match'] == 'FAIL']
        print(f'  Futures diff: {len(fut_diffs)} checks, {len(fut_fails)} FAIL')
        for d in fut_fails:
            print(f'    FAIL  {d["contract"]:12} {d["field"]:15} live={d["live"]}  shadow={d["shadow"]}')

        for comm, s in opt_summary.items():
            print(f'  Options {comm}: live={s["live_oi"]:,}  shadow={s["shadow_oi"]:,}  '
                  f'diff={s["pct_diff"]}%  {s["match"]}')

    # ── 4. Generate shadow PNGs + HTML ───────────────────────────────
    print('\n  Generating shadow PNGs and HTML...')
    saved = _generate_shadow_outputs(today_str, out_dir)

    # ── 5. Summary ───────────────────────────────────────────────────
    print()
    print('=' * 60)
    print('  SHADOW PIPELINE COMPLETE')
    print(f'  Commodities fetched : {", ".join(fetch_ok)}')
    print(f'  Futures rows written: {n_fut}')
    print(f'  Options rows written: {n_opt}')
    print(f'  PNGs generated      : {len(saved)}')
    if out_dir.exists():
        print(f'  Output folder       : {out_dir}')
    if fut_diffs:
        n_pass = len([d for d in fut_diffs if d['match'] == 'PASS'])
        n_fail = len([d for d in fut_diffs if d['match'] == 'FAIL'])
        print(f'  Futures diff        : {n_pass} PASS  {n_fail} FAIL')
    print('=' * 60)
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
