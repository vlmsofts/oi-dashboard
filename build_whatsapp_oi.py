"""
build_whatsapp_oi.py — VLM OI Dashboard WhatsApp Image Generator
Generates a single PNG with:
  1. Cotton Futures OI table (current day)
  2. Top 10 Options OI Changes (day over day)

Usage:
  python build_whatsapp_oi.py
  python build_whatsapp_oi.py --output "output/whatsapp"

Requires: playwright (pip install playwright && playwright install chromium)
Run from Desktop: Open interest dashboard folder
"""

import csv, json, pathlib, argparse
from datetime import datetime, date

BASE_DIR    = pathlib.Path(__file__).parent
DATA_DIR    = BASE_DIR / 'data'
OI_FILE     = DATA_DIR / 'oi_data.csv'
OPT_FILE    = DATA_DIR / 'options_oi.csv'

# Ticker orders defined in TICKER_ORDERS dict in load_futures()

_MONTH_MAP = {
    'JAN':'Jan','FEB':'Feb','MAR':'Mar','APR':'Apr','MAY':'May','JUN':'Jun',
    'JUL':'Jul','AUG':'Aug','SEP':'Sep','OCT':'Oct','NOV':'Nov','DEC':'Dec',
}

def _fmt_cont(tk, fnd=''):
    """Format ticker label for display. CTJUL1 -> 'Jul 1 \'26', CTOCT2 -> 'Oct 2 \'26'"""
    suffix = tk.replace(' Comdty','')
    suffix = suffix[2:]  # strip CT/KC/CC/SB
    slot = suffix[-1]
    month_code = suffix[:-1]
    month = _MONTH_MAP.get(month_code, month_code)
    yr = f" '{fnd[2:4]}" if fnd and len(fnd) >= 4 else ''
    return f'{month} {slot}{yr}'


# Ticker order per commodity
TICKER_ORDERS = {
    'CT': ['CTMAR1','CTMAY1','CTJUL1','CTOCT1','CTDEC1','CTMAR2','CTMAY2','CTJUL2','CTOCT2','CTDEC2'],
    'KC': ['KCMAR1','KCMAY1','KCJUL1','KCSEP1','KCDEC1','KCMAR2','KCMAY2','KCJUL2','KCSEP2','KCDEC2'],
    'CC': ['CCMAR1','CCMAY1','CCJUL1','CCSEP1','CCDEC1','CCMAR2','CCMAY2','CCJUL2','CCSEP2','CCDEC2'],
    'SB': ['SBMAR1','SBMAY1','SBJUL1','SBOCT1','SBMAR2','SBMAY2','SBJUL2','SBOCT2'],
}

COMM_NAMES = {'CT': 'COTTON', 'KC': 'COFFEE', 'CC': 'COCOA', 'SB': 'SUGAR'}


def load_futures(comm='CT'):
    """Load today's futures OI rows for a given commodity, sorted by first notice date."""
    rows = list(csv.DictReader(OI_FILE.open(encoding='utf-8')))
    last_date = max(r['date'] for r in rows)
    today = [r for r in rows if r['date'] == last_date and r['commodity'] == comm]
    result = []
    for r in today:
        tk = r['contract']
        fnd = r.get('first_notice','')
        result.append({
            'ticker':   tk,
            'cont':     _fmt_cont(tk, fnd),
            'oi':       int(r['open_int'])   if r.get('open_int')   else 0,
            'oi_chg':   int(r['oi_chg'])     if r.get('oi_chg') and r['oi_chg'] not in ('','None') else 0,
            'settle':   float(r['settle'])   if r.get('settle')     else 0,
            'fnd':      fnd,
        })
    # Sort by first notice date when available; fall back to TICKER_ORDERS position
    order = TICKER_ORDERS.get(comm, [])
    def _sort_key(x):
        if x['fnd']:
            return (0, x['fnd'], 0)
        pos = order.index(x['ticker']) if x['ticker'] in order else 999
        return (1, '', pos)
    result.sort(key=_sort_key)
    agg_oi  = sum(r['oi']     for r in result)
    agg_chg = sum(r['oi_chg'] for r in result)
    return result, agg_oi, agg_chg, last_date

def load_options_top10(comm='CT'):
    """Load top 10 options by absolute OI change today for a given commodity."""
    rows = list(csv.DictReader(OPT_FILE.open(encoding='utf-8')))
    last_date = max(r['date'] for r in rows)
    today = [r for r in rows if r['date'] == last_date and r.get('commodity', 'CT') == comm]
    # Parse and filter
    parsed = []
    for r in today:
        chg = r.get('oi_chg','')
        if not chg or chg in ('','None'): continue
        try:
            chg_i = int(chg)
            if chg_i == 0: continue
            # Parse strike from security_des
            sec = r['security_des'].strip()
            strike = 0.0
            pc = r.get('put_call','')
            for i, ch in enumerate(sec):
                if i > 3 and ch in ('C','P'):
                    pc = ch
                    try: strike = float(sec[i+1:].strip())
                    except: pass
                    break
            parsed.append({
                'sec':    sec,
                'month':  r.get('contract_month',''),
                'pc':     pc,
                'strike': strike,
                'oi':     int(r['open_int'])    if r.get('open_int')    else 0,
                'chg':    chg_i,
                'settle': float(r['px_settle']) if r.get('px_settle') and r['px_settle'] != '' else None,
                'vol':    int(r['px_volume'])   if r.get('px_volume') and r['px_volume'] not in ('','0') else 0,
            })
        except Exception:
            continue
    # Sort by absolute change, take top 10
    parsed.sort(key=lambda x: abs(x['chg']), reverse=True)
    return parsed[:10], last_date


def build_html(futures, agg_oi, agg_chg, opts, as_of, comm='CT'):
    GREEN  = '#16a34a'
    RED    = '#dc2626'
    GOLD   = '#E8C547'
    TEXT   = '#0f172a'
    DIM    = '#1f2937'
    ROW1   = '#ffffff'
    ROW2   = '#f3f6fa'
    SECT   = '#e8eef5'
    HDR    = '#080f1a'
    PAGE   = '#f1f5f9'
    TICKER = '#1d4ed8'
    AMBER  = '#92400e'

    def fc(v, show_plus=True):
        if v is None: return '—'
        s = f'+{v:,}' if (v >= 0 and show_plus) else f'{v:,}'
        return s

    def color_chg(v):
        if v > 0: return GREEN
        if v < 0: return RED
        return DIM

    # ── Futures rows ────────────────────────────────────────────────
    FCOLS = '190px 110px 175px 148px 165px 260px'
    fut_rows = ''
    for i, r in enumerate(futures):
        bg = ROW1 if i % 2 == 0 else ROW2
        chg_color = color_chg(r['oi_chg'])
        oi_color  = AMBER if r['oi'] >= 50000 else GREEN if r['oi'] >= 10000 else TEXT
        fut_rows += f"""
        <div style="display:grid;grid-template-columns:{FCOLS};
                    background:{bg};padding:7px 16px;align-items:center;border-bottom:1px solid #e2e8f0;">
          <div style="font-size:20px;font-weight:700;color:{TICKER};">{r['ticker']}</div>
          <div style="font-size:18px;color:{DIM};text-align:right;">{r['cont']}</div>
          <div style="font-size:20px;font-weight:700;color:{oi_color};text-align:right;">{r['oi']:,}</div>
          <div style="font-size:20px;font-weight:700;color:{chg_color};text-align:right;">{fc(r['oi_chg'])}</div>
          <div style="font-size:20px;color:{TEXT};text-align:right;">{r['settle']:.2f}</div>
          <div style="font-size:18px;color:{DIM};text-align:right;">{r['fnd'] or '—'}</div>
        </div>"""
    # ── Totals bar ──────────────────────────────────────────────────
    tot_color = color_chg(agg_chg)
    fut_rows += f"""
        <div style="display:grid;grid-template-columns:{FCOLS};
                    background:#1e293b;padding:8px 16px;align-items:center;">
          <div style="font-size:17px;font-weight:700;color:#f1f5f9;letter-spacing:1px;">TOTAL</div>
          <div></div>
          <div style="font-size:20px;font-weight:700;color:{GOLD};text-align:right;">{agg_oi:,}</div>
          <div style="font-size:20px;font-weight:700;color:{tot_color};text-align:right;">{fc(agg_chg)}</div>
          <div></div>
          <div></div>
        </div>"""

    # ── Options top 10 rows ─────────────────────────────────────────
    OCOLS = '252px 130px 62px 164px 152px 142px 146px'
    opt_rows = ''
    for i, r in enumerate(opts):
        bg = ROW1 if i % 2 == 0 else ROW2
        pc_color = GREEN if r['pc'] == 'C' else RED
        chg_color = color_chg(r['chg'])
        oi_color  = AMBER if r['oi'] >= 5000 else GREEN if r['oi'] >= 1000 else TEXT
        opt_rows += f"""
        <div style="display:grid;grid-template-columns:{OCOLS};
                    background:{bg};padding:7px 16px;align-items:center;border-bottom:1px solid #e2e8f0;">
          <div style="font-size:20px;font-weight:700;color:{TEXT};">{r['sec']}</div>
          <div style="font-size:18px;color:{DIM};text-align:right;">{r['month']}</div>
          <div style="font-size:20px;font-weight:700;color:{pc_color};text-align:center;">{r['pc']}</div>
          <div style="font-size:20px;font-weight:700;color:{oi_color};text-align:right;">{r['oi']:,}</div>
          <div style="font-size:20px;font-weight:700;color:{chg_color};text-align:right;">{fc(r['chg'])}</div>
          <div style="font-size:18px;color:{TEXT};text-align:right;">{f"{r['settle']:.2f}" if r['settle'] else "—"}</div>
          <div style="font-size:18px;color:{DIM};text-align:right;">{r['vol']:,}</div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:{PAGE}; font-family:'Segoe UI',sans-serif; width:1080px; }}
</style></head><body>

<!-- Header -->
<div style="background:{HDR};padding:12px 16px;display:flex;
            align-items:center;justify-content:space-between;
            border-bottom:2px solid {GOLD};">
  <div>
    <div style="font-size:16px;font-weight:700;letter-spacing:2px;color:#94a3b8;">VLM COMMODITIES</div>
    <div style="font-size:26px;font-weight:700;color:#f1f5f9;">{COMM_NAMES.get(comm, comm)} OPEN INTEREST MONITOR</div>
  </div>
  <div style="font-size:18px;font-weight:600;color:#cbd5e1;">As of: {as_of}</div>
</div>

<!-- SECTION 1: FUTURES -->
<div style="background:{SECT};padding:6px 16px;border-left:4px solid {GOLD};margin-top:2px;">
  <span style="font-size:16px;font-weight:700;letter-spacing:2px;color:{TEXT};">◆ {COMM_NAMES.get(comm, comm)} FUTURES — OPEN INTEREST</span>
</div>

<!-- Futures header -->
<div style="display:grid;grid-template-columns:190px 110px 175px 148px 165px 260px;
            background:{HDR};padding:5px 16px;border-bottom:1px solid #1e3a5f;">
  {''.join(f'<div style="font-size:16px;font-weight:700;color:#94a3b8;text-align:right;letter-spacing:.6px;">{h}</div>'
           for h in ['TICKER','FUT CONT','OPEN INT','OI CHG','SETTLE PX','1ST NOTICE'])}
</div>
{fut_rows}

<!-- SECTION 2: OPTIONS TOP 10 -->
<div style="background:{SECT};padding:6px 16px;border-left:4px solid {GREEN};margin-top:8px;">
  <span style="font-size:16px;font-weight:700;letter-spacing:2px;color:{TEXT};">◆ TOP 10 OPTIONS — LARGEST OI CHANGES (DAY OVER DAY)</span>
</div>

<!-- Options header -->
<div style="display:grid;grid-template-columns:252px 130px 62px 164px 152px 142px 146px;
            background:{HDR};padding:5px 16px;border-bottom:1px solid #1e3a5f;">
  {''.join(f'<div style="font-size:16px;font-weight:700;color:#94a3b8;text-align:right;letter-spacing:.6px;">{h}</div>'
           for h in ['TICKER','MONTH','P/C','OPEN INT','OI CHG','SETTLE','VOLUME'])}
</div>
{opt_rows}

<!-- Footer -->
<div style="background:{HDR};padding:8px 16px;border-top:1px solid #1e3a5f;margin-top:2px;
            display:flex;justify-content:space-between;align-items:center;">
  <span style="font-size:14px;color:#64748b;letter-spacing:1px;">VLM COMMODITIES LTD — BLOOMBERG EOD — UPDATES DAILY 09:35 EST</span>
  <span style="font-size:14px;color:#64748b;">{datetime.now().strftime('%B %d, %Y')}</span>
</div>

</body></html>"""
    return html


def render_png(html, png_path):
    """Render HTML to PNG using Playwright, cropped to content height."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print('ERROR: playwright not installed.')
        return None
    _CLIP_JS = """() => {
        let bottom = 0;
        for (const el of document.body.children) {
            const r = el.getBoundingClientRect();
            if (r.bottom > bottom) bottom = r.bottom;
        }
        return Math.ceil(bottom) + 8;
    }"""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(device_scale_factor=2)
        page    = context.new_page()
        page.set_viewport_size({'width': 1080, 'height': 1800})
        page.set_content(html, wait_until='networkidle')
        clip_h = page.evaluate(_CLIP_JS)
        clip_w = page.evaluate("() => Math.ceil(document.documentElement.scrollWidth)")
        page.screenshot(path=str(png_path), clip={'x': 0, 'y': 0, 'width': clip_w, 'height': clip_h})
        browser.close()
    return str(png_path)


def build_whatsapp_oi(output_base='output'):
    """Generate one PNG per commodity (CT, KC, CC, SB)."""
    # Get date from futures data
    all_rows = list(csv.DictReader(OI_FILE.open(encoding='utf-8')))
    as_of = max(r['date'] for r in all_rows)

    out_dir = pathlib.Path(output_base) / 'whatsapp' / as_of
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for comm in ['CT', 'KC', 'CC', 'SB']:
        futures, agg_oi, agg_chg, _ = load_futures(comm)
        if not futures:
            print(f'  No futures data for {comm} — skipping')
            continue
        opts, _ = load_options_top10(comm)
        html = build_html(futures, agg_oi, agg_chg, opts, as_of, comm)
        png_path = out_dir / f'OI_Monitor_{comm}_{as_of}.png'
        result = render_png(html, png_path)
        if result:
            print(f'  Saved: {png_path.name}')
            saved.append(result)

    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='output')
    args = parser.parse_args()
    paths = build_whatsapp_oi(args.output)
    if paths:
        import subprocess, sys as _sys
        folder = pathlib.Path(paths[0]).parent
        subprocess.Popen(f'explorer "{folder}"')

        try:
            from vlm_post import post_to_vlm
            as_of = folder.name
            if _sys.stdin.isatty():
                _ans = input(f'\nPost OI Monitor ({as_of}) to the VLM site? [y/N] ').strip().lower()
            else:
                _ans = 'n'
            if _ans == 'y':
                all_rows = list(csv.DictReader(OI_FILE.open(encoding='utf-8')))
                as_of_date = max(r['date'] for r in all_rows)
                site_html = ''
                for comm in ['CT', 'KC', 'CC', 'SB']:
                    futures, agg_oi, agg_chg, _ = load_futures(comm)
                    opts, _ = load_options_top10(comm)
                    site_html += build_html(futures, agg_oi, agg_chg, opts, as_of_date, comm)
                post_to_vlm(
                    title    = f'Open Interest Monitor — {as_of_date}',
                    content  = site_html,
                    category = 'oi',
                    excerpt  = f'VLM daily open interest monitor — as of {as_of_date}.',
                )
            else:
                print('[vlm_post] skipped')
        except Exception as _e:
            print(f'[vlm_post] skipped: {_e}')

if __name__ == '__main__':
    main()
