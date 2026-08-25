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

import csv, json, pathlib, argparse, re
from datetime import datetime, date, timedelta

BASE_DIR    = pathlib.Path(__file__).parent
DATA_DIR    = BASE_DIR / 'data'
OI_FILE     = DATA_DIR / 'oi_data.csv'
OPT_FILE    = DATA_DIR / 'options_oi.csv'

# Row order comes from first_notice (see load_futures._sort_key), NOT from a
# generic-ticker list -- rows are keyed by dated contract, so a generic's position
# is meaningless. TICKER_ORDERS below is retained only as a reference of the
# month cycle each commodity lists.

_MONTH_MAP = {
    'JAN':'Jan','FEB':'Feb','MAR':'Mar','APR':'Apr','MAY':'May','JUN':'Jun',
    'JUL':'Jul','AUG':'Aug','SEP':'Sep','OCT':'Oct','NOV':'Nov','DEC':'Dec',
}

_MONTHS_NUM = {1:'JAN',2:'FEB',3:'MAR',4:'APR',5:'MAY',6:'JUN',
               7:'JUL',8:'AUG',9:'SEP',10:'OCT',11:'NOV',12:'DEC'}

# Baseline offsets in TRADING SESSIONS present in oi_data.csv -- not calendar days,
# so holidays never silently skew a window.
WINDOWS = [('DoD', 1), ('WoW', 5), ('MoM', 21)]


def contract_key(commodity, first_notice, last_trade):
    """Map a row to its DATED contract (e.g. 'OCT26') from the row's own dates.

    Identical rule to build_prelim_oi.py's contract_key(), which reconciles 36/36
    against ICE's published change column. The anchor is commodity-specific and is
    NOT interchangeable:
      * CT/CC/KC -- last_trade falls IN the delivery month.
      * SB       -- cash-settle, LTD falls the month BEFORE delivery (SBOCT1
                    expires 09-30), so first_notice is the correct anchor.

    Why this exists at all: oi_data.csv stores rows under GENERIC tickers, which are
    pointers that re-aim at a different dated contract on roll days (verified: KCJUL1
    pointed at Jul-26 on 2026-07-20 and Jul-27 on 2026-07-21). Differencing a generic
    across a roll subtracts two different instruments -- Bloomberg's own KCN7 history
    shows the true flow was -46 while the stored oi_chg column reported +4,817.
    Keying by dated contract removes that class of error entirely.
    """
    anchor = first_notice if commodity == 'SB' else last_trade
    if not anchor:
        return None
    try:
        d = datetime.strptime(anchor, '%Y-%m-%d')
    except ValueError:
        return None
    return f'{_MONTHS_NUM[d.month]}{d.strftime("%y")}'


def _fmt_cont(key):
    """Dated contract key -> display label. 'OCT26' -> "Oct '26"."""
    if not key:
        return '—'
    return f"{_MONTH_MAP.get(key[:3], key[:3])} '{key[3:]}"


def _generation(contract):
    """Trailing slot digit of a month-specific generic (CCMAY1 -> 1)."""
    m = re.search(r'(\d+)$', contract or '')
    return int(m.group(1)) if m else 99


# Ticker order per commodity
TICKER_ORDERS = {
    'CT': ['CTMAR1','CTMAY1','CTJUL1','CTOCT1','CTDEC1','CTMAR2','CTMAY2','CTJUL2','CTOCT2','CTDEC2'],
    'KC': ['KCMAR1','KCMAY1','KCJUL1','KCSEP1','KCDEC1','KCMAR2','KCMAY2','KCJUL2','KCSEP2','KCDEC2'],
    'CC': ['CCMAR1','CCMAY1','CCJUL1','CCSEP1','CCDEC1','CCMAR2','CCMAY2','CCJUL2','CCSEP2','CCDEC2'],
    'SB': ['SBMAR1','SBMAY1','SBJUL1','SBOCT1','SBMAR2','SBMAY2','SBJUL2','SBOCT2'],
}

COMM_NAMES = {'CT': 'COTTON', 'KC': 'COFFEE', 'CC': 'COCOA', 'SB': 'SUGAR'}


def _int_or_none(v):
    """Blank/None/non-numeric -> None. Live data DOES carry blank open_int on a
    newly-listed back month (observed KCJUL2 on 2026-07-21, no OI and no dates),
    so every read of this column must tolerate it rather than assume 0."""
    if v in (None, '', 'None'):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def load_futures(comm='CT'):
    """Load the latest session's futures rows for a commodity, with DoD/WoW/MoM
    computed per DATED contract (see contract_key).

    Returns (rows, agg_oi, agg_chg_by_window, partial_by_window, last_date).
    A contract with no baseline in a window gets None for that window (renders as
    a dash) -- e.g. a back month listed more recently than 21 sessions ago. The
    window total then sums only the contracts that DO have a baseline and is
    flagged partial, rather than being blanked over one 1-lot back month.
    """
    rows = list(csv.DictReader(OI_FILE.open(encoding='utf-8')))
    last_date = max(r['date'] for r in rows)

    # Sessions present for THIS commodity -- baselines walk back in trading
    # sessions, so exchange holidays can never skew a window.
    sessions = sorted({r['date'] for r in rows if r['commodity'] == comm})
    prior = [s for s in sessions if s < last_date]
    baselines = {}
    for label, n in WINDOWS:
        baselines[label] = prior[-n] if len(prior) >= n else None

    # Index baseline sessions by dated contract. On a roll day two generics can
    # briefly map to the same dated contract; keep the LOWER generation (the nearer
    # generic is the live contract the label refers to).
    wanted = {d for d in baselines.values() if d}
    base_idx = {}
    for r in rows:
        if r['commodity'] != comm or r['date'] not in wanted:
            continue
        k = contract_key(comm, r.get('first_notice',''), r.get('last_trade',''))
        if not k:
            continue
        slot = (r['date'], k)
        cur = base_idx.get(slot)
        if cur is None or _generation(r['contract']) < _generation(cur['contract']):
            base_idx[slot] = r

    today = [r for r in rows if r['date'] == last_date and r['commodity'] == comm]
    result = []
    for r in today:
        oi = _int_or_none(r.get('open_int'))
        if oi is None:
            # No OI reported (newly-listed month Bloomberg has not populated yet).
            # Skipped rather than shown as 0, which would be a phantom row and
            # would understate nothing but mislead on the chain's contents.
            continue
        key = contract_key(comm, r.get('first_notice',''), r.get('last_trade',''))
        if key is None:
            # OI present but no FND/LTD to date it (8,624 such rows exist in the
            # 2008-era history; none on recent sessions). Without dates the row
            # cannot be identified, labelled, or compared to any baseline -- it
            # would render as a nameless '—' whose OI still inflated the total.
            # Dropped rather than shown as an unattributable number.
            continue
        chg = {}
        for label, _ in WINDOWS:
            bd = baselines[label]
            brow = base_idx.get((bd, key)) if (bd and key) else None
            bo = _int_or_none(brow.get('open_int')) if brow else None
            chg[label] = (oi - bo) if bo is not None else None
        result.append({
            'key':      key,
            'cont':     _fmt_cont(key),
            'oi':       oi,
            'chg':      chg,
            'settle':   float(r['settle']) if r.get('settle') not in (None,'','None') else 0,
            'fnd':      r.get('first_notice',''),
        })

    # Chronological by first notice (the delivery order traders read). Rows lacking
    # an FND sort last by delivery derived from the dated key, so ordering never
    # depends on the generic ticker.
    def _sort_key(x):
        if x['fnd']:
            return (0, x['fnd'], '')
        k = x['key'] or ''
        if len(k) == 5:
            mon = next((n for n, v in _MONTHS_NUM.items() if v == k[:3]), 99)
            return (1, f'20{k[3:]}-{mon:02d}', '')
        return (2, '', x['cont'])
    result.sort(key=_sort_key)

    agg_oi  = sum(r['oi'] for r in result)
    agg_chg = {lab: sum(r['chg'][lab] for r in result if r['chg'][lab] is not None)
               for lab, _ in WINDOWS}
    partial = {lab: any(r['chg'][lab] is None for r in result)
               for lab, _ in WINDOWS}
    return result, agg_oi, agg_chg, partial, last_date

def _next_bday(d_str):
    """Next business day (weekend-only shift, mirrors vlm_master_fetch.py's convention).
    Options rows are stamped with the RELEASE date = trade_date + 1 business day, while
    oi_data.csv is stamped with the TRADE date. To pair the same session, advance the
    futures trade date by one business day to hit the matching options release row."""
    d = datetime.strptime(d_str, '%Y-%m-%d').date() + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d.strftime('%Y-%m-%d')


def load_options_top10(comm='CT', target_date=None):
    """Load top 10 options by absolute OI change for a given commodity.
    target_date: the options RELEASE date (= futures trade date + 1 bday) so options and
    futures show the SAME trading session. Falls back to options max date (with a warning)
    if target_date not present in options CSV.
    """
    rows = list(csv.DictReader(OPT_FILE.open(encoding='utf-8')))
    available = sorted({r['date'] for r in rows})
    stale = False
    if target_date and target_date in available:
        last_date = target_date
    elif target_date and target_date not in available:
        # T+1 options release not landed yet — degrade LOUDLY: use the latest available
        # options session but flag it, so a stale-by-a-day panel is never shown as same-day.
        last_date = available[-1] if available else ''
        stale = True
        print(f'  WARNING: options CSV has no release for {target_date} '
              f'(latest: {last_date or "none"}) — showing latest available, flagged STALE')
    else:
        last_date = available[-1] if available else ''
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
    return parsed[:10], last_date, stale


def build_html(futures, agg_oi, agg_chg, partial, opts, as_of, comm='CT'):
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
        # None = no baseline in that window (contract listed more recently than the
        # lookback); renders as a neutral dash, never colored as a zero change.
        if v is None: return DIM
        if v > 0: return GREEN
        if v < 0: return RED
        return DIM

    # ── Futures rows ────────────────────────────────────────────────
    # One CONTRACT column (dated, e.g. "Oct '26") replaces the old TICKER + FUT CONT
    # pair: the generic ticker was a pointer that re-aimed on roll days, which is
    # exactly what made a generic-keyed delta wrong. Freed width funds WoW/MoM at
    # the same font size; 1ST NOTICE drops the century ('26-10-01') for the rest.
    FCOLS = '150px 168px 136px 136px 136px 128px 154px'
    def _short_fnd(d):
        return d[2:] if d and len(d) == 10 else (d or '—')
    fut_rows = ''
    for i, r in enumerate(futures):
        bg = ROW1 if i % 2 == 0 else ROW2
        oi_color  = AMBER if r['oi'] >= 50000 else GREEN if r['oi'] >= 10000 else TEXT
        chg_cells = ''.join(
            f'<div style="font-size:20px;font-weight:700;color:{color_chg(r["chg"][lab])};'
            f'text-align:right;">{fc(r["chg"][lab])}</div>'
            for lab, _ in WINDOWS)
        fut_rows += f"""
        <div style="display:grid;grid-template-columns:{FCOLS};
                    background:{bg};padding:7px 16px;align-items:center;border-bottom:1px solid #e2e8f0;">
          <div style="font-size:20px;font-weight:700;color:{TICKER};">{r['cont']}</div>
          <div style="font-size:20px;font-weight:700;color:{oi_color};text-align:right;">{r['oi']:,}</div>
          {chg_cells}
          <div style="font-size:20px;color:{TEXT};text-align:right;">{r['settle']:.2f}</div>
          <div style="font-size:18px;color:{DIM};text-align:right;">{_short_fnd(r['fnd'])}</div>
        </div>"""
    # ── Totals bar ──────────────────────────────────────────────────
    # "TOTAL (SHOWN)": this sums the months DISPLAYED, which is not the exchange
    # aggregate -- Bloomberg's SB chain on 2026-08-25 carried 4 further months
    # (Oct28/Mar29/May29/Jul29, 11,477 lots) beyond the 8 shown. Labelled so the
    # number cannot be misread as Aggr Open Int.
    # A window whose total excludes a contract with no baseline is marked with a
    # trailing asterisk rather than blanked -- one 1-lot back month must not wipe
    # out the most-watched number on the card.
    tot_cells = ''.join(
        f'<div style="font-size:20px;font-weight:700;color:{color_chg(agg_chg[lab])};'
        f'text-align:right;">{fc(agg_chg[lab])}{"*" if partial.get(lab) else ""}</div>'
        for lab, _ in WINDOWS)
    fut_rows += f"""
        <div style="display:grid;grid-template-columns:{FCOLS};
                    background:#1e293b;padding:8px 16px;align-items:center;">
          <div style="font-size:15px;font-weight:700;color:#f1f5f9;letter-spacing:1px;">TOTAL (SHOWN)</div>
          <div style="font-size:20px;font-weight:700;color:{GOLD};text-align:right;">{agg_oi:,}</div>
          {tot_cells}
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
<div style="display:grid;grid-template-columns:{FCOLS};
            background:{HDR};padding:5px 16px;border-bottom:1px solid #1e3a5f;">
  <div style="font-size:16px;font-weight:700;color:#94a3b8;text-align:left;letter-spacing:.6px;">CONTRACT</div>
  {''.join(f'<div style="font-size:16px;font-weight:700;color:#94a3b8;text-align:right;letter-spacing:.6px;">{h}</div>'
           for h in ['OPEN INT','DOD','WOW','MOM','SETTLE PX','1ST NOTICE'])}
</div>
{fut_rows}

<!-- SECTION 2: OPTIONS TOP 10 -->
<div style="background:{SECT};padding:6px 16px;border-left:4px solid {GREEN};margin-top:8px;">
  <span style="font-size:16px;font-weight:700;letter-spacing:2px;color:{TEXT};">◆ TOP 10 OPTIONS — LARGEST OI CHANGES (DAY OVER DAY)</span>
</div>

<!-- Options header -->
<div style="display:grid;grid-template-columns:{OCOLS};
            background:{HDR};padding:5px 16px;border-bottom:1px solid #1e3a5f;">
  {''.join(f'<div style="font-size:16px;font-weight:700;color:#94a3b8;'
           f'text-align:{a};letter-spacing:.6px;">{h}</div>'
           for h, a in [('TICKER','left'), ('MONTH','right'), ('P/C','center'),
                        ('OPEN INT','right'), ('OI CHG','right'),
                        ('SETTLE','right'), ('VOLUME','right')])}
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


def check_freshness(as_of, out_dir):
    """Freshness guard for the unattended 09:35 scheduled send.
    Two independent checks, both must pass or the send is skipped:
      1. Staleness: oi_data.csv must have been modified TODAY (calendar date).
         If 'vlm master fetch' (09:30 daily) failed or hasn't landed yet, the
         CSV's mtime will still be yesterday's — the file exists and has SOME
         max date, but it's not a fresh row, just the last-known one. Comparing
         only trade_date is not enough because OI is legitimately T+1 (today's
         run always shows yesterday's completed session) — mtime is what proves
         *today's* fetch actually touched the file.
      2. Idempotency: a `.sent_<date>` marker in the dated output folder means
         this session's images were already sent — guards against a double-fire
         of the scheduled task (or a second unattended run) re-sending the same
         PNGs to WhatsApp.
    Returns (ok: bool, reason: str).
    """
    mtime = date.fromtimestamp(OI_FILE.stat().st_mtime)
    if mtime != date.today():
        return False, (f'oi_data.csv last modified {mtime} (not today, {date.today()}) — '
                        f'master fetch has not updated it today, refusing to send stale data')
    marker = out_dir / f'.sent_{as_of}'
    if marker.exists():
        return False, f'already sent for {as_of} (marker: {marker.name}) — skipping duplicate send'
    return True, ''


def build_whatsapp_oi(output_base='output'):
    """Generate one PNG per commodity (CT, KC, CC, SB)."""
    # oi_data.csv is ALREADY trade-date-stamped (daily job writes the actual trade
    # date via `trade_date = max(all_dates)`, 2026-07 fix). Its max date IS the trade
    # date to display — do NOT subtract another business day (that double-shifted the
    # label, e.g. 07-06 -> 07-03 across the Jul-4 holiday). Use the max date directly.
    all_rows = list(csv.DictReader(OI_FILE.open(encoding='utf-8')))
    trade_date = max(r['date'] for r in all_rows)   # oi_data.csv is trade-date-stamped
    as_of = trade_date
    # Options rows are RELEASE-date-stamped (= trade_date + 1 bday). Target that release
    # so the options panel shows the SAME session as the futures panel (not yesterday's).
    opt_target = _next_bday(trade_date)

    out_dir = pathlib.Path(output_base) / 'whatsapp' / as_of
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for comm in ['CT', 'KC', 'CC', 'SB']:
        futures, agg_oi, agg_chg, partial, _ = load_futures(comm)
        if not futures:
            print(f'  No futures data for {comm} — skipping')
            continue
        opts, _, _ = load_options_top10(comm, target_date=opt_target)
        html = build_html(futures, agg_oi, agg_chg, partial, opts, as_of, comm)
        png_path = out_dir / f'OI_Monitor_{comm}_{as_of}.png'
        result = render_png(html, png_path)
        if result:
            print(f'  Saved: {png_path.name}')
            saved.append(result)

    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='output')
    parser.add_argument('--force', action='store_true',
                         help='Bypass the freshness/idempotency guard and send anyway.')
    args = parser.parse_args()
    paths = build_whatsapp_oi(args.output)
    if not paths:
        return

    import subprocess
    folder = pathlib.Path(paths[0]).parent
    subprocess.Popen(f'explorer "{folder}"')

    # Freshness guard: only gates the SEND step, never image generation, so a
    # manual re-run always regenerates PNGs for review even if sending would be
    # blocked. Use --force to send anyway (e.g. after manually confirming stale
    # data is actually fine).
    as_of = folder.name
    ok, reason = check_freshness(as_of, folder)
    if not ok and not args.force:
        print(f'\n[freshness guard] SEND SKIPPED: {reason}')
        print('  (images were still generated above; re-run with --force to send anyway)')
        return

    # Each send path is isolated in its own try/except so a failure in one
    # (Twilio outage, R2 credential issue, site down) never blocks the others.
    # Failures accumulate into _send_failures purely for the end-of-run summary
    # printed to the terminal — nothing here prompts or blocks unattended runs.
    _send_failures = []

    print()
    try:
        from send_oi_whatsapp import main as _send_wa
        _summary = _send_wa(whatsapp_dir=str(folder))
        if _summary.get('failed'):
            _send_failures.append(f"whatsapp send: {_summary['errors']}")
    except Exception as _e:
        print(f'[whatsapp] send FAILED: {_e}')
        _send_failures.append(f"whatsapp send: {_e}")

    try:
        from vlm_post import post_to_vlm
        as_of = folder.name
        all_rows = list(csv.DictReader(OI_FILE.open(encoding='utf-8')))
        as_of_date = max(r['date'] for r in all_rows)   # futures trade date
        opt_target = _next_bday(as_of_date)              # options release date
        site_html = ''
        for comm in ['CT', 'KC', 'CC', 'SB']:
            futures, agg_oi, agg_chg, partial, _ = load_futures(comm)
            if not futures:
                # Mirror build_whatsapp_oi()'s skip: no futures data that day (holiday/
                # gap) means no PNG was generated/sent for this commodity either, so the
                # site post must not show an empty "TOTAL 0" section for it.
                continue
            opts, _, _ = load_options_top10(comm, target_date=opt_target)
            site_html += build_html(futures, agg_oi, agg_chg, partial, opts, as_of_date, comm)
        post_to_vlm(
            title    = f'Open Interest Monitor — {as_of_date}',
            content  = site_html,
            category = 'oi',
            excerpt  = f'VLM daily open interest monitor — as of {as_of_date}.',
        )
    except Exception as _e:
        print(f'[vlm_post] FAILED: {_e}')
        _send_failures.append(f"vlm site post: {_e}")

    if _send_failures:
        print(f'\n{len(_send_failures)} send path(s) failed:')
        for _f in _send_failures:
            print(f'  - {_f}')
        # No marker on failure — a genuine Twilio/site outage should stay retryable
        # (via --force) rather than getting permanently silenced by the idempotency
        # guard once whichever cause is fixed.
    else:
        print('\nAll send paths OK (WhatsApp + site post).')
        (folder / f'.sent_{as_of}').write_text(
            f'sent {datetime.now().isoformat()}\n', encoding='utf-8')

if __name__ == '__main__':
    main()
