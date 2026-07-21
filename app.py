# app.py — VLM Open Interest Monitor  v5
# Run:  python app.py   ->  http://127.0.0.1:8052
# Production: gunicorn entry point is  app:server

__version__ = "2026.04.21-oi-v6"
import sys; sys.dont_write_bytecode = True
import json, pathlib, csv
from datetime import datetime
from flask import Flask, jsonify
from vlm_auth import install_gate

BASE_DIR  = pathlib.Path(__file__).parent
DATA_FILE    = BASE_DIR / 'data' / 'oi_data.csv'
OPT_FILE     = BASE_DIR / 'data' / 'options_oi.csv'
CSS_FILE  = BASE_DIR / 'vlm_design_system.css'

app    = Flask(__name__)
install_gate(app, gated_paths=("/",))  # VLM auth gate
server = app

# ── Contract code -> human label  (CTK6 -> "MAY 26") ────────────────────────────
_MC_OLD = {'F':'JAN','G':'FEB','H':'MAR','J':'APR','K':'MAY','M':'JUN',
           'N':'JUL','Q':'AUG','U':'SEP','V':'OCT','X':'NOV','Z':'DEC'}
_MONTHS_LIST = ['DEC','NOV','OCT','SEP','AUG','JUL','JUN','MAY','APR','MAR','FEB','JAN']

def contract_label(code, first_notice=''):
    """
    Handles two formats:
      Generic monthly: CTDEC1 -> 'DEC 1 \'26', CTJUL2 -> 'JUL 2 \'26'
      Specific contract: CTZ26 -> 'DEC 26', CTN16 -> 'JUL 16'
    Year is derived from first_notice date when available.
    """
    if not code or len(code) < 3:
        return code or '—'
    code = code.strip().upper()
    yr = f" '{first_notice[2:4]}" if first_notice and len(first_notice) >= 4 else ''
    # New format: 2-char commodity + 3-char month + slot digit (1 or 2)
    for comm in ('CT', 'SB', 'KC', 'CC'):
        if code.startswith(comm):
            rest = code[len(comm):]
            for mo in _MONTHS_LIST:
                if rest == mo + '1': return f'{mo} 1{yr}'
                if rest == mo + '2': return f'{mo} 2{yr}'
    # Old specific format: CTZ26, CTK6, SBN26 etc
    try:
        i = len(code) - 1
        while i >= 0 and code[i].isdigit(): i -= 1
        mo  = _MC_OLD.get(code[i], '???')
        yr_str = code[i+1:]
        yr  = int(yr_str)
        fy  = 2000 + yr if len(yr_str) == 2 else 2020 + yr if yr <= 9 else 2010 + yr
        return f'{mo} {str(fy)[2:]}'
    except Exception:
        return code


# ── Data loader ──────────────────────────────────────────────────────────────────
def load_data():
    if not DATA_FILE.exists():
        return None

    raw_rows = {}
    with open(DATA_FILE, 'r', encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            try:
                key = (r['date'], r['bbg_ticker'])
                raw_rows[key] = {
                    'date':         r['date'],
                    'commodity':    r['commodity'],
                    'contract':     r['contract'],
                    'bbg_ticker':   r['bbg_ticker'],
                    'settle':       float(r['settle'])  if r.get('settle')   else None,
                    'open_int':     int(r['open_int'])  if r.get('open_int') else None,
                    'oi_chg':       int(r['oi_chg'])    if r.get('oi_chg')   else None,
                    'first_notice': r.get('first_notice', ''),
                    'last_trade':   r.get('last_trade',   ''),
                }
            except Exception:
                continue
    rows = list(raw_rows.values())

    if not rows:
        return None

    last_date = max(r['date'] for r in rows)
    today_dt  = datetime.strptime(last_date, '%Y-%m-%d')

    TICKER_ORDER = {
        'CT': ['CTMAY1 Comdty','CTJUL1 Comdty','CTOCT1 Comdty','CTDEC1 Comdty','CTMAR1 Comdty',
               'CTMAY2 Comdty','CTJUL2 Comdty','CTOCT2 Comdty','CTDEC2 Comdty','CTMAR2 Comdty'],
        'SB': ['SBMAY1 Comdty','SBJUL1 Comdty','SBOCT1 Comdty','SBMAR1 Comdty',
               'SBMAY2 Comdty','SBJUL2 Comdty','SBOCT2 Comdty','SBMAR2 Comdty'],
        'KC': ['KCMAY1 Comdty','KCJUL1 Comdty','KCSEP1 Comdty','KCDEC1 Comdty','KCMAR1 Comdty',
               'KCMAY2 Comdty','KCJUL2 Comdty','KCSEP2 Comdty','KCDEC2 Comdty','KCMAR2 Comdty'],
        'CC': ['CCMAY1 Comdty','CCJUL1 Comdty','CCSEP1 Comdty','CCDEC1 Comdty','CCMAR1 Comdty',
               'CCMAY2 Comdty','CCJUL2 Comdty','CCSEP2 Comdty','CCDEC2 Comdty','CCMAR2 Comdty'],
    }

    comm_data = {}
    for comm in ['CT','SB','KC','CC']:
        comm_rows = [r for r in rows if r['commodity'] == comm]
        if not comm_rows:
            continue

        def tk_yr_range(tk_history, years):
            cutoff = today_dt.toordinal() - int(365 * years)
            vals = [h['open_int'] for h in tk_history
                    if h['open_int'] and datetime.strptime(h['date'],'%Y-%m-%d').toordinal() >= cutoff]
            return (min(vals), max(vals)) if vals else (0, 0)

        tickers = {}
        for r in comm_rows:
            tk = r['bbg_ticker']
            if tk not in tickers:
                tickers[tk] = {
                    'label':        tk.replace(' Comdty',''),
                    'contract':     r['contract'],
                    'contract_lbl': contract_label(r['contract'], r.get('first_notice','')),
                    'settle':       None, 'open_int': None, 'oi_chg': None,
                    'first_notice': '', 'history': [],
                }
            tickers[tk]['history'].append({
                'date': r['date'], 'settle': r['settle'],
                'open_int': r['open_int'], 'oi_chg': r['oi_chg'],
            })
            if r['date'] == last_date:
                tickers[tk].update({
                    'contract':     r['contract'],
                    'contract_lbl': contract_label(r['contract'], r.get('first_notice','')),
                    'settle':       r['settle'],
                    'open_int':     r['open_int'],
                    'oi_chg':       r['oi_chg'],
                    'first_notice': r.get('first_notice', ''),
                    'last_trade':   r.get('last_trade',   ''),
                })
            tickers[tk]['last_row_first_notice'] = r.get('first_notice','') or tickers[tk].get('last_row_first_notice','')

        for tk in tickers:
            tickers[tk]['history'].sort(key=lambda x: x['date'])
            tlo5, thi5   = tk_yr_range(tickers[tk]['history'], 5)
            tlo15, thi15 = tk_yr_range(tickers[tk]['history'], 15)
            tickers[tk]['tk_lo5']  = tlo5
            tickers[tk]['tk_hi5']  = thi5
            tickers[tk]['tk_lo15'] = tlo15
            tickers[tk]['tk_hi15'] = thi15
            # B3: settle day-change (today - prior session) from the two latest
            # history rows with a numeric settle. Powers the OI-vs-price conviction
            # read (new longs / new shorts / short covering / long liquidation).
            # None when <2 priced sessions exist. Additive field, safe for consumers.
            settle_chg = None
            priced = [h for h in tickers[tk]['history']
                      if h.get('settle') not in (None, '')]
            if len(priced) >= 2:
                try:
                    settle_chg = float(priced[-1]['settle']) - float(priced[-2]['settle'])
                except (TypeError, ValueError):
                    settle_chg = None
            tickers[tk]['settle_chg'] = settle_chg

        ordered = [tk for tk in TICKER_ORDER.get(comm, []) if tk in tickers]
        ordered += [tk for tk in tickers if tk not in ordered]
        tk_order = {tk: i for i, tk in enumerate(TICKER_ORDER.get(comm, []))}
        ordered.sort(key=lambda tk: (
            tickers[tk].get('first_notice','') or '9999',
            tk_order.get(tk, 99)
        ))

        last_rows = [r for r in comm_rows if r['date'] == last_date]
        agg_oi    = sum(r['open_int'] for r in last_rows if r['open_int'] is not None)
        agg_chg   = sum(r['oi_chg']   for r in last_rows if r['oi_chg']   is not None)

        from collections import defaultdict
        daily_agg    = defaultdict(int)
        for r in comm_rows:
            if r['open_int'] is not None:
                daily_agg[r['date']] += r['open_int']
        sorted_dates = sorted(daily_agg.keys())

        def yr_range_agg(years):
            vals = [daily_agg[d] for d in sorted_dates
                    if (today_dt - datetime.strptime(d, '%Y-%m-%d')).days <= 365*years]
            return (min(vals), max(vals)) if vals else (0, 0)

        # Aggregate ranges for sparkline display
        lo5,  hi5  = yr_range_agg(5)
        lo15, hi15 = yr_range_agg(15)
        sparkline  = [daily_agg[d] for d in sorted_dates[-252:]]

        if sorted_dates:
            first_dt  = datetime.strptime(sorted_dates[0], '%Y-%m-%d')
            max_years = max(1, round((today_dt - first_dt).days / 365))
        else:
            max_years = 18

        # Inline JSON — only current-year ticker history (monitor grid use)
        # Full history fetched on demand via /api/history/<comm>
        cur_year = last_date[:4]
        ticker_history_inline = {}
        for tk in ordered:
            lbl = tk.replace(' Comdty', '')
            ticker_history_inline[lbl] = [
                {'date': h['date'], 'open_int': h['open_int'],
                 'oi_chg': h['oi_chg'], 'settle': h['settle']}
                for h in tickers[tk]['history']
                if h['date'][:4] == cur_year
            ]

        # 5yr daily_agg for inline seasonal fallback
        cutoff_5y        = sorted_dates[-1825:] if len(sorted_dates) > 1825 else sorted_dates
        daily_agg_inline = {d: daily_agg[d] for d in cutoff_5y}

        comm_data[comm] = {
            'tickers':       {tk: {k: v for k, v in tickers[tk].items() if k != 'history'}
                              for tk in ordered},
            'ordered':       ordered,
            'agg_oi':        agg_oi,
            'agg_chg':       agg_chg,
            'lo5': lo5, 'hi5': hi5, 'lo15': lo15, 'hi15': hi15,
            'sparkline':     sparkline,
            'daily_agg':     daily_agg_inline,
            'ticker_history': ticker_history_inline,
            'max_years':     max_years,
        }

    return {'last_date': last_date, 'commodities': comm_data}


def load_css():
    return CSS_FILE.read_text(encoding='utf-8') if CSS_FILE.exists() else ''


# ── HTML template ────────────────────────────────────────────────────────────────
# Note: uses %%CSS%%, %%DATA%%, %%VERSION%% as placeholders (not {{ }} to avoid
# conflicts with JS template literals and CSS variable syntax)
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>OI Monitor — VLM</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
%%CSS%%

/* OI dashboard overrides */
body { color: var(--text); font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important; }
* { font-family: inherit; }
.vlm-logo     { color: var(--text) !important; }
.vlm-logo-sub { color: var(--dim)  !important; letter-spacing:2px; }
.vlm-asof     { color: var(--dim)  !important; }
.vlm-topbar-meta { color: var(--dim) !important; }
.vlm-theme-btn { color: var(--dim) !important; font-weight:600; }
.vlm-theme-btn:hover { color: var(--text) !important; border-color:var(--text) !important; }

/* Tabs — no background change, just gold underline */
.vlm-secnav { background:var(--hdr) !important; border-bottom:1px solid var(--bord) !important;
              display:flex; align-items:center; }
.vlm-sectab { color:var(--muted) !important; font-weight:700; letter-spacing:2px;
              padding:10px 18px !important;
              border:none !important; border-bottom:2px solid transparent !important;
              background:none !important; outline:none !important; box-shadow:none !important; }
.vlm-sectab:hover { color:var(--dim) !important; background:none !important; }
.vlm-sectab.act   { color:var(--acc) !important; border-bottom-color:var(--acc) !important;
                    background:none !important; }
.png-btn { font-size:10px; padding:3px 10px; border-radius:5px; cursor:pointer;
           background:rgba(212,180,74,.12); border:1px solid rgba(212,180,74,.35);
           color:#d4b44a; font-family:inherit; font-weight:600; letter-spacing:.04em;
           white-space:nowrap; transition:background .12s,border-color .12s; }
.png-btn:hover { background:rgba(212,180,74,.28); border-color:#d4b44a; }

/* Buttons */
.vlm-btn { color:var(--dim) !important; font-weight:600; letter-spacing:1px; }
.vlm-btn:hover { color:var(--text) !important; border-color:var(--dim) !important; }
.vlm-btn.act { color:#fff !important; background:#1e3a5f !important; border-color:var(--blue) !important; }
.vlm-btn.act { color:#fff !important; background:#1e40af !important; border-color:#1e40af !important; }
body.dark .vlm-btn.act { color:var(--text) !important; background:#1e2a3f !important; border-color:#4a6080 !important; }
.vlm-pos { color:var(--grn) !important; font-weight:700; }
.vlm-neg { color:var(--red) !important; font-weight:700; }
.vlm-muted { color:var(--muted) !important; }
select.oi-sel { color:var(--text) !important; font-weight:600; }
select.oi-sel option { background:var(--surf2); color:var(--text); }
input.d-inp { color:var(--text) !important; }
select.oi-sel option { background:#fff; color:#1a202c; }
body.dark select.oi-sel option { background:var(--surf2); color:var(--text); }

/* Year range slider */
.yr-slider-wrap { display:flex; align-items:center; gap:8px; }
.yr-slider { -webkit-appearance:none; appearance:none; width:130px; height:4px;
             background:var(--bord2); border-radius:2px; outline:none; cursor:pointer; }
.yr-slider::-webkit-slider-thumb { -webkit-appearance:none; width:14px; height:14px;
             border-radius:50%; background:var(--acc); cursor:pointer; }
.yr-label { font-size:11px; font-weight:700; color:var(--acc); min-width:34px; letter-spacing:1px; }

/* Layout */
.main-wrap { max-width:1600px; margin:0 auto; }
.tab-content { display:none; }
.tab-content.act { display:block; }

/* Monitor grid — 12 columns (Lo merged into Hi cell) */
.G { display:grid;
     grid-template-columns:140px 90px 100px 84px 90px 110px 110px minmax(110px,1fr) 150px 100px 120px;
     gap:0; }
.oi-badge { display:inline-flex;align-items:center;gap:4px;border-radius:3px;padding:3px 8px;font-size:10px;font-weight:700;letter-spacing:.6px;white-space:nowrap;border-width:1px;border-style:solid; }
.grid-head { background:var(--hdr); border-bottom:1px solid var(--acc);
             padding:4px 14px; position:sticky; top:49px; z-index:9; }
.gh { color:var(--dim); font-size:12px; font-weight:700; letter-spacing:.5px;
      text-transform:uppercase; text-align:right; padding:4px 6px;
      white-space:nowrap; overflow:hidden; }
.gh:first-child { text-align:left; }

.agg-row { padding:10px 14px; cursor:pointer; align-items:center;
           background:var(--surf); border-bottom:1px solid var(--bord);
           transition:background .1s; }
.agg-row:hover { background:var(--surf2); }
.ct-row  { padding:8px 14px; align-items:center; background:var(--bg);
           border-bottom:1px solid #1a2030; cursor:pointer; transition:background .1s; }
.ct-row:hover { background:var(--surf2); }
.ct-row.sel   { background:#0f1e35; border-left:3px solid var(--acc); }

.c  { font-size:15px; font-weight:700; text-align:right; padding:3px 6px;
      white-space:nowrap; overflow:hidden; color:var(--text); }
.cl { text-align:left; padding:3px 6px; font-size:15px; }
.ticker-lbl { display:flex; align-items:center; gap:5px;
              font-size:16px; font-weight:700; letter-spacing:1px; }
.ticker-sub { font-size:11px; font-weight:600; color:var(--muted);
              margin-left:3px; letter-spacing:.5px; }
.arr { font-size:9px; color:var(--muted); transition:transform .2s;
       display:inline-block; line-height:1; }
.arr.open { transform:rotate(90deg); }
.ctlbl    { color:var(--muted); font-size:12px; font-weight:700;
            letter-spacing:1px; padding-left:14px; }
.ct-month { color:var(--dim); font-size:11px; font-weight:600; margin-left:5px; }
.fn { color:var(--muted); font-size:11px; font-weight:600; letter-spacing:.5px; }

.spark-wrap { position:relative; width:100%; height:28px; cursor:crosshair; }
.spark-wrap svg rect { filter: brightness(0.6); }
body.dark .spark-wrap svg rect { filter: none; }
.oi-tooltip { display:none; position:fixed; background:var(--surf);
              border:1px solid var(--bord2); border-radius:3px; padding:5px 10px;
              font-size:11px; white-space:nowrap; z-index:999;
              color:var(--text); pointer-events:none;
              box-shadow:0 2px 12px rgba(0,0,0,.5); }

/* Chart panel */
.chart-panel { background:var(--surf); border-top:2px solid var(--acc);
               padding:14px 16px; display:none; }
.chart-panel.open { display:block; }
.chart-hdr { display:flex; align-items:center; gap:8px;
             margin-bottom:10px; flex-wrap:wrap; }
.chart-ttl { color:var(--acc); font-size:12px; font-weight:700;
             letter-spacing:2px; text-transform:uppercase; }
.d-row { display:flex; align-items:center; gap:6px; flex-wrap:wrap; }
.d-inp { background:var(--surf2); border:1px solid var(--bord); color:var(--text);
         padding:3px 7px; font-size:10px; font-family:inherit;
         border-radius:2px; width:118px; }
.cw  { position:relative; width:100%; height:240px; margin-top:8px; }
.leg-row { display:flex; gap:14px; flex-wrap:wrap; margin-bottom:4px; }
.leg { display:flex; align-items:center; gap:5px; font-size:11px;
       color:var(--dim); letter-spacing:.5px; font-weight:600; }
.lsw   { width:22px; height:2px; }
.lsw-b { width:22px; height:9px; border-radius:2px; }

/* Seasonal */
.seas-outer { padding:14px 16px; }
.seas-top   { display:flex; align-items:center; gap:8px;
              margin-bottom:14px; flex-wrap:wrap; }
.seas-grid  { display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:14px; }
.seas-card  { background:var(--surf); border:1px solid var(--bord);
              border-radius:3px; padding:12px; }
.scw { position:relative; width:100%; height:200px; }

/* Seasonal individual-years layouts (spaghetti rail + grid small-multiples) */
.seas-chart-row { display:grid; grid-template-columns:1fr 160px; gap:12px; align-items:stretch; }
.seas-chart-row.no-rail { grid-template-columns:1fr; }
.seas-rail { background:var(--hdr); border:1px solid var(--bord); border-radius:4px;
             padding:8px 6px; font-size:11px; font-variant-numeric:tabular-nums;
             max-height:360px; overflow:auto; }
.seas-rail-h { font-size:9px; color:var(--muted); letter-spacing:1px; text-transform:uppercase;
               margin-bottom:6px; padding-bottom:4px; border-bottom:1px solid var(--bord); }
.seas-rail-row { display:flex; justify-content:space-between; padding:2px 3px; border-radius:3px; }
.seas-rail-row .rr-yr { font-weight:700; }
.seas-rail-row .rr-val { color:var(--dim); }
.seas-sm-grid { display:grid; grid-template-columns:repeat(6,1fr); gap:10px; }
.seas-sm-cell { background:var(--surf2,var(--surf)); border:1px solid var(--bord); border-radius:4px;
                padding:8px 8px 6px; position:relative; }
.seas-sm-yr { font-size:12px; font-weight:800; letter-spacing:.5px; margin-bottom:2px; }
.seas-sm-sub { font-size:10px; color:var(--muted); margin-bottom:4px; }
.seas-sm-hover { position:absolute; top:4px; right:6px; font-size:11px; font-weight:700;
                 font-variant-numeric:tabular-nums; text-align:right; line-height:1.25;
                 pointer-events:none; opacity:0; transition:opacity .08s; }
.seas-sm-hover.on { opacity:1; }
.seas-sm-cw { position:relative; height:92px; }
@media (max-width:900px){ .seas-sm-grid { grid-template-columns:repeat(3,1fr); }
                          .seas-chart-row { grid-template-columns:1fr; }
                          .seas-rail { display:none!important; } }

/* Table */
.tbl-outer { padding:14px 16px; }
.tbl-top   { display:flex; align-items:center; gap:8px;
             margin-bottom:12px; flex-wrap:wrap; }
.oi-sel { background:var(--surf2); border:1px solid var(--bord); color:var(--text);
          padding:5px 9px; font-size:11px; font-family:inherit;
          letter-spacing:1px; border-radius:2px; }
.tbl-scroll { overflow-x:auto; border:1px solid var(--bord); border-radius:3px; }
.htbl { border-collapse:collapse; width:100%; }
.htbl th { background:var(--hdr); color:var(--dim); font-size:11px; font-weight:700;
           letter-spacing:1px; padding:5px 10px; text-align:right;
           border-bottom:1px solid var(--acc); white-space:nowrap;
           text-transform:uppercase; }
.htbl th:first-child { text-align:left; }
.htbl td { padding:5px 10px; text-align:right; border-bottom:1px solid #1a2030;
           color:var(--text); font-size:13px; font-weight:600; }
.htbl td:first-child { text-align:left; color:var(--muted); font-size:12px; }
.htbl tr:hover td { background:var(--surf2); }

.ct-row { background:var(--surf2); border-bottom:1px solid var(--bord); }
.ct-row:hover { background:var(--hdr); }
.ct-row.sel { background:#dce8ff; border-left:3px solid var(--acc); }
.htbl td { border-bottom:1px solid var(--bord); }
body.dark .ct-row { background:var(--bg); }
body.dark .ct-row:hover { background:var(--surf2); }
body.dark .ct-row.sel { background:#0f1e35; }
body.dark .htbl td { border-bottom:1px solid #1a2030; }

@media(max-width:800px){
  .G { grid-template-columns:90px 72px 72px 76px 1fr 64px 54px !important; }
  .hm { display:none !important; }
  .cw,.scw { height:180px; }
  .seas-grid { grid-template-columns:1fr; }
  .grid-head { top:44px; }
}
</style>
</head>
<body>
<div class="main-wrap">
<div class="oi-tooltip" id="oiTip"></div>

<div class="vlm-topbar">
  <div><div class="vlm-logo">VLM</div><div class="vlm-logo-sub">OPEN INTEREST MONITOR</div></div>
  <div class="vlm-topbar-meta">
    <div class="vlm-dot"></div>
    <span class="vlm-asof" id="asof">Loading...</span>
    <span class="vlm-asof" id="clk"></span>
    <a href="https://vlmdata.com" target="_blank" rel="noopener" style="color:var(--acc);text-decoration:none;font-size:11px;letter-spacing:.05em;font-family:inherit;">&#8592; vlmdata.com</a>
    <button class="vlm-theme-btn" id="themeBtn" onclick="toggleTheme()">LIGHT</button>
  </div>
</div>

<div class="vlm-secnav">
  <button class="vlm-sectab act" onclick="switchTab('monitor',this)">Monitor</button>
  <button class="vlm-sectab"     onclick="switchTab('seasonal',this)">Seasonal</button>
  <button class="vlm-sectab"     onclick="switchTab('table',this)">Table</button>
  <button class="vlm-sectab"     onclick="switchTab('options',this)">Options</button>
  <div style="flex:1;"></div>
  <button class="png-btn" onclick="exportCurrentTabPng()" style="margin-right:12px;">&#8659; PNG</button>
</div>

<!-- MONITOR -->
<div class="tab-content act" id="tab-monitor">
  <div id="monBody"></div>
</div>

<!-- SEASONAL -->
<div class="tab-content" id="tab-seasonal">
  <div class="seas-outer">
    <div class="seas-top">
      <span class="vlm-sec-title" style="margin:0;">Seasonal OI</span>
      <div class="yr-slider-wrap">
        <span style="font-size:10px;letter-spacing:1px;color:var(--muted);">YRS</span>
        <input type="range" class="yr-slider" id="seasYrSlider"
               min="1" max="18" value="5" step="1"
               oninput="onSeasYrSlide(this)">
        <span class="yr-label" id="seasYrLbl">5 YR</span>
      </div>
      <div class="vlm-ctrl-btns" style="padding:0;" id="seasModeBtns">
        <button class="vlm-btn act" onclick="setSeasMode('band',this)">HI / AVG / LO</button>
        <button class="vlm-btn"     onclick="setSeasMode('individual',this)">INDIVIDUAL YRS</button>
      </div>
      <div class="vlm-ctrl-btns" style="padding:0;display:none;" id="seasLayoutBtns">
        <button class="vlm-btn act" onclick="setSeasLayout('spaghetti',this)">SPAGHETTI</button>
        <button class="vlm-btn"     onclick="setSeasLayout('grid',this)">GRID</button>
      </div>
      <select class="oi-sel" id="seasSingleComm"
              onchange="onSeasCommChange()">
        <option value="CT">COTTON (CT)</option>
        <option value="SB">SUGAR (SB)</option>
        <option value="KC">COFFEE (KC)</option>
        <option value="CC">COCOA (CC)</option>
      </select>
      <select class="oi-sel" id="seasContract" onchange="buildSeasonal()">
        <option value="Aggregate">AGGREGATE</option>
      </select>
    </div>
    <div class="seas-grid" id="seasGrid"></div>
    <div id="seasSingle" style="display:none;">
      <div class="seas-card">
        <div id="seasSingleHdr" style="margin-bottom:8px;"></div>
        <div class="leg-row" id="seasSingleLeg" style="margin-bottom:6px;"></div>
        <!-- band / spaghetti share the single canvas; spaghetti adds a hover rail beside it -->
        <div id="seasChartRow" class="seas-chart-row">
          <div class="scw" style="height:360px;">
            <canvas id="scSingle" role="img" aria-label="Seasonal OI chart">Seasonal OI.</canvas>
          </div>
          <div class="seas-rail" id="seasRail" style="display:none;">
            <div class="seas-rail-h" id="seasRailHdr">hover a month</div>
            <div id="seasRailBody"></div>
          </div>
        </div>
        <!-- grid layout: small-multiples, one panel per prior year -->
        <div class="seas-sm-grid" id="seasSmGrid" style="display:none;"></div>
      </div>
    </div>
  </div>
</div>

<!-- OPTIONS -->
<div class="tab-content" id="tab-options">
  <div id="optRoot" style="padding:10px 14px;"></div>
</div>

<!-- TABLE -->
<div class="tab-content" id="tab-table">
  <div class="tbl-outer">
    <div class="tbl-top">
      <span class="vlm-sec-title" style="margin:0;">Historical OI Table</span>
      <select class="oi-sel" id="tblComm" onchange="onTblCommChange()">
        <option value="CT">COTTON (CT)</option>
        <option value="SB">SUGAR (SB)</option>
        <option value="KC">COFFEE (KC)</option>
        <option value="CC">COCOA (CC)</option>
      </select>
      <select class="oi-sel" id="tblContract" onchange="buildTbl()">
        <option value="Aggregate">AGGREGATE</option>
      </select>
      <div class="vlm-ctrl-btns" style="padding:0;" id="freqBtns">
        <button class="vlm-btn act" onclick="setFreq('daily',this)">DAILY</button>
        <button class="vlm-btn"     onclick="setFreq('weekly',this)">WEEKLY</button>
        <button class="vlm-btn"     onclick="setFreq('monthly',this)">MONTHLY</button>
      </div>
      <div class="d-row">
        <span style="font-size:10px;letter-spacing:1px;color:var(--muted);">FROM</span>
        <input class="d-inp" type="date" id="tblFrom" value="2023-01-01">
        <span style="font-size:10px;letter-spacing:1px;color:var(--muted);">TO</span>
        <input class="d-inp" type="date" id="tblTo">
        <button class="vlm-btn" onclick="buildTbl()">GO</button>
      </div>
    </div>
    <div class="tbl-scroll"><table class="htbl" id="histTbl"></table></div>
  </div>
</div>

</div><!-- /main-wrap -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
<div class="vlm-footer">VLM COMMODITIES — OI MONITOR &nbsp;·&nbsp; BLOOMBERG EOD &nbsp;·&nbsp;
  UPDATES DAILY 09:35 EST &nbsp;·&nbsp; v%%VERSION%% &nbsp;·&nbsp;
  <a href="https://vlmdata.com" target="_blank" rel="noopener" style="color:var(--acc);text-decoration:none;">vlmdata.com</a></div>

<script>
const DATA = %%DATA%%;
const OPTDATA = %%OPTDATA%%;
const CFG = {
  CT: {name:'COTTON NO.2', color:'#E8C547', ca:'rgba(232,197,71,0.12)'},
  SB: {name:'SUGAR NO.11', color:'#E07B54', ca:'rgba(224,123,84,0.12)'},
  KC: {name:'COFFEE C',    color:'#C8956D', ca:'rgba(200,149,109,0.12)'},
  CC: {name:'COCOA',       color:'#A07855', ca:'rgba(160,120,85,0.12)'},
};
const COMMS = ['CT','SB','KC','CC'];

/* ── Single state block — NO duplicates ── */
let expanded  = null;
let selKey    = null;
let cMode     = 'seasonal';
let seasYr    = 5;
let seasMode   = 'band';
let seasLayout = 'spaghetti';   // 'spaghetti' | 'grid' (only applies when seasMode==='individual')
let tblFreq   = 'daily';
const CH = {};
const HIST = {};

/* ── Format contract code for display: CTN6 -> "CTJUL 26" ── */
function fmtContract(code) {
  if (!code) return '—';
  code = code.trim().toUpperCase();
  var MC = {F:'JAN',G:'FEB',H:'MAR',J:'APR',K:'MAY',M:'JUN',
            N:'JUL',Q:'AUG',U:'SEP',V:'OCT',X:'NOV',Z:'DEC'};
  var i = code.length - 1;
  while (i >= 0 && /[0-9]/.test(code[i])) i--;
  var mo = MC[code[i]];
  if (!mo) return code;
  var prefix = code.slice(0, i);
  var yrStr  = code.slice(i+1);
  var yr     = parseInt(yrStr);
  // Bloomberg always uses 2-digit year: 26=2026, 16=2016, 06=2006
  var fy = yrStr.length === 2 ? 2000 + yr : yr;
  return prefix + mo + ' ' + String(fy).slice(2);
}

/* ── Per-ticker 5yr/15yr hi/lo from history ── */
function tkRange(history, years) {
  if (!history || !history.length) return {hi:null, lo:null};
  var cutMs = years * 365 * 86400000;
  var now   = new Date();
  var vals  = history.filter(function(r) {
    return r.open_int && (now - new Date(r.date)) <= cutMs;
  }).map(function(r){ return r.open_int; });
  if (!vals.length) return {hi:null, lo:null};
  return {hi: Math.max.apply(null,vals), lo: Math.min.apply(null,vals)};
}

/* ── Contract sequence labels (MAY 1, MAY 2, JUL 1 etc) ── */
function buildContractLabels(cd) {
  // Walk ordered tickers, count occurrences of each month abbrev
  // contract_lbl is like "MAY 26" — strip year to get month
  var counts = {};
  var labels = {};
  (cd.ordered || []).forEach(function(tk) {
    var td  = cd.tickers[tk];
    var lbl = td ? td.contract_lbl : '';          // e.g. "MAY 26"
    var mo  = lbl ? lbl.split(' ')[0] : '';       // e.g. "MAY"
    if (!mo) { labels[tk] = tk.replace(' Comdty',''); return; }
    counts[mo] = (counts[mo] || 0) + 1;
    labels[tk] = mo + ' ' + counts[mo];           // e.g. "MAY 1", "MAY 2"
  });
  return labels;
}

/* ── Global tooltip ── */
var _tip = null;
function initTooltip() {
  _tip = document.getElementById('oiTip');
  document.addEventListener('mousemove', function(e) {
    if (_tip && _tip.style.display === 'block') {
      _tip.style.left = (e.clientX + 14) + 'px';
      _tip.style.top  = (e.clientY - 10) + 'px';
    }
  });
}
function showTip(e, txt) {
  if (!_tip) return;
  _tip.textContent = txt;
  _tip.style.display = 'block';
  _tip.style.left = (e.clientX + 14) + 'px';
  _tip.style.top  = (e.clientY - 10) + 'px';
}
function hideTip() {
  if (_tip) _tip.style.display = 'none';
}   // full history cache from /api/history/<comm>

const light = () => !document.body.classList.contains('dark');

/* ── Formatters ── */
const f0 = n => (n == null || n === '' || isNaN(+n)) ? '—'
              : (+n).toLocaleString('en-US', {maximumFractionDigits: 0});
const fp = n => (n == null || n === '' || isNaN(+n)) ? '—'
              : (+n < 100 ? (+n).toFixed(2) : Math.round(+n).toLocaleString());
const fc = n => (n == null || n === '' || isNaN(+n)) ? '—'
              : ((+n >= 0 ? '+' : '') + Math.round(+n).toLocaleString());
const tk = v => v >= 1e6 ? (v/1e6).toFixed(1)+'M' : v >= 1e3 ? (v/1e3).toFixed(0)+'k' : String(v);

function cc() {
  const l = light();
  return {
    grid:  l ? '#c8d4e0' : '#1e2a3a',
    tick:  l ? '#2d3748' : '#94a3b8',
    tip: { bg: l?'#fff':'#0f1520', title: l?'#1a202c':'#f8fafc',
           body: l?'#2d3748':'#94a3b8', border: l?'#c8d4e0':'#2a3548' },
  };
}

/* ── Theme ── */
function toggleTheme() {
  document.body.classList.toggle('dark');
  document.getElementById('themeBtn').textContent = light() ? 'DARK' : 'LIGHT';
  Object.values(CH).forEach(c => { if (c) c.destroy(); });
  Object.keys(CH).forEach(k => delete CH[k]);
  buildMonitor();
  if (document.getElementById('tab-seasonal').classList.contains('act')) buildSeasonal();
}

/* ── Tabs ── */
function switchTab(id, el) {
  document.querySelectorAll('.vlm-sectab').forEach(t => t.classList.remove('act'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('act'));
  el.classList.add('act');
  document.getElementById('tab-' + id).classList.add('act');
  if (id === 'seasonal') { populateSeasContract(); buildSeasonal(); }
  if (id === 'options')  { buildOptions(); }
  if (id === 'table')    { populateTblContract();  buildTbl(); }
}

/* ── Sparkline with range band and hover tooltip ── */
function makeSpark(sparkData, val, lo5, hi5, lo15, hi15, color) {
  if (!sparkData || sparkData.length < 2) {
    // Fallback simple bar if no history yet
    var tot = hi15 || hi5 || val * 1.5 || 1;
    var pct = Math.min(96, Math.max(2, Math.round(val / tot * 100)));
    var sTip = 'Now: ' + f0(val) + ' | 5yr Hi: ' + f0(hi5) + ' | 5yr Lo: ' + f0(lo5) + ' | 15yr Hi: ' + f0(hi15) + ' | 15yr Lo: ' + f0(lo15);
    return '<div class="spark-wrap" data-tip="' + sTip + '" onmouseenter="showTip(event,this.dataset.tip)" onmouseleave="hideTip()">'
      + '<svg width="100%" height="28" viewBox="0 0 140 28" preserveAspectRatio="none">'
      + '<rect x="0" y="8" width="140" height="12" fill="rgba(74,96,128,0.15)"/>'
      + '<rect x="' + Math.round(140*(lo5||0)/(hi15||tot)) + '" y="9" width="'
      + Math.round(140*((hi5||tot)-(lo5||0))/(hi15||tot)) + '" height="10" fill="rgba(74,96,128,0.28)"/>'
      + '<rect x="' + Math.round(pct*1.4-1) + '" y="6" width="3" height="16" rx="1" fill="' + color + '"/>'
      + '</svg>'
      + '</div>';
  }

  var W = 138, H = 26, PAD = 2;
  var mn  = Math.min.apply(null, sparkData);
  var mx  = Math.max.apply(null, sparkData);
  var rng = mx - mn || 1;
  var n   = sparkData.length;

  function sx(i) { return PAD + Math.round(i / (n-1) * (W - PAD*2)); }
  function sy(v) { return PAD + Math.round((1 - (v-mn)/rng) * (H - PAD*2)); }

  // Line path
  var pts = sparkData.map(function(v, i) { return sx(i) + ',' + sy(v); }).join(' ');

  // Hi/Lo range lines
  var hiY  = sy(Math.min(hi15||mx, mx));
  var loY  = sy(Math.max(lo15||mn, mn));
  var hi5Y = sy(Math.min(hi5||mx, mx));
  var lo5Y = sy(Math.max(lo5||mn, mn));

  // Last dot
  var lx = sx(n-1), ly = sy(sparkData[n-1]);

  // Pct vs hi5
  var vsHi5 = hi5 ? ((val/hi5-1)*100).toFixed(1) : '—';
  var tipTxt = 'Now: ' + f0(val) + ' | 5yr Hi: ' + f0(hi5) + ' | 5yr Lo: ' + f0(lo5) + ' | vs 5yr Hi: ' + vsHi5 + '% | 15yr Hi: ' + f0(hi15) + ' | 15yr Lo: ' + f0(lo15);

  return '<div class="spark-wrap" data-tip="' + tipTxt + '" onmouseenter="showTip(event,this.dataset.tip)" onmouseleave="hideTip()">'
    + '<svg width="100%" height="28" viewBox="0 0 ' + (W+PAD*2) + ' ' + (H+PAD*2) + '" preserveAspectRatio="none">'
    + '<rect x="' + PAD + '" y="' + hiY + '" width="' + (W-PAD*2) + '" height="' + Math.max(1,loY-hiY) + '" fill="rgba(74,96,128,0.20)"/>'
    + '<rect x="' + PAD + '" y="' + hi5Y + '" width="' + (W-PAD*2) + '" height="' + Math.max(1,lo5Y-hi5Y) + '" fill="rgba(74,96,128,0.38)"/>'
    + '<polyline points="' + pts + '" fill="none" stroke="' + color + '" stroke-width="1.8" stroke-linejoin="round"/>'
    + '<circle cx="' + lx + '" cy="' + ly + '" r="2.5" fill="' + color + '"/>'
    // B4: faint current-value label at the line's right endpoint, so the shape is anchored to a number.
    // right-align near the endpoint; nudge y toward mid so it doesn't clip top/bottom.
    + '<text x="' + (W+PAD*2-1) + '" y="' + Math.min(H, Math.max(7, ly)) + '" text-anchor="end" '
    + 'font-size="7" font-weight="700" fill="' + color + '" opacity="0.72" '
    + 'style="font-variant-numeric:tabular-nums;">' + (val>=1e6?(val/1e6).toFixed(2)+'M':Math.round(val/1e3)+'k') + '</text>'
    + '</svg>'
    + '</div>';
}

/* ═══════════════════════════════════════════════════════════
   MONITOR
═══════════════════════════════════════════════════════════ */
function makeSignalBadge(cd) {
  if (!cd || !cd.lo5 || !cd.hi5 || cd.hi5 <= cd.lo5) return '<div class="c">—</div>';
  var pct = Math.round((cd.agg_oi - cd.lo5) / (cd.hi5 - cd.lo5) * 100);
  pct = Math.max(0, Math.min(100, pct));

  // OI trend: latest vs 10-day avg from 5-15 days ago in sparkline
  var spark = cd.sparkline || [];
  var trendUp = null;
  if (spark.length >= 15) {
    var cur      = spark[spark.length - 1];
    var baseline = 0;
    for (var _i = spark.length - 15; _i < spark.length - 5; _i++) baseline += spark[_i];
    baseline /= 10;
    trendUp = cur > baseline;
  }

  var col = pct >= 80 ? '#ef4444' : pct >= 60 ? '#f97316' : pct >= 30 ? '#fbbf24' : '#22c55e';
  var bg  = pct >= 80 ? 'rgba(239,68,68,.12)' : pct >= 60 ? 'rgba(249,115,22,.10)' : pct >= 30 ? 'rgba(251,191,36,.10)' : 'rgba(34,197,94,.10)';
  var aCl = trendUp === null ? '' : trendUp ? '#22c55e' : '#ef4444';
  var arr = trendUp === null ? '' : trendUp ? '▲' : '▼';
  var tip = pct + 'th %ile of 5yr range' + (trendUp !== null ? (trendUp ? ' · EXPANDING' : ' · CONTRACTING') : '');

  return '<div class="c" style="padding:3px 6px;">'
    + '<span class="oi-badge" style="color:' + col + ';background:' + bg + ';border-color:' + col + '55;" title="' + tip + '">'
    + pct + '<span style="font-size:9px;">%</span>'
    + (arr ? '<span style="color:' + aCl + ';font-size:12px;line-height:1;">' + arr + '</span>' : '')
    + '</span></div>';
}

/* B1: per-tenor share of the commodity aggregate OI, shown on child rows in place of
   the (previously repeated) aggregate figure. Tiny inline bar + percent. */
function shareCell(tenorOi, aggOi, color) {
  var oi = +tenorOi, agg = +aggOi;
  if (!oi || !agg || agg <= 0) return '<div class="c vlm-muted">—</div>';
  var pct = oi / agg * 100;
  var pctTxt = pct >= 9.95 ? pct.toFixed(0) : pct.toFixed(1);
  var w = Math.max(2, Math.min(100, pct));
  return '<div class="c" style="flex-direction:column;align-items:flex-end;gap:1px;padding:2px 6px;" '
    + 'title="' + f0(oi) + ' of ' + f0(agg) + ' aggregate">'
    + '<span style="font-size:12px;font-weight:700;font-variant-numeric:tabular-nums;color:var(--dim);">' + pctTxt + '%</span>'
    + '<span style="display:block;width:38px;height:3px;background:var(--bord);border-radius:2px;overflow:hidden;">'
    + '<span style="display:block;height:100%;width:' + w + '%;background:' + color + ';"></span></span>'
    + '</div>';
}

/* B2: one range bar replacing the 5yr Hi/Lo + 15yr Hi/Lo columns.
   15yr range = faint full track; 5yr range = brighter sub-band; current OI = gold marker.
   Exact hi/lo numbers live in the hover tooltip (delegated via .oi-range dataset). */
function rangeBar(cur, lo5, hi5, lo15, hi15, color) {
  var c = +cur, a5 = +lo5, b5 = +hi5, a15 = +lo15, b15 = +hi15;
  // Fall back to the widest available window if 15yr missing
  var lo = isFinite(a15) ? a15 : a5, hi = isFinite(b15) ? b15 : b5;
  if (!isFinite(lo) || !isFinite(hi) || hi <= lo) return '<div class="c vlm-muted">—</div>';
  var span = hi - lo;
  var pct = function(v){ return Math.max(0, Math.min(100, (v - lo) / span * 100)); };
  var s5 = isFinite(a5) ? pct(a5) : 0, e5 = isFinite(b5) ? pct(b5) : 100;
  var cm = pct(isFinite(c) ? c : lo);
  var tip = 'Now: ' + f0(c) + ' | 5yr Hi: ' + f0(hi5) + ' | 5yr Lo: ' + f0(lo5)
          + ' | 15yr Hi: ' + f0(hi15) + ' | 15yr Lo: ' + f0(lo15);
  return '<div class="c oi-range" style="padding:2px 8px;" data-tip="' + tip + '" '
    + 'onmouseenter="showTip(event,this.dataset.tip)" onmouseleave="hideTip()">'
    + '<div style="position:relative;width:100%;height:14px;">'
    + '<div style="position:absolute;top:5px;left:0;width:100%;height:4px;background:var(--bord);border-radius:2px;"></div>'
    + '<div style="position:absolute;top:4px;left:' + s5 + '%;width:' + Math.max(1,(e5-s5)) + '%;height:6px;background:' + color + '55;border-radius:2px;"></div>'
    + '<div style="position:absolute;top:1px;left:' + cm + '%;width:3px;height:12px;background:' + color + ';border-radius:1px;transform:translateX(-1.5px);"></div>'
    + '</div></div>';
}

/* B3: OI-vs-price conviction read. Combines the day's OI change with the day's settle
   change into the classic futures interpretation. Returns a small colored tag + a
   descriptive tooltip; empty string when either input is missing/zero (no signal). */
function convictionTag(oiChg, settleChg) {
  var oc = +oiChg, sc = +settleChg;
  if (!isFinite(oc) || !isFinite(sc) || oc === 0 || sc === 0) return '';
  var oiUp = oc > 0, pxUp = sc > 0, tag, full, color;
  if (oiUp && pxUp)        { tag = 'NL'; full = 'New longs (OI up, price up)';        color = 'var(--grn)'; }
  else if (oiUp && !pxUp)  { tag = 'NS'; full = 'New shorts (OI up, price down)';      color = 'var(--red)'; }
  else if (!oiUp && pxUp)  { tag = 'SC'; full = 'Short covering (OI down, price up)';  color = 'var(--grn)'; }
  else                     { tag = 'LL'; full = 'Long liquidation (OI down, price down)'; color = 'var(--red)'; }
  return '<span class="oi-conv" style="color:' + color + ';border:1px solid ' + color
    + '55;border-radius:3px;padding:0 3px;margin-left:4px;font-size:9px;font-weight:800;'
    + 'letter-spacing:.5px;vertical-align:middle;cursor:default;" '
    + 'data-tip="' + full + '" onmouseenter="showTip(event,this.dataset.tip)" onmouseleave="hideTip()">'
    + tag + '</span>';
}

function buildMonitor() {
  const body = document.getElementById('monBody');
  body.innerHTML = '';

  const hdr = document.createElement('div');
  hdr.className = 'grid-head G';
  hdr.innerHTML =
    '<div class="gh">Ticker</div>'
    + '<div class="gh">Fut Cont</div>'
    + '<div class="gh">Open Int</div>'
    + '<div class="gh">OI Chg</div>'
    + '<div class="gh">Settle Px</div>'
    + '<div class="gh">Aggte O.I.</div>'
    + '<div class="gh">Aggte OI Chg</div>'
    + '<div class="gh" style="text-align:left;padding-left:4px;">Aggte OI (1yr)</div>'
    + '<div class="gh" title="Current OI position within 5yr range (bar) with 15yr range behind">OI Range 5/15yr</div>'
    + '<div class="gh">1st Notice</div>'
    + '<div class="gh" title="5yr percentile + OI trend direction">5yr %ile</div>';
  body.appendChild(hdr);

  /* Tooltip delegation for .oi-cell elements */
  body.addEventListener('mouseover', function(e) {
    var el = e.target.closest('.oi-cell');
    if (!el) return;
    var comm2 = el.dataset.comm;
    var tkKey2 = el.dataset.tk;
    var oi = el.dataset.oi;
    if (!comm2 || !tkKey2) return;
    var cd2 = DATA.commodities[comm2];
    var tkLbl2 = tkKey2.replace(' Comdty','');
    // Use full history from HIST cache if available, else fall back to inline
    // Use stored per-ticker ranges if available (instant, no history needed)
    var td2 = cd2 && cd2.tickers[tkKey2] ? cd2.tickers[tkKey2] : {};
    var lo5t  = td2.tk_lo5  || null;
    var hi5t  = td2.tk_hi5  || null;
    var lo15t = td2.tk_lo15 || null;
    var hi15t = td2.tk_hi15 || null;
    // Fall back to full history if stored ranges not available
    if (!hi5t) {
      var fullH = (HIST[comm2] && HIST[comm2].ticker_history && HIST[comm2].ticker_history[tkLbl2])
                  ? HIST[comm2].ticker_history[tkLbl2] : [];
      var r5  = tkRange(fullH, 5);
      var r15 = tkRange(fullH, 15);
      lo5t = r5.lo; hi5t = r5.hi; lo15t = r15.lo; hi15t = r15.hi;
    }
    var tip = 'Now: ' + f0(+oi)
            + ' | 5yr Hi: '  + f0(hi5t)  + ' | 5yr Lo: '  + f0(lo5t)
            + ' | 15yr Hi: ' + f0(hi15t) + ' | 15yr Lo: ' + f0(lo15t);
    showTip(e, tip);
  });
  body.addEventListener('mouseout', function(e) {
    if (e.target.closest('.oi-cell')) hideTip();
  });

  COMMS.forEach(function(comm) {
    var cd  = DATA.commodities[comm]; if (!cd) return;
    var cfg = CFG[comm];
    var isExp    = expanded === comm;
    var ordered  = cd.ordered || Object.keys(cd.tickers);
    var frontTk  = ordered[0];
    var front    = cd.tickers[frontTk] || {};
    var frontGen = frontTk.replace(' Comdty', '');
    var frontMth = front.contract_lbl || '—';

    var grp = document.createElement('div');
    grp.style.borderBottom = '1px solid var(--bord)';

    /* Aggregate row */
    var ar = document.createElement('div');
    ar.className = 'agg-row G';
    ar.innerHTML =
      '<div class="cl"><span class="ticker-lbl" style="color:' + cfg.color + '">'
        + '<span class="arr ' + (isExp ? 'open' : '') + '">&#9654;</span>'
        + frontGen + '<span class="ticker-sub">' + frontMth + '</span>'
      + '</span></div>'
      + '<div class="c" style="color:var(--dim);font-size:11px;">' + frontMth + '</div>'
      + '<div class="c oi-cell" data-comm="' + comm + '" data-tk="' + frontTk + '" data-oi="' + (front.open_int||'') + '">' + f0(front.open_int) + '</div>'
      + '<div class="c ' + ((front.oi_chg||0)>=0?'vlm-pos':'vlm-neg') + '">' + fc(front.oi_chg) + convictionTag(front.oi_chg, front.settle_chg) + '</div>'
      + '<div class="c" style="color:' + ((front.oi_chg||0)>=0?'var(--grn)':'var(--red)') + ';">' + fp(front.settle) + '</div>'
      + '<div class="c" style="color:' + cfg.color + ';font-weight:700;">' + f0(cd.agg_oi) + '</div>'
      + '<div class="c ' + ((cd.agg_chg||0)>=0?'vlm-pos':'vlm-neg') + '">' + fc(cd.agg_chg) + '</div>'
      + '<div class="cl" style="padding:2px 5px;">' + makeSpark(cd.sparkline, cd.agg_oi, cd.lo5, cd.hi5, cd.lo15, cd.hi15, cfg.color) + '</div>'
      + rangeBar(cd.agg_oi, cd.lo5, cd.hi5, cd.lo15, cd.hi15, cfg.color)
      + '<div class="c fn">' + (front.first_notice || '—') + '</div>'
      + makeSignalBadge(cd);
    ar.addEventListener('click', function() {
      expanded = isExp ? null : comm; selKey = null;
      if (expanded) ensureHist(expanded).then(function(){ buildMonitor(); });
      else buildMonitor();
    });
    grp.appendChild(ar);

    if (isExp) {
      var seqLabels = buildContractLabels(cd);
      ordered.slice(1).forEach(function(tkKey) {
        var td  = cd.tickers[tkKey]; if (!td) return;
        var key = comm + '-' + tkKey;
        var isSel = selKey === key;
        var lbl = tkKey.replace(' Comdty', '');
        var mth = td.contract_lbl || '—';

        var cr = document.createElement('div');
        cr.className = 'ct-row G' + (isSel ? ' sel' : '');
        cr.innerHTML =
          '<div class="cl"><span class="ctlbl">' + (seqLabels[tkKey]||lbl) + '</span></div>'
          + '<div class="c" style="color:var(--dim);font-size:11px;">' + mth + '</div>'
          + '<div class="c oi-cell" data-comm="' + comm + '" data-tk="' + tkKey + '" data-oi="' + (td.open_int||'') + '">' + f0(td.open_int) + '</div>'
          + '<div class="c ' + ((td.oi_chg||0)>=0?'vlm-pos':'vlm-neg') + '">' + fc(td.oi_chg) + convictionTag(td.oi_chg, td.settle_chg) + '</div>'
          + '<div class="c" style="color:' + ((td.oi_chg||0)>=0?'var(--grn)':'var(--red)') + ';">' + fp(td.settle) + '</div>'
          + shareCell(td.open_int, cd.agg_oi, cfg.color)
          + '<div class="c vlm-muted">—</div>'
          + '<div class="cl" style="padding:2px 5px;">' + makeSpark(null, td.open_int, td.tk_lo5||cd.lo5, td.tk_hi5||cd.hi5, td.tk_lo15||cd.lo15, td.tk_hi15||cd.hi15, cfg.color) + '</div>'
          + rangeBar(td.open_int, td.tk_lo5||cd.lo5, td.tk_hi5||cd.hi5, td.tk_lo15||cd.lo15, td.tk_hi15||cd.hi15, cfg.color)
          + '<div class="c fn">' + (td.first_notice || '—') + '</div>'
          + '<div class="c"></div>';

        cr.addEventListener('click', function(e) {
          e.stopPropagation();
          selKey = isSel ? null : key;
          buildMonitor();
          if (selKey) setTimeout(function() { drawChart(comm, tkKey); }, 40);
        });
        grp.appendChild(cr);
      });

      if (selKey && selKey.startsWith(comm + '-')) {
        var tkKey2 = selKey.slice(comm.length + 1);
        var td2    = cd.tickers[tkKey2] || {};
        var lbl2   = tkKey2.replace(' Comdty', '');
        var cp = document.createElement('div');
        cp.className = 'chart-panel open';
        cp.id = 'cp-' + comm;
        cp.innerHTML =
          '<div class="chart-hdr">'
          + '<span class="chart-ttl">' + lbl2 + ' ' + (td2.contract_lbl||'') + ' — Open Interest</span>'
          + '<div class="vlm-ctrl-btns" style="padding:0;">'
          + '<button class="vlm-btn ' + (cMode==='seasonal'?'act':'')   + '" onclick="setChMode(&quot;seasonal&quot;,this)">SEASONAL</button>'
          + '<button class="vlm-btn ' + (cMode==='historical'?'act':'') + '" onclick="setChMode(&quot;historical&quot;,this)">HISTORICAL</button>'
          + '<button class="vlm-btn ' + (cMode==='daily'?'act':'')      + '" onclick="setChMode(&quot;daily&quot;,this)">DAILY CHG</button>'
          + '</div>'
          + '<div class="d-row" id="cpDates" style="display:' + (cMode==='historical'?'flex':'none') + ';">'
          + '<span style="font-size:10px;letter-spacing:1px;color:var(--muted);">FROM</span>'
          + '<input class="d-inp" type="date" id="cpFrom" value="2020-01-01">'
          + '<span style="font-size:10px;letter-spacing:1px;color:var(--muted);">TO</span>'
          + '<input class="d-inp" type="date" id="cpTo" value="' + DATA.last_date + '">'
          + '<button class="vlm-btn" onclick="redrawChart()">GO</button>'
          + '</div></div>'
          + '<div class="leg-row" id="cpLeg"></div>'
          + '<div class="cw"><canvas id="cpCanvas" role="img" aria-label="OI chart">OI history.</canvas></div>';
        grp.appendChild(cp);
      }
    }
    body.appendChild(grp);
  });

  if (selKey) {
    var dash = selKey.indexOf('-');
    drawChart(selKey.slice(0, dash), selKey.slice(dash + 1));
  }
}

/* ═══════════════════════════════════════════════════════════
   CONTRACT CHART
═══════════════════════════════════════════════════════════ */
function setChMode(m, btn) {
  cMode = m;
  var comm = selKey ? selKey.split('-')[0] : null;
  if (comm) {
    var cp = document.getElementById('cp-' + comm);
    if (cp) cp.querySelectorAll('.vlm-btn').forEach(function(b) { b.classList.remove('act'); });
  }
  btn.classList.add('act');
  var dr = document.getElementById('cpDates');
  if (dr) dr.style.display = m === 'historical' ? 'flex' : 'none';
  redrawChart();
}
function redrawChart() {
  if (!selKey) return;
  var dash = selKey.indexOf('-');
  drawChart(selKey.slice(0, dash), selKey.slice(dash + 1));
}

function drawChart(comm, tkKey) {
  var canvas = document.getElementById('cpCanvas'); if (!canvas) return;
  if (CH.contract) { CH.contract.destroy(); CH.contract = null; }
  var cd  = DATA.commodities[comm];
  var td  = cd.tickers[tkKey]; if (!td) return;
  var cfg = CFG[comm];
  var leg = document.getElementById('cpLeg');
  var C   = cc();
  var bo  = {
    responsive: true, maintainAspectRatio: false,
    interaction: {mode: 'index', intersect: false},
    plugins: {
      legend: {display: false},
      tooltip: {mode:'index', intersect:false, backgroundColor:C.tip.bg,
                titleColor:C.tip.title, bodyColor:C.tip.body,
                borderColor:C.tip.border, borderWidth:1}
    },
    scales: {
      x: {grid:{color:C.grid}, ticks:{color:C.tick, font:{size:9}, maxTicksLimit:12}},
      y: {grid:{color:C.grid}, ticks:{color:C.tick, font:{size:10}, callback:tk}}
    }
  };

  var hist     = td.history || [];
  var curYear  = new Date(DATA.last_date).getFullYear();
  var labels   = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

  if (cMode === 'seasonal') {
    var byMo  = Array.from({length:12}, function() { return []; });
    var cutDt = new Date(DATA.last_date);
    cutDt.setFullYear(cutDt.getFullYear() - seasYr);
    hist.forEach(function(r) {
      var dt = new Date(r.date);
      if (dt >= cutDt && dt.getFullYear() < curYear && r.open_int)
        byMo[dt.getMonth()].push(r.open_int);
    });
    var avg = byMo.map(function(a) { return a.length ? Math.round(a.reduce(function(s,v){return s+v;},0)/a.length) : null; });
    var hi  = byMo.map(function(a) { return a.length ? Math.max.apply(null,a) : null; });
    var lo  = byMo.map(function(a) { return a.length ? Math.min.apply(null,a) : null; });
    var cur = Array(12).fill(null);
    hist.filter(function(r){ return new Date(r.date).getFullYear()===curYear&&r.open_int; })
        .forEach(function(r){ var m=new Date(r.date).getMonth(); if(cur[m]===null||r.open_int>cur[m])cur[m]=r.open_int; });
    if (leg) leg.innerHTML =
      '<div class="leg"><div class="lsw-b" style="background:' + cfg.ca + ';border:1px dashed ' + cfg.color + '88;height:8px;border-radius:2px;"></div>' + seasYr + 'yr Range</div>'
      + '<div class="leg"><div class="lsw" style="background:' + C.tick + ';height:2px;"></div>' + seasYr + 'yr Avg</div>'
      + '<div class="leg"><div class="lsw" style="background:' + cfg.color + ';height:2px;"></div>' + curYear + '</div>';
    CH.contract = new Chart(canvas, {type:'line', data:{labels:labels, datasets:[
      {data:hi, fill:'+1', backgroundColor:cfg.ca, borderColor:cfg.color+'55', borderWidth:1, borderDash:[2,3], pointRadius:0, label:'Hi'},
      {data:lo, fill:false, borderColor:cfg.color+'33', borderWidth:1, borderDash:[2,3], pointRadius:0, label:'Lo'},
      {data:avg, borderColor:C.tick, borderWidth:1.5, borderDash:[5,4], pointRadius:0, fill:false, label:seasYr+'yr Avg'},
      {data:cur, borderColor:cfg.color, borderWidth:2.5, pointRadius:4, pointBackgroundColor:cfg.color, fill:false, spanGaps:false, label:String(curYear)},
    ]}, options:bo});

  } else if (cMode === 'historical') {
    var from = document.getElementById('cpFrom') ? document.getElementById('cpFrom').value : '2020-01-01';
    var to   = document.getElementById('cpTo')   ? document.getElementById('cpTo').value   : DATA.last_date;
    var f = hist.filter(function(r){ return r.date>=from&&r.date<=to&&r.open_int!=null; });
    var stride = Math.max(1, Math.floor(f.length/150));
    f = f.filter(function(_,i){ return i%stride===0; });
    if (leg) leg.innerHTML = '<div class="leg"><div class="lsw" style="background:' + cfg.color + ';height:2px;"></div>' + tkKey.replace(' Comdty','') + ' Daily OI</div>';
    CH.contract = new Chart(canvas, {type:'line', data:{
      labels: f.map(function(r){ return r.date.slice(5); }),
      datasets:[{data:f.map(function(r){return r.open_int;}), borderColor:cfg.color,
                 borderWidth:1.5, pointRadius:0, fill:true, backgroundColor:cfg.ca, label:'OI'}]
    }, options:bo});

  } else {
    var f2 = hist.filter(function(r){ return r.oi_chg!=null; }).slice(-120);
    var stride2 = Math.max(1, Math.floor(f2.length/60));
    f2 = f2.filter(function(_,i){ return i%stride2===0; });
    if (leg) leg.innerHTML =
      '<div class="leg"><div class="lsw" style="background:var(--grn);height:2px;"></div>+OI</div>'
      + '<div class="leg"><div class="lsw" style="background:var(--red);height:2px;"></div>-OI</div>';
    CH.contract = new Chart(canvas, {type:'bar', data:{
      labels: f2.map(function(r){ return r.date.slice(5); }),
      datasets:[{data:f2.map(function(r){return r.oi_chg;}),
                 backgroundColor:f2.map(function(r){ return (r.oi_chg||0)>=0?'rgba(34,197,94,0.7)':'rgba(239,68,68,0.7)'; }),
                 borderWidth:0, label:'OI Chg'}]
    }, options:bo});
  }
}

/* ═══════════════════════════════════════════════════════════
   SEASONAL
═══════════════════════════════════════════════════════════ */
function onSeasYrSlide(el) {
  seasYr = parseInt(el.value);
  document.getElementById('seasYrLbl').textContent = seasYr + ' YR';
  buildSeasonal();
}
function setSeasMode(m, btn) {
  seasMode = m;
  document.querySelectorAll('#seasModeBtns .vlm-btn').forEach(function(b){ b.classList.remove('act'); });
  btn.classList.add('act');
  // SPAGHETTI|GRID toggle only applies to individual-years mode; hide it for the band view
  var lb = document.getElementById('seasLayoutBtns');
  if (lb) lb.style.display = m === 'individual' ? '' : 'none';
  buildSeasonal();
}
function setSeasLayout(v, btn) {
  seasLayout = v;
  document.querySelectorAll('#seasLayoutBtns .vlm-btn').forEach(function(b){ b.classList.remove('act'); });
  btn.classList.add('act');
  buildSeasonal();
}
function onSeasCommChange() { populateSeasContract(); buildSeasonal(); }

function populateSeasContract() {
  var refComm = document.getElementById('seasSingleComm').value;
  var cd = DATA.commodities[refComm]; if (!cd) return;
  var sel  = document.getElementById('seasContract');
  var prev = sel.value;
  sel.innerHTML = '<option value="Aggregate">AGGREGATE</option>';
  var tblSeqLabels = buildContractLabels(cd);
  // In stacked mode, show contract slot labels (MAY 1, JUL 1 etc)
  // which map to each commodity's own front May/Jul contract
  (cd.ordered || []).forEach(function(tk) {
    var lbl = tk.replace(' Comdty', '');
    // Extract slot label e.g. "MAY 1" from CTMAY1 -> "MAY 1"
    var slotLbl = tblSeqLabels[tk] || lbl;
    var opt = document.createElement('option');
    opt.value = slotLbl;   // store slot label as value e.g. "MAY 1"
    opt.textContent = slotLbl;
    sel.appendChild(opt);
  });
  if ([].slice.call(sel.options).some(function(o){ return o.value===prev; })) sel.value = prev;
  var maxY   = cd.max_years || 18;
  var slider = document.getElementById('seasYrSlider');
  if (slider) {
    slider.max = maxY;
    if (seasYr > maxY) { seasYr = maxY; slider.value = maxY; document.getElementById('seasYrLbl').textContent = maxY + ' YR'; }
  }
}

async function ensureHist(comm) {
  if (HIST[comm]) return;
  try {
    var r = await fetch('/api/history/' + comm);
    HIST[comm] = await r.json();
  } catch(e) { console.warn('history fetch failed', comm, e); }
}

function getSeasHist(comm) {
  var contractSel = document.getElementById('seasContract').value || 'Aggregate';
  var src = HIST[comm] || null;
  var agg = src ? src.daily_agg : DATA.commodities[comm].daily_agg;
  var th  = src ? src.ticker_history : DATA.commodities[comm].ticker_history;
  if (contractSel === 'Aggregate')
    return Object.entries(agg).map(function(e){ return {date:e[0], open_int:e[1]}; });
  // contractSel is a slot label like "MAY 1" — map to this commodity's ticker
  // e.g. CT+"MAY 1" -> "CTMAY1", SB+"MAY 1" -> "SBMAY1"
  var slotMap = {'MAR 1':'MAR1','MAY 1':'MAY1','JUL 1':'JUL1','OCT 1':'OCT1','DEC 1':'DEC1','SEP 1':'SEP1',
                 'MAR 2':'MAR2','MAY 2':'MAY2','JUL 2':'JUL2','OCT 2':'OCT2','DEC 2':'DEC2','SEP 2':'SEP2'};
  var slot = slotMap[contractSel];
  var tkKey = slot ? comm + slot : contractSel;
  return (th || {})[tkKey] || [];
}

function computeBand(hist, curYear) {
  var cutDt = new Date(DATA.last_date);
  cutDt.setFullYear(cutDt.getFullYear() - seasYr);
  var byMo = Array.from({length:12}, function(){ return []; });
  hist.forEach(function(r) {
    var dt = new Date(r.date);
    if (dt >= cutDt && dt.getFullYear() < curYear && r.open_int)
      byMo[dt.getMonth()].push(r.open_int);
  });
  var avg = byMo.map(function(a){ return a.length ? Math.round(a.reduce(function(s,v){return s+v;},0)/a.length) : null; });
  var hi  = byMo.map(function(a){ return a.length ? Math.max.apply(null,a) : null; });
  var lo  = byMo.map(function(a){ return a.length ? Math.min.apply(null,a) : null; });
  var cur = Array(12).fill(null);
  hist.filter(function(r){ return new Date(r.date).getFullYear()===curYear&&r.open_int; })
      .forEach(function(r){ var m=new Date(r.date).getMonth(); if(cur[m]===null||r.open_int>cur[m])cur[m]=r.open_int; });
  return {avg:avg, hi:hi, lo:lo, cur:cur};
}

var YR_COLORS = ['#E8C547','#5ba3e8','#2ecc8a','#e85555','#c084fc','#E07B54','#22d3ee',
                 '#f97316','#a3e635','#fb7185','#818cf8','#34d399','#fbbf24','#60a5fa',
                 '#f472b6','#4ade80','#c084fc','#38bdf8'];

function computeIndividual(hist, curYear) {
  var cutDt = new Date(DATA.last_date);
  cutDt.setFullYear(cutDt.getFullYear() - seasYr);
  var byYear = {};
  hist.forEach(function(r) {
    var dt = new Date(r.date);
    if (dt < cutDt || !r.open_int) return;
    var yr = dt.getFullYear(), mo = dt.getMonth();
    if (!byYear[yr]) byYear[yr] = Array(12).fill(null);
    if (byYear[yr][mo] === null || r.open_int > byYear[yr][mo]) byYear[yr][mo] = r.open_int;
  });
  var years = Object.keys(byYear).map(Number).sort(function(a,b){return b-a;});
  return {byYear:byYear, years:years};
}

function seasChartOpts() {
  var C = cc();
  return {
    responsive: true, maintainAspectRatio: false,
    interaction: {mode:'index', intersect:false},
    plugins: {
      legend: {display:false},
      tooltip: {mode:'index', intersect:false, backgroundColor:C.tip.bg,
                titleColor:C.tip.title, bodyColor:C.tip.body,
                borderColor:C.tip.border, borderWidth:1,
                callbacks: {label: function(ctx) {
                  var v = ctx.parsed.y; if (v == null) return null;
                  var fmt = v>=1e6?(v/1e6).toFixed(2)+'M':v>=1e3?(v/1e3).toFixed(0)+'k':String(Math.round(v));
                  return '  ' + ctx.dataset.label + ': ' + fmt;
                }}}
    },
    scales: {
      x: {grid:{color:C.grid}, ticks:{color:C.tick, font:{size:9}}},
      y: {grid:{color:C.grid}, ticks:{color:C.tick, font:{size:9}, callback:tk}}
    }
  };
}

function buildSeasCard(comm) {
  var cfg     = CFG[comm];
  var hist    = getSeasHist(comm);
  var curYear = new Date(DATA.last_date).getFullYear();
  var labels  = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  var C       = cc();
  var ds, legH;

  if (seasMode === 'band') {
    var b = computeBand(hist, curYear);
    ds = [
      {data:b.hi, fill:'+1', backgroundColor:cfg.ca, borderColor:cfg.color+'55',
       borderWidth:1, borderDash:[2,3], pointRadius:0, label:'Hi'},
      {data:b.lo, fill:false, borderColor:cfg.color+'33',
       borderWidth:1, borderDash:[2,3], pointRadius:0, label:'Lo'},
      {data:b.avg, borderColor:C.tick, borderWidth:1.5, borderDash:[5,4],
       pointRadius:0, fill:false, label:seasYr+'yr Avg'},
      {data:b.cur, borderColor:cfg.color, borderWidth:2.5, pointRadius:4,
       pointBackgroundColor:cfg.color, fill:false, spanGaps:false, label:String(curYear)},
    ];
    legH =
      '<div class="leg"><div class="lsw-b" style="background:'+cfg.ca+';border:1px dashed '+cfg.color+'88;height:8px;border-radius:2px;"></div>Hi/Lo Range</div>'
      + '<div class="leg"><div class="lsw" style="background:'+C.tick+';height:2px;"></div>'+seasYr+'yr Avg</div>'
      + '<div class="leg"><div class="lsw" style="background:'+cfg.color+';height:2px;"></div>'+curYear+'</div>';

  } else {
    var ind = computeIndividual(hist, curYear);
    ds = ind.years.map(function(yr, i) {
      return {
        data: ind.byYear[yr],
        borderColor: yr===curYear ? cfg.color : YR_COLORS[i % YR_COLORS.length],
        borderWidth: yr===curYear ? 2.5 : 1.2,
        pointRadius: yr===curYear ? 3 : 0,
        fill: false, spanGaps: false,
        label: String(yr), order: yr===curYear ? 0 : 1,
      };
    });
    legH = ind.years.map(function(yr, i) {
      var c = yr===curYear ? cfg.color : YR_COLORS[i % YR_COLORS.length];
      return '<div class="leg">'
        + '<div style="width:12px;height:12px;border-radius:2px;background:' + c
        + ';flex-shrink:0;"></div>' + yr + '</div>';
    }).join('');
  }
  return {ds:ds, legH:legH, labels:labels, cfg:cfg};
}

/* nearest-month snap + crosshair shared by both individual-years layouts */
var SEAS_MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
function seasNearestMonth(ch, clientX) {
  var r = ch.canvas.getBoundingClientRect(), xIn = clientX - r.left, best = 0, bd = Infinity;
  for (var m = 0; m < 12; m++) {
    var px = ch.scales.x.getPixelForValue(m), dd = Math.abs(px - xIn);
    if (dd < bd) { bd = dd; best = m; }
  }
  return best;
}
var seasCrosshair = {id:'seasCrosshair', afterDraw:function(ch){
  var mi = ch.$hoverMonth; if (mi == null) return;
  var x = ch.scales.x.getPixelForValue(mi), a = ch.chartArea, c = ch.ctx;
  c.save(); c.strokeStyle = YR_COLORS[0]; c.globalAlpha = 0.5; c.lineWidth = 1;
  c.setLineDash([3,3]); c.beginPath(); c.moveTo(x, a.top); c.lineTo(x, a.bottom); c.stroke(); c.restore();
}};
function seasYRange(ind) {
  var lo = Infinity, hi = -Infinity;
  ind.years.forEach(function(yr){ ind.byYear[yr].forEach(function(v){
    if (v != null) { if (v < lo) lo = v; if (v > hi) hi = v; } }); });
  if (!isFinite(lo)) return {min:undefined, max:undefined};
  var p = (hi - lo) * 0.08; return {min:lo - p, max:hi + p};
}

async function buildSeasonal() {
  Object.keys(CH).filter(function(k){ return k.startsWith('s-') || k.startsWith('sg-'); })
        .forEach(function(k){ if(CH[k]) CH[k].destroy(); delete CH[k]; });
  populateSeasContract();

  var comm = document.getElementById('seasSingleComm').value;
  await ensureHist(comm);

  var grid         = document.getElementById('seasGrid');
  var single       = document.getElementById('seasSingle');
  var chartRow     = document.getElementById('seasChartRow');
  var rail         = document.getElementById('seasRail');
  var smGrid       = document.getElementById('seasSmGrid');
  var contractSel  = document.getElementById('seasContract').value || 'Aggregate';
  var contractLbl  = contractSel === 'Aggregate' ? 'Aggregate OI' : contractSel;
  var layoutLbl    = seasMode === 'band' ? 'Hi / Avg / Lo'
                     : (seasLayout === 'grid' ? 'Individual Years (grid)' : 'Individual Years (spaghetti)');

  grid.style.display = 'none'; single.style.display = '';
  var cfg = CFG[comm];
  document.getElementById('seasSingleHdr').innerHTML =
    '<span style="font-size:13px;font-weight:700;letter-spacing:1px;color:'+cfg.color+'">'+comm+' '+cfg.name+'</span>'
    + '<span style="color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:1px;margin-left:8px;">'+seasYr+'yr · '+contractLbl+' · '+layoutLbl+'</span>';

  var isGrid = (seasMode === 'individual' && seasLayout === 'grid');

  if (isGrid) {
    // GRID small-multiples: hide chart row, show panel grid
    chartRow.style.display = 'none';
    smGrid.style.display = '';
    document.getElementById('seasSingleLeg').innerHTML = '';
    buildSeasGrid(comm);
    return;
  }

  // SPAGHETTI or BAND: single chart, rail only for spaghetti
  chartRow.style.display = '';
  smGrid.style.display = 'none';
  var wantRail = (seasMode === 'individual' && seasLayout === 'spaghetti');
  rail.style.display = wantRail ? '' : 'none';
  chartRow.classList.toggle('no-rail', !wantRail);

  var card = buildSeasCard(comm);
  document.getElementById('seasSingleLeg').innerHTML = card.legH;
  setTimeout(function() {
    var cv = document.getElementById('scSingle'); if (!cv) return;
    if (CH['s-single']) { CH['s-single'].destroy(); delete CH['s-single']; }
    CH['s-single'] = new Chart(cv, {type:'line', data:{labels:card.labels, datasets:card.ds},
                                    options:seasChartOpts(), plugins: wantRail ? [seasCrosshair] : []});
    if (wantRail) attachSeasRail(CH['s-single'], comm);
  }, 60);
}

/* SPAGHETTI hover rail: sweep the chart, read every year's OI at the hovered month */
function attachSeasRail(ch, comm) {
  var ind = computeIndividual(getSeasHist(comm), new Date(DATA.last_date).getFullYear());
  var body = document.getElementById('seasRailBody'), hdr = document.getElementById('seasRailHdr');
  var curYear = new Date(DATA.last_date).getFullYear(), cfg = CFG[comm];
  body.innerHTML = ind.years.map(function(yr, i){
    var c = yr===curYear ? cfg.color : YR_COLORS[i % YR_COLORS.length];
    return '<div class="seas-rail-row"><span class="rr-yr" style="color:'+c+';">'+yr
      + '</span><span class="rr-val" id="srv-'+yr+'">—</span></div>';
  }).join('');
  ch.canvas.onmousemove = function(e){
    var mi = seasNearestMonth(ch, e.clientX);
    ch.$hoverMonth = mi; ch.draw();
    hdr.textContent = SEAS_MONTHS[mi] + ' · OI by year';
    ind.years.forEach(function(yr){
      var el = document.getElementById('srv-'+yr); if (!el) return;
      var v = ind.byYear[yr][mi];
      el.textContent = v==null ? '—' : (v>=1e6?(v/1e6).toFixed(2)+'M':Math.round(v/1e3)+'k');
    });
  };
  ch.canvas.onmouseleave = function(){
    ch.$hoverMonth = null; ch.draw();
    hdr.textContent = 'hover a month';
    ind.years.forEach(function(yr){ var el = document.getElementById('srv-'+yr); if (el) el.textContent = '—'; });
  };
}

/* GRID small-multiples: one panel per prior year, gold current-year ghost in each, synced hover */
function buildSeasGrid(comm) {
  var ind = computeIndividual(getSeasHist(comm), new Date(DATA.last_date).getFullYear());
  var curYear = new Date(DATA.last_date).getFullYear(), cfg = CFG[comm], C = cc();
  var yr = seasYRange(ind), curArr = ind.byYear[curYear] || Array(12).fill(null);
  var priors = ind.years.filter(function(y){ return y !== curYear; });
  var g = document.getElementById('seasSmGrid'); g.innerHTML = '';
  var fmt = function(v){ return v==null ? '—' : (v>=1e6?(v/1e6).toFixed(2)+'M':Math.round(v/1e3)+'k'); };

  priors.forEach(function(y){
    var cell = document.createElement('div'); cell.className = 'seas-sm-cell';
    cell.innerHTML = '<div class="seas-sm-yr" style="color:'+cfg.color+';">'+y+'</div>'
      + '<div class="seas-sm-sub">vs '+curYear+'</div><div class="seas-sm-hover" id="sgh-'+y+'"></div>'
      + '<div class="seas-sm-cw"><canvas id="sg-cv-'+y+'"></canvas></div>';
    g.appendChild(cell);
  });
  var smOpts = {
    responsive:true, maintainAspectRatio:false, animation:false,
    plugins:{legend:{display:false}, tooltip:{enabled:false}},
    scales:{x:{grid:{color:C.grid, drawTicks:false}, ticks:{color:C.tick, font:{size:7}, maxTicksLimit:6, maxRotation:0}},
            y:{min:yr.min, max:yr.max, grid:{color:C.grid, drawTicks:false},
               ticks:{color:C.tick, font:{size:7}, maxTicksLimit:3,
                      callback:function(v){ return v>=1e6?(v/1e6).toFixed(1)+'M':Math.round(v/1e3)+'k'; }}}}
  };
  priors.forEach(function(y){
    var cv = document.getElementById('sg-cv-'+y); if (!cv) return;
    var ch = new Chart(cv, {type:'line', data:{labels:SEAS_MONTHS, datasets:[
      {data:curArr, borderColor:cfg.color, borderWidth:2, pointRadius:0, tension:0, spanGaps:false, order:0},
      {data:ind.byYear[y], borderColor:'#5ba3e8', borderWidth:1.6, pointRadius:0, tension:0, spanGaps:false, order:1}]},
      options:smOpts, plugins:[seasCrosshair]});
    ch.$year = y; CH['sg-'+y] = ch;
  });
  // synced hover: hovering any panel lights the same month in every panel + corner readouts
  priors.forEach(function(y){
    var ch = CH['sg-'+y]; if (!ch) return;
    ch.canvas.onmousemove = function(e){
      var mi = seasNearestMonth(ch, e.clientX);
      priors.forEach(function(yy){
        var c2 = CH['sg-'+yy]; if (!c2) return;
        c2.$hoverMonth = mi; c2.draw();
        var hv = document.getElementById('sgh-'+yy);
        hv.innerHTML = '<div style="color:'+cfg.color+';">'+fmt(curArr[mi])+'</div>'
          + '<div style="color:#5ba3e8;">'+fmt(ind.byYear[yy][mi])+'</div>';
        hv.classList.add('on');
      });
    };
    ch.canvas.onmouseleave = function(){
      priors.forEach(function(yy){
        var c2 = CH['sg-'+yy]; if (c2){ c2.$hoverMonth = null; c2.draw(); }
        var hv = document.getElementById('sgh-'+yy); if (hv) hv.classList.remove('on');
      });
    };
  });
}

/* ═══════════════════════════════════════════════════════════
   TABLE
═══════════════════════════════════════════════════════════ */
function setFreq(f, btn) {
  tblFreq = f;
  document.querySelectorAll('#freqBtns .vlm-btn').forEach(function(b){ b.classList.remove('act'); });
  btn.classList.add('act');
  buildTbl();
}
function populateTblContract() {
  var comm = document.getElementById('tblComm').value;
  var cd   = DATA.commodities[comm]; if (!cd) return;
  var sel  = document.getElementById('tblContract');
  var prev = sel.value;
  sel.innerHTML = '<option value="Aggregate">AGGREGATE</option>';
  var tblSeqLabels = buildContractLabels(cd);
  (cd.ordered || []).forEach(function(tk) {
    var lbl = tk.replace(' Comdty', '');
    var opt = document.createElement('option');
    opt.value = lbl;
    opt.textContent = tblSeqLabels[tk] || lbl;   // e.g. "MAY 1", "MAY 2"
    sel.appendChild(opt);
  });
  if ([].slice.call(sel.options).some(function(o){ return o.value===prev; })) sel.value = prev;
}
function onTblCommChange() { populateTblContract(); buildTbl(); }

async function buildTbl() {
  var tbl      = document.getElementById('histTbl'); if (!tbl) return;
  var comm     = document.getElementById('tblComm').value;
  var contract = document.getElementById('tblContract').value || 'Aggregate';
  var cd       = DATA.commodities[comm]; if (!cd) return;
  var from     = document.getElementById('tblFrom').value || '2023-01-01';
  var to       = document.getElementById('tblTo').value   || DATA.last_date;

  /* Fetch full history for table */
  await ensureHist(comm);
  var src = HIST[comm];
  var rawRows;
  if (contract === 'Aggregate') {
    var agg = src ? src.daily_agg : cd.daily_agg;
    var keys = Object.keys(agg).sort();
    rawRows = keys.map(function(d, i) {
      return {date:d, open_int:agg[d],
              oi_chg: i > 0 ? agg[d] - agg[keys[i-1]] : null};
    });
  } else {
    var th = src ? src.ticker_history : cd.ticker_history;
    var tkRows = ((th || {})[contract] || []).slice().sort(function(a,b){ return a.date<b.date?-1:1; });
    rawRows = tkRows.map(function(r, i) {
      var newContract = i > 0 && r.contract && tkRows[i-1].contract && r.contract !== tkRows[i-1].contract;
      return {
        date:     r.date,
        open_int: r.open_int,
        contract: r.contract,
        oi_chg:   (i > 0 && !newContract) ? r.open_int - tkRows[i-1].open_int : null
      };
    });
  }

  var rows = rawRows.filter(function(r){ return r.date>=from && r.date<=to && r.open_int!=null; });

  if (tblFreq === 'weekly') {
    var w = {};
    rows.forEach(function(r) {
      var d  = new Date(r.date);
      var wn = Math.floor((d - new Date(d.getFullYear(),0,1))/604800000);
      w[d.getFullYear() + '-' + String(wn).padStart(2,'0')] = r;
    });
    rows = Object.values(w).sort(function(a,b){ return a.date<b.date?-1:1; });
  }
  if (tblFreq === 'monthly') {
    var m = {};
    rows.forEach(function(r){ m[r.date.slice(0,7)] = r; });
    rows = Object.values(m).sort(function(a,b){ return a.date<b.date?-1:1; });
  }

  var todayDt = new Date(DATA.last_date);
  var hist5   = rawRows.filter(function(r) {
    return ((todayDt - new Date(r.date))/86400000) <= 365*5 && r.open_int;
  }).map(function(r){ return r.open_int; });
  var hi5 = hist5.length ? Math.max.apply(null, hist5) : null;
  var lo5 = hist5.length ? Math.min.apply(null, hist5) : null;

  tbl.innerHTML = '<thead><tr>'
    + '<th>Date</th><th>Contract</th><th>Open Int</th><th>OI Chg</th>'
    + '<th>% Chg</th><th>vs 5yr Hi</th><th>5yr Hi</th><th>5yr Lo</th>'
    + '</tr></thead>';
  var tbody = document.createElement('tbody');
  rows.slice().reverse().forEach(function(r) {
    var pct  = r.open_int ? (((r.oi_chg||0)/r.open_int)*100).toFixed(2) : '0.00';
    var vsHi = hi5 ? ((r.open_int/hi5-1)*100).toFixed(1) : '—';
    var tr = document.createElement('tr');
    var ctLabel = r.contract ? fmtContract(r.contract) : '—';
    tr.innerHTML =
      '<td>' + r.date + '</td>'
      + '<td style="color:var(--dim);font-size:11px;letter-spacing:.5px;">' + ctLabel + '</td>'
      + '<td>' + f0(r.open_int) + '</td>'
      + '<td class="' + ((r.oi_chg||0)>=0?'vlm-pos':'vlm-neg') + '">' + fc(r.oi_chg) + '</td>'
      + '<td class="' + (parseFloat(pct)>=0?'vlm-pos':'vlm-neg') + '">' + (parseFloat(pct)>=0?'+':'') + pct + '%</td>'
      + '<td class="' + (parseFloat(vsHi)>=0?'vlm-pos':'vlm-neg') + '">' + vsHi + '%</td>'
      + '<td class="vlm-muted">' + f0(hi5) + '</td>'
      + '<td class="vlm-muted">' + f0(lo5) + '</td>';
    tbody.appendChild(tr);
  });
  tbl.appendChild(tbody);
}

/* ── Init ── */
var _optHistCache = null;
var _optComm = 'CT';
var _optData = {};

var OPT_COMM_CFG = {
  CT: {label:'COTTON (CT)',  color:'#E8C547'},
  KC: {label:'COFFEE (KC)',  color:'#C8956D'},
  CC: {label:'COCOA (CC)',   color:'#A07855'},
  SB: {label:'SUGAR (SB)',   color:'#E07B54'},
};

/* ── Init ── */
setInterval(function() {
  var el = document.getElementById('clk');
  if (el) el.textContent = new Date().toLocaleTimeString('en-US',
    {hour12:false, timeZone:'America/New_York'}) + ' EST';
}, 1000);

document.getElementById('tblTo').value  = DATA.last_date;
document.getElementById('asof').textContent = 'As of: ' + DATA.last_date + ' 09:35 EST';

var maxYrs = Math.max.apply(null, Object.values(DATA.commodities).map(function(c){ return c.max_years||18; }));
var slider = document.getElementById('seasYrSlider');
if (slider) { slider.max = maxYrs; }

/* Light is the default (VLM master palette); DARK is an opt-in toggle.
   Previously this was gated on the OS prefers-color-scheme, so a dark-mode
   machine loaded the dark theme -- light must not depend on the OS. */
document.getElementById('themeBtn').textContent = 'DARK';
initTooltip();
populateTblContract();
populateSeasContract();
buildMonitor();

function setOptComm(comm, btn) {
  _optComm = comm;
  document.querySelectorAll('.opt-comm-btn').forEach(function(b){ b.classList.remove('act'); });
  btn.classList.add('act');
  renderOptTab();
}

function buildOptions() {
  // Inject commodity selector buttons, then load CT data if not cached
  var root = document.getElementById('optRoot');
  if (!root) return;

  // Seed CT from inline OPTDATA on first load
  if (!_optData['CT'] && OPTDATA && OPTDATA.months) {
    _optData['CT'] = OPTDATA;
  }

  // Build selector bar (persists across re-renders)
  var GREEN='#22c55e', RED='#ef4444', DIM='#64748b', WHITE='#f1f5f9';
  var DARK='#0d1520', ALT='#0a1018', MONTH_BG='#0f1e35', HDR_BG='#080f1a', GOLD='#E8C547';

  var selectorHtml = '<div style="display:flex;align-items:center;gap:8px;padding:8px 14px;'+
    'border-bottom:1px solid #1e3a5f;margin-bottom:10px;">'+
    '<span style="font-size:11px;font-weight:700;letter-spacing:1px;color:'+DIM+';">COMMODITY</span>';
  Object.keys(OPT_COMM_CFG).forEach(function(c) {
    var cfg = OPT_COMM_CFG[c];
    selectorHtml += '<button class="vlm-btn opt-comm-btn'+(c===_optComm?' act':'')+
      '" onclick="setOptComm(\\''+c+'\\',this)" '+
      'style="letter-spacing:1px;font-size:11px;">'+cfg.label+'</button>';
  });
  selectorHtml += '<span id="optLoadStatus" style="font-size:11px;color:'+DIM+';margin-left:8px;"></span>';
  selectorHtml += '</div>';
  selectorHtml += '<div id="optContent"></div>';
  root.innerHTML = selectorHtml;

  renderOptTab();
}

function renderOptTab() {
  var status = document.getElementById('optLoadStatus');
  var content = document.getElementById('optContent');
  if (!content) return;

  // If we have cached data for this commodity, render immediately
  if (_optData[_optComm]) {
    _renderOptContent(_optData[_optComm]);
    return;
  }

  // Otherwise fetch from API
  if (status) status.textContent = 'Loading...';
  content.innerHTML = '<div style="padding:40px;color:var(--muted);font-size:13px;">Loading '+_optComm+' options data...</div>';
  fetch('/api/options/data?commodity='+_optComm)
    .then(function(r){ return r.json(); })
    .then(function(data) {
      _optData[_optComm] = data;
      if (status) status.textContent = '';
      _renderOptContent(data);
    })
    .catch(function(e) {
      if (status) status.textContent = 'Error loading data';
      content.innerHTML = '<div style="padding:40px;color:#ef4444;font-size:13px;">Error: '+e.message+'</div>';
    });
}

function _renderOptContent(od) {
  var content = document.getElementById('optContent');
  if (!content) return;
  if (!od || !od.months || !od.months.length) {
    content.innerHTML = '<div style="padding:40px;color:var(--muted);font-size:13px;">No options data available for '+_optComm+'.</div>';
    return;
  }

  var GREEN='#22c55e', RED='#ef4444', DIM='#64748b', WHITE='#f1f5f9';
  var DARK='#0d1520', ALT='#0a1018', MONTH_BG='#0f1e35', HDR_BG='#080f1a', GOLD='#E8C547';
  var commColor = OPT_COMM_CFG[_optComm] ? OPT_COMM_CFG[_optComm].color : GOLD;
  var GRID = '1fr 1fr 1fr 1fr 1fr';

  function fmtN(v)   { if(v===null||v===undefined||v==='') return '—'; return Number(v).toLocaleString(); }
  function fmtS(v)   { if(v===null||v===undefined||v==='') return '—'; return Number(v).toFixed(2); }
  function fmtC(v)   { if(v===null||v===undefined||v==='') return '—'; return (Number(v)>=0?'+':'')+Number(v).toLocaleString(); }
  function cOI(oi)   { return oi>=5000?commColor:oi>=1000?GREEN:WHITE; }
  function cChg(v)   { if(v===null||v===undefined||v==='') return DIM; return v>0?GREEN:v<0?RED:DIM; }

  function colHdr(label) {
    return '<div style="font-size:11px;font-weight:700;color:'+DIM+';text-align:right;'+
           'letter-spacing:.6px;padding:4px 8px;border-bottom:1px solid #1e3a5f;">'+label+'</div>';
  }
  function cell(content, color, size) {
    return '<div style="font-size:'+(size||14)+'px;font-weight:700;color:'+(color||WHITE)+
           ';text-align:right;padding:4px 8px;">'+content+'</div>';
  }

  function buildSection(label, color, pcKey, bg) {
    var html = '<div style="padding:8px 14px 6px;background:'+bg+';border-left:4px solid '+color+
               ';margin-bottom:6px;display:flex;align-items:center;justify-content:space-between;">'+
               '<span style="font-size:13px;font-weight:700;letter-spacing:2px;color:'+color+';">◆ '+label+'</span>'+
               '</div>';

    od.months.forEach(function(month) {
      var rows = (od[pcKey]&&od[pcKey][month]) ? od[pcKey][month] : [];
      if (!rows.length) return;
      var totOI  = rows.reduce(function(s,r){return s+r.oi;},0);
      var totVol = rows.reduce(function(s,r){return s+(r.vol||0);},0);
      var totChg = rows.reduce(function(s,r){return s+(r.chg||0);},0);

      html += '<div style="background:'+MONTH_BG+';padding:5px 14px;margin-top:4px;'+
              'display:flex;align-items:center;justify-content:space-between;">'+
              '<span style="font-size:13px;font-weight:700;color:'+color+';">'+month+'</span>'+
              '<span style="font-size:11px;color:'+DIM+';">'+
              'OI: <span style="color:'+WHITE+';">'+totOI.toLocaleString()+'</span>'+
              '&nbsp;&nbsp;Vol: <span style="color:'+WHITE+';">'+totVol.toLocaleString()+'</span>'+
              '&nbsp;&nbsp;OI Chg: <span style="color:'+cChg(totChg)+';">'+fmtC(totChg)+'</span>'+
              '</span></div>';

      html += '<div style="display:grid;grid-template-columns:'+GRID+';background:'+HDR_BG+';">'+
              colHdr('STRIKE')+colHdr('OPEN INT')+colHdr('OI CHG')+colHdr('SETTLE')+colHdr('VOLUME')+
              '</div>';

      rows.forEach(function(r, idx) {
        html += '<div style="display:grid;grid-template-columns:'+GRID+';background:'+(idx%2===0?DARK:ALT)+';">'+
                cell(r.strike.toFixed(2))+
                cell(fmtN(r.oi), cOI(r.oi))+
                cell(fmtC(r.chg), cChg(r.chg))+
                cell(fmtS(r.settle))+
                cell(fmtN(r.vol), DIM)+
                '</div>';
      });

      html += '<div style="display:grid;grid-template-columns:'+GRID+';background:#0a1525;'+
              'border-top:1px solid #1e3a5f;margin-bottom:8px;">'+
              '<div style="font-size:11px;font-weight:700;color:'+DIM+';padding:4px 8px;text-align:right;">TOTAL</div>'+
              cell(fmtN(totOI), commColor)+
              cell(fmtC(totChg), cChg(totChg))+
              '<div></div>'+
              cell(fmtN(totVol), DIM)+
              '</div>';
    });
    return html;
  }

  // ── History search UI ──
  // Build sorted unique strike list from all months
  var allStrikes = [];
  od.months.forEach(function(m) {
    ['calls','puts'].forEach(function(k) {
      if (od[k]&&od[k][m]) od[k][m].forEach(function(r) {
        if (allStrikes.indexOf(r.strike)<0) allStrikes.push(r.strike);
      });
    });
  });
  allStrikes.sort(function(a,b){return a-b;});

  var SEL = 'background:#0d1520;border:1px solid #2a3548;color:'+WHITE+';padding:4px 8px;font-size:12px;border-radius:3px;';
  var histHtml = '<div style="background:#0a1525;border:1px solid #1e3a5f;border-radius:4px;'+
                 'padding:10px 14px;margin-bottom:12px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;">'+
                 '<span style="font-size:11px;font-weight:700;letter-spacing:1px;color:'+DIM+';">HISTORY SEARCH</span>'+
                 '<select id="optHMonth" style="'+SEL+'">'+
                 '<option value="">All Months</option>'+
                 od.months.map(function(m){return '<option>'+m+'</option>';}).join('')+
                 '</select>'+
                 '<div style="display:flex;flex-direction:column;gap:2px;align-items:flex-start;">'+
                 '<span style="font-size:9px;color:#6b8090;letter-spacing:.5px;font-family:inherit;text-transform:uppercase;">Strike — ctrl+click multi</span>'+
                 '<select id="optHStrike" multiple size="5" style="'+SEL+';cursor:pointer;min-width:90px;">'+
                 allStrikes.map(function(s){return '<option value="'+s+'">'+s.toFixed(2)+'</option>';}).join('')+
                 '</select></div>'+
                 '<select id="optHPC" style="'+SEL+'">'+
                 '<option value="">C & P</option><option value="C">Calls</option><option value="P">Puts</option>'+
                 '</select>'+
                 '<span style="font-size:11px;color:'+DIM+';">FROM</span>'+
                 '<input id="optHFrom" type="date" style="'+SEL+'">'+
                 '<span style="font-size:11px;color:'+DIM+';">TO</span>'+
                 '<input id="optHTo" type="date" style="'+SEL+'">'+
                 '<button onclick="runOptHistory()" style="background:#1e3a5f;border:1px solid #3b82f6;'+
                 'color:'+WHITE+';padding:4px 14px;font-size:12px;border-radius:3px;cursor:pointer;font-weight:700;">GO</button>'+
                 '<button onclick="clearOptHistory()" '+
                 'style="background:transparent;border:1px solid #2a3548;color:'+DIM+';'+
                 'padding:4px 10px;font-size:12px;border-radius:3px;cursor:pointer;">CLEAR</button>'+
                 '<span id="optHStatus" style="font-size:11px;color:'+DIM+';"></span>'+
                 '</div>'+
                 '<div id="optHResult"></div>';

  var commName = OPT_COMM_CFG[_optComm] ? OPT_COMM_CFG[_optComm].label : _optComm+' OPTIONS';
  var out = '<div style="display:flex;align-items:center;padding:8px 14px;'+
            'border-bottom:1px solid #1e3a5f;margin-bottom:10px;">'+
            '<span style="font-size:12px;font-weight:700;letter-spacing:2px;color:'+DIM+';">'+
            commName.toUpperCase()+' OPTIONS — OI BY CONTRACT MONTH</span>'+
            '<span style="font-size:11px;color:#475569;margin-left:auto;">'+od.last_date+' &nbsp;·&nbsp; as of trade date '+_prevBday(od.last_date)+'</span></div>';
  out += histHtml;
  out += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">';
  out += '<div>'+buildSection('CALLS', GREEN, 'calls', '#0a2010')+'</div>';
  out += '<div>'+buildSection('PUTS',  RED,   'puts',  '#20080a')+'</div>';
  out += '</div>';
  content.innerHTML = out;
}

/* ═══════════════════════════════════════════════════════════
   PNG EXPORT — VLM OI MONITOR
═══════════════════════════════════════════════════════════ */
function _oiPngHdr(titleTxt) {
  var rd = DATA.last_date;
  return '<div style="background:#0d1117;padding:14px 24px 10px;border-bottom:1px solid #1e2d3d;display:flex;align-items:center;">'
    + '<div><div style="font-size:26px;font-weight:700;color:#fff;letter-spacing:1px;line-height:1;font-family:Arial,sans-serif;">VLM COMMODITIES</div>'
    + '<div style="font-size:13px;color:#4a9aba;letter-spacing:2px;text-transform:uppercase;margin-top:4px;font-family:Arial,sans-serif;">OPEN INTEREST MONITOR</div></div>'
    + '</div>'
    + '<div style="background:#111825;padding:7px 24px;border-bottom:3px solid #d4b44a;display:flex;gap:32px;align-items:center;">'
    + '<span style="font-size:13px;color:#6b8090;font-family:Arial,sans-serif;">As of&nbsp;<strong style="color:#e0e8f0;">' + rd + ' 09:35 EST</strong></span>'
    + '<span style="font-size:13px;color:#6b8090;font-family:Arial,sans-serif;">Source&nbsp;<strong style="color:#e0e8f0;">Bloomberg / ICE</strong></span>'
    + '</div>'
    + '<div style="background:#151e2a;padding:7px 24px;font-size:14px;font-weight:700;color:#d4b44a;letter-spacing:2px;text-transform:uppercase;font-family:Arial,sans-serif;border-left:4px solid #d4b44a;">' + titleTxt + '</div>';
}
function _oiPngFtr() {
  return '<div style="background:#0d1117;padding:7px 24px;font-size:12px;color:#2a3a48;font-family:Arial,sans-serif;border-top:1px solid #1e2d3d;">VLM Commodities &nbsp;|&nbsp; Open Interest Monitor &nbsp;|&nbsp; vlmdata.com &nbsp;|&nbsp; ' + DATA.last_date + '</div>';
}
/* Prior business day of a YYYY-MM-DD string (weekend-only shift). Parses date
   components explicitly to avoid new Date('YYYY-MM-DD') UTC-midnight timezone drift. */
function _prevBday(dStr) {
  var p = dStr.split('-'); var d = new Date(+p[0], +p[1]-1, +p[2]);
  d.setDate(d.getDate() - 1);
  while (d.getDay() === 0 || d.getDay() === 6) d.setDate(d.getDate() - 1);
  var mm = String(d.getMonth()+1).padStart(2,'0'), dd = String(d.getDate()).padStart(2,'0');
  return d.getFullYear() + '-' + mm + '-' + dd;
}
/* Options-only PNG header/footer. Options rows are RELEASE-date-stamped and carry the
   PRIOR session's OI (standard OI convention). Headline shows the release date; subtext
   shows "as of trade date X". Mirrors _oiPngHdr styling but never uses DATA.last_date
   (the futures date), which would be the wrong convention for an options image. */
function _oiPngHdrOpt(titleTxt, release, trade) {
  return '<div style="background:#0d1117;padding:14px 24px 10px;border-bottom:1px solid #1e2d3d;display:flex;align-items:center;">'
    + '<div><div style="font-size:26px;font-weight:700;color:#fff;letter-spacing:1px;line-height:1;font-family:Arial,sans-serif;">VLM COMMODITIES</div>'
    + '<div style="font-size:13px;color:#4a9aba;letter-spacing:2px;text-transform:uppercase;margin-top:4px;font-family:Arial,sans-serif;">OPTIONS OPEN INTEREST</div></div>'
    + '</div>'
    + '<div style="background:#111825;padding:7px 24px;border-bottom:3px solid #d4b44a;display:flex;gap:32px;align-items:center;">'
    + '<span style="font-size:13px;color:#6b8090;font-family:Arial,sans-serif;">' + release + '&nbsp;<strong style="color:#e0e8f0;">as of trade date ' + trade + '</strong></span>'
    + '<span style="font-size:13px;color:#6b8090;font-family:Arial,sans-serif;">Source&nbsp;<strong style="color:#e0e8f0;">Bloomberg / ICE</strong></span>'
    + '</div>'
    + '<div style="background:#151e2a;padding:7px 24px;font-size:14px;font-weight:700;color:#d4b44a;letter-spacing:2px;text-transform:uppercase;font-family:Arial,sans-serif;border-left:4px solid #d4b44a;">' + titleTxt + '</div>';
}
function _oiPngFtrOpt(release) {
  return '<div style="background:#0d1117;padding:7px 24px;font-size:12px;color:#2a3a48;font-family:Arial,sans-serif;border-top:1px solid #1e2d3d;">VLM Commodities &nbsp;|&nbsp; Options Open Interest &nbsp;|&nbsp; vlmdata.com &nbsp;|&nbsp; ' + release + '</div>';
}
function _oiPngRender(innerHtml, filename, width) {
  var w = width || 1300;
  var wrap = document.createElement('div');
  wrap.style.cssText = 'position:fixed;left:-9999px;top:0;width:'+w+'px;background:#f0f2f5;font-family:Arial,sans-serif;';
  /* Inject light-mode CSS tokens so all var(--x) references resolve to light palette */
  var lightToks = {'--bg':'#f0f2f5','--surf':'#ffffff','--surf2':'#f4f6f8','--hdr':'#e8edf2',
                   '--bord':'#c8d4e0','--bord2':'#b0bfd0','--text':'#1a202c','--dim':'#2d3748',
                   '--muted':'#4a5568','--grn':'#15803d','--red':'#b91c1c',
                   '--acc':'#92680a','--gold':'#92680a','--blue':'#1e40af'};
  Object.keys(lightToks).forEach(function(k){ wrap.style.setProperty(k, lightToks[k]); });
  wrap.innerHTML = innerHtml;
  document.body.appendChild(wrap);
  /* remove sticky so html2canvas captures full table */
  wrap.querySelectorAll('[style*="sticky"],[style*="position:sticky"]').forEach(function(el){
    el.style.position = 'relative';
  });
  var hd = wrap.querySelector('.grid-head');
  if (hd) hd.style.position = 'relative';
  html2canvas(wrap, {scale:2, useCORS:true, backgroundColor:'#0d1117', logging:false})
    .then(function(canvas) {
      document.body.removeChild(wrap);
      var a = document.createElement('a');
      a.download = filename;
      a.href = canvas.toDataURL('image/png');
      a.click();
    }).catch(function(e) {
      document.body.removeChild(wrap);
      alert('Export failed: ' + e.message);
    });
}

function exportCurrentTabPng() {
  var active = document.querySelector('.tab-content.act');
  if (!active) return;
  var id = active.id.replace('tab-','');
  if      (id === 'monitor')  exportMonitorPng();
  else if (id === 'seasonal') exportSeasPng();
  else if (id === 'table')    exportTablePng();
  else if (id === 'options')  exportOptionsPng();
}

function exportMonitorPng() {
  var body = document.getElementById('monBody'); if (!body) return;
  var clone = body.cloneNode(true);
  /* hide expanded chart panels — they don't render well offscreen */
  clone.querySelectorAll('.chart-panel').forEach(function(el){ el.remove(); });
  var inner = _oiPngHdr('OPEN INTEREST MONITOR') + '<div style="background:#080b0f;">' + clone.outerHTML + '</div>' + _oiPngFtr();
  _oiPngRender(inner, 'OI_Monitor_' + DATA.last_date + '.png', 1500);
}

function exportSeasPng() {
  var contractSel = (document.getElementById('seasContract') || {}).value || 'Aggregate';
  var modeLbl     = seasMode === 'band' ? 'HI/AVG/LO'
                    : (seasLayout === 'grid' ? 'INDIVIDUAL YRS · GRID' : 'INDIVIDUAL YRS · SPAGHETTI');
  var titleTxt    = 'SEASONAL OI · ' + contractSel.toUpperCase() + ' · ' + modeLbl + ' · ' + seasYr + 'YR';
  var curYear     = new Date(DATA.last_date).getFullYear();
  var months      = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

  /* current-year data table for PNG annotation */
  function getCur(comm) {
    var hist = getSeasHist(comm);
    var cur  = Array(12).fill(null);
    hist.filter(function(r){ return new Date(r.date).getFullYear()===curYear&&r.open_int; })
        .forEach(function(r){ var m=new Date(r.date).getMonth(); if(cur[m]===null||r.open_int>cur[m])cur[m]=r.open_int; });
    return cur;
  }
  var commsToShow = [document.getElementById('seasSingleComm').value];  // always single-commodity now
  var hdrRow = '<tr style="background:#e8edf2;">'
    + '<td style="padding:4px 10px;font-size:11px;color:#4a5568;font-family:Arial,sans-serif;border-right:1px solid #c8d4e0;font-weight:700;">' + curYear + '</td>'
    + months.map(function(mo){ return '<td style="padding:4px 8px;font-size:11px;color:#4a5568;text-align:right;font-family:Arial,sans-serif;border-right:1px solid #c8d4e0;">' + mo + '</td>'; }).join('')
    + '</tr>';
  var dataRows = commsToShow.map(function(comm, ri) {
    var cur = getCur(comm);
    var bg  = ri%2===0 ? '#ffffff' : '#f4f7fa';
    return '<tr style="background:'+bg+';">'
      + '<td style="padding:4px 10px;font-size:12px;font-weight:700;color:'+CFG[comm].color+';font-family:Arial,sans-serif;border-right:1px solid #c8d4e0;">'+comm+'</td>'
      + cur.map(function(v){ return '<td style="padding:4px 8px;font-size:12px;font-weight:700;text-align:right;font-family:Arial,sans-serif;border-right:1px solid #c8d4e0;color:'+(v!=null?'#1a202c':'#94a3b8')+'">'+(v!=null?(v/1000).toFixed(0)+'k':'—')+'</td>'; }).join('')
      + '</tr>';
  }).join('');
  var curTableHtml = '<div style="background:#e8edf2;border-top:2px solid #d4b44a;padding:10px 24px 14px;">'
    + '<div style="font-size:11px;font-weight:700;color:#92680a;letter-spacing:1px;margin-bottom:6px;font-family:Arial,sans-serif;">' + curYear + ' CURRENT PERIOD DATA</div>'
    + '<table style="border-collapse:collapse;width:100%;border:1px solid #c8d4e0;">'
    + '<tbody>' + hdrRow + dataRows + '</tbody></table></div>';

  /* Re-render a Chart.js chart on an offscreen canvas with white background + light grid.
     opt = {yMin,yMax,fontSize,maxYTicks} for the grid small-multiples (shared y-scale). */
  function lightChartImg(ds, labels, w, h, opt) {
    opt = opt || {};
    return new Promise(function(resolve) {
      var cv = document.createElement('canvas');
      cv.width = w; cv.height = h;
      cv.style.cssText = 'position:fixed;left:-9999px;top:0;';
      document.body.appendChild(cv);
      var whiteBg = { id:'whitebg', beforeDraw:function(ch){
        ch.ctx.save(); ch.ctx.fillStyle='#ffffff';
        ch.ctx.fillRect(0,0,ch.canvas.width,ch.canvas.height); ch.ctx.restore();
      }};
      var fs = opt.fontSize || 9;
      var chart = new Chart(cv, {
        type:'line', data:{labels:labels, datasets:ds},
        options:{
          responsive:false, maintainAspectRatio:false, animation:{duration:0},
          plugins:{ legend:{display:false}, tooltip:{enabled:false} },
          scales:{
            x:{ grid:{color:'#d0dce8'}, ticks:{color:'#2d3748', font:{size:fs}, maxTicksLimit:opt.maxXTicks, maxRotation:0} },
            y:{ min:opt.yMin, max:opt.yMax, grid:{color:'#d0dce8'},
                ticks:{color:'#2d3748', font:{size:fs}, maxTicksLimit:opt.maxYTicks, callback:tk} }
          }
        },
        plugins:[whiteBg]
      });
      setTimeout(function(){
        resolve(cv.toDataURL('image/png'));
        chart.destroy();
        document.body.removeChild(cv);
      }, 150);
    });
  }

  /* Compact text legend with hardcoded light colors */
  function buildLegLight(card) {
    var items = [];
    if (seasMode === 'band') {
      items.push({swatch:'box', color:card.cfg.color+'88', bg:card.cfg.ca, label:seasYr+'yr Range'});
      items.push({swatch:'line', color:'#2d3748', bg:null, label:seasYr+'yr Avg'});
      items.push({swatch:'line', color:card.cfg.color, bg:null, label:String(curYear)});
    } else {
      card.ds.forEach(function(d){
        items.push({swatch:'line', color:d.borderColor, bg:null, label:d.label});
      });
    }
    return items.map(function(it) {
      var sw = it.swatch==='box'
        ? '<span style="display:inline-block;width:16px;height:8px;background:'+it.bg+';border:1px solid '+it.color+';border-radius:2px;vertical-align:middle;margin-right:4px;"></span>'
        : '<span style="display:inline-block;width:16px;height:2px;background:'+it.color+';vertical-align:middle;margin-right:4px;"></span>';
      return '<span style="font-size:11px;color:#2d3748;font-weight:600;font-family:Arial,sans-serif;margin-right:12px;white-space:nowrap;">'+sw+it.label+'</span>';
    }).join('');
  }

  /* PNG mirrors whatever is on screen: band | spaghetti | grid (single-commodity). */
  var sComm  = document.getElementById('seasSingleComm').value;
  var hdrEl  = document.getElementById('seasSingleHdr');
  var hdrTxt = hdrEl ? hdrEl.textContent : sComm + ' ' + CFG[sComm].name;

  if (seasMode === 'individual' && seasLayout === 'grid') {
    /* GRID small-multiples: one panel per prior year, gold current + blue prior,
       shared y-scale — same as buildSeasGrid() on screen. */
    var cfg     = CFG[sComm];
    var ind     = computeIndividual(getSeasHist(sComm), curYear);
    var yr      = seasYRange(ind);
    var curArr  = ind.byYear[curYear] || Array(12).fill(null);
    var priors  = ind.years.filter(function(y){ return y !== curYear; });
    Promise.all(priors.map(function(y){
      var ds = [
        {data:curArr,         borderColor:cfg.color, borderWidth:2,   pointRadius:0, tension:0, spanGaps:false, order:0},
        {data:ind.byYear[y],  borderColor:'#5ba3e8', borderWidth:1.6, pointRadius:0, tension:0, spanGaps:false, order:1},
      ];
      return lightChartImg(ds, months, 300, 150, {yMin:yr.min, yMax:yr.max, fontSize:8, maxYTicks:3, maxXTicks:6})
        .then(function(img){ return {year:y, img:img}; });
    })).then(function(cells) {
      var cellHtml = cells.map(function(c){
        return '<div style="background:#ffffff;border:1px solid #c8d4e0;border-radius:4px;padding:8px 8px 6px;">'
          + '<div style="font-size:12px;font-weight:800;color:'+cfg.color+';font-family:Arial,sans-serif;">'+c.year+'</div>'
          + '<div style="font-size:10px;color:#4a5568;font-family:Arial,sans-serif;margin-bottom:3px;">vs '+curYear+'</div>'
          + '<img src="'+c.img+'" style="width:100%;height:112px;object-fit:fill;display:block;">'
          + '</div>';
      }).join('');
      var legRow = '<div style="margin:2px 0 8px;font-family:Arial,sans-serif;">'
        + '<span style="font-size:11px;color:#2d3748;font-weight:600;margin-right:12px;"><span style="display:inline-block;width:16px;height:2px;background:'+cfg.color+';vertical-align:middle;margin-right:4px;"></span>'+curYear+' (current)</span>'
        + '<span style="font-size:11px;color:#2d3748;font-weight:600;"><span style="display:inline-block;width:16px;height:2px;background:#5ba3e8;vertical-align:middle;margin-right:4px;"></span>prior year</span></div>';
      var inner = _oiPngHdr(titleTxt)
        + '<div style="background:#f0f2f5;padding:10px 24px 12px;">'
        + '<div style="font-size:13px;font-weight:700;color:'+cfg.color+';font-family:Arial,sans-serif;margin-bottom:2px;">'+hdrTxt+'</div>'
        + legRow
        + '<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;">'+cellHtml+'</div>'
        + '</div>'
        + curTableHtml + _oiPngFtr();
      _oiPngRender(inner, 'OI_Seasonal_' + DATA.last_date + '.png', 1500);
    });
    return;
  }

  /* BAND or SPAGHETTI: single multi-line card (unchanged). */
  var card = buildSeasCard(sComm);
  lightChartImg(card.ds, card.labels, 1012, 420).then(function(img) {
    var inner = _oiPngHdr(titleTxt)
      + '<div style="background:#ffffff;padding:10px 24px 8px;">'
      + '<div style="font-size:13px;font-weight:700;color:'+card.cfg.color+';font-family:Arial,sans-serif;margin-bottom:4px;">'+hdrTxt+'</div>'
      + '<div style="margin-bottom:6px;">'+buildLegLight(card)+'</div>'
      + '<img src="'+img+'" style="width:100%;height:420px;object-fit:fill;display:block;border:1px solid #e0e8f0;">'
      + '</div>'
      + curTableHtml + _oiPngFtr();
    _oiPngRender(inner, 'OI_Seasonal_' + DATA.last_date + '.png', 1060);
  });
}

function exportTablePng() {
  var tbl = document.getElementById('histTbl'); if (!tbl) return;
  var comm     = document.getElementById('tblComm').value;
  var contract = (document.getElementById('tblContract').value || 'Aggregate');
  var freq     = tblFreq.charAt(0).toUpperCase() + tblFreq.slice(1);
  var titleTxt = comm + ' OI · ' + contract + ' · ' + freq;
  var inner = _oiPngHdr(titleTxt) + '<div style="background:#080b0f;overflow:hidden;">' + tbl.outerHTML + '</div>' + _oiPngFtr();
  _oiPngRender(inner, 'OI_Table_'+comm+'_'+DATA.last_date+'.png', 1060);
}

function exportOptionsPng() {
  var od   = _optData[_optComm]; if (!od || !od.months) { alert('Load options data first'); return; }
  var comm = _optComm;
  var titleTxt = (OPT_COMM_CFG[comm]?OPT_COMM_CFG[comm].label:comm) + ' OPTIONS OI';
  /* Rebuild options content with light-mode hardcoded colors for PNG */
  var GREEN='#15803d', RED='#b91c1c', DIM='#4a5568', BLACK='#1a202c';
  var ROW1='#ffffff',  ROW2='#f4f7fa', MON_BG='#e8edf2', HDR_BG='#c8d4e0', GOLD='#92680a';
  var GRID = '1fr 1fr 1fr 1fr 1fr';
  function fmtN(v){if(v===null||v===undefined||v==='')return '—';return Number(v).toLocaleString();}
  function fmtS(v){if(v===null||v===undefined||v==='')return '—';return Number(v).toFixed(2);}
  function fmtC(v){if(v===null||v===undefined||v==='')return '—';return(Number(v)>=0?'+':'')+Number(v).toLocaleString();}
  function cOI(oi){return oi>=5000?GOLD:oi>=1000?GREEN:BLACK;}
  function cChg(v){if(v===null||v===undefined||v==='')return DIM;return v>0?GREEN:v<0?RED:DIM;}
  function colHdr(lbl){return '<div style="font-size:11px;font-weight:700;color:'+DIM+';text-align:right;letter-spacing:.6px;padding:4px 8px;border-bottom:1px solid #c8d4e0;background:'+HDR_BG+';">'+lbl+'</div>';}
  function cell(c,color,size){return '<div style="font-size:'+(size||14)+'px;font-weight:700;color:'+(color||BLACK)+';text-align:right;padding:4px 8px;">'+c+'</div>';}
  function buildSec(label,color,pcKey,bg) {
    var html='<div style="padding:6px 14px;background:'+bg+';border-left:4px solid '+color+';margin-bottom:4px;">'
      +'<span style="font-size:13px;font-weight:700;letter-spacing:2px;color:'+color+';">◆ '+label+'</span></div>';
    od.months.forEach(function(month) {
      var rows=(od[pcKey]&&od[pcKey][month])?od[pcKey][month]:[];
      if(!rows.length)return;
      var totOI=rows.reduce(function(s,r){return s+r.oi;},0);
      var totVol=rows.reduce(function(s,r){return s+(r.vol||0);},0);
      var totChg=rows.reduce(function(s,r){return s+(r.chg||0);},0);
      html+='<div style="background:'+MON_BG+';padding:5px 14px;margin-top:4px;display:flex;align-items:center;justify-content:space-between;">'
        +'<span style="font-size:13px;font-weight:700;color:'+color+';">'+month+'</span>'
        +'<span style="font-size:11px;color:'+DIM+';">OI: <span style="color:'+BLACK+';">'+totOI.toLocaleString()+'</span>'
        +'&nbsp;&nbsp;Vol: <span style="color:'+BLACK+';">'+totVol.toLocaleString()+'</span>'
        +'&nbsp;&nbsp;OI Chg: <span style="color:'+cChg(totChg)+';">'+fmtC(totChg)+'</span></span></div>';
      html+='<div style="display:grid;grid-template-columns:'+GRID+';background:'+HDR_BG+';">'
        +colHdr('STRIKE')+colHdr('OPEN INT')+colHdr('OI CHG')+colHdr('SETTLE')+colHdr('VOLUME')+'</div>';
      rows.forEach(function(r,idx){
        var bg=idx%2===0?ROW1:ROW2;
        html+='<div style="display:grid;grid-template-columns:'+GRID+';background:'+bg+';">'
          +cell(r.strike.toFixed(2),DIM)+cell(fmtN(r.oi),cOI(r.oi))+cell(fmtC(r.chg),cChg(r.chg))
          +cell(fmtS(r.settle),BLACK)+cell(fmtN(r.vol),DIM)+'</div>';
      });
      html+='<div style="display:grid;grid-template-columns:'+GRID+';background:'+MON_BG+';border-top:1px solid #c8d4e0;margin-bottom:6px;">'
        +'<div style="font-size:11px;font-weight:700;color:'+DIM+';padding:4px 8px;text-align:right;">TOTAL</div>'
        +cell(fmtN(totOI),GOLD)+cell(fmtC(totChg),cChg(totChg))+'<div></div>'+cell(fmtN(totVol),DIM)+'</div>';
    });
    return html;
  }
  /* Options convention: the CSV date IS the release date (published today), and the
     data is the PRIOR session's OI (standard OI convention). So the PNG label shows the
     release date (od.last_date), with "as of trade date X" = release - 1 business day.
     Do NOT use _oiPngHdr here — it hardcodes DATA.last_date (the FUTURES date), which is
     the wrong convention for an options-only image. */
  var optRelease = od.last_date;
  var optTrade   = _prevBday(optRelease);
  var out='<div style="display:flex;align-items:center;padding:8px 14px;border-bottom:1px solid #c8d4e0;margin-bottom:8px;">'
    +'<span style="font-size:12px;font-weight:700;letter-spacing:2px;color:'+DIM+';">'+titleTxt+'</span>'
    +'<span style="font-size:11px;color:'+DIM+';margin-left:auto;">'+optRelease+' &nbsp;·&nbsp; as of trade date '+optTrade+'</span></div>';
  out+='<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:0 14px;">';
  out+='<div>'+buildSec('CALLS',GREEN,'calls','#f0faf4')+'</div>';
  out+='<div>'+buildSec('PUTS',RED,'puts','#fef2f2')+'</div>';
  out+='</div>';
  var inner = _oiPngHdrOpt(titleTxt, optRelease, optTrade) + '<div style="background:#f0f2f5;padding:4px 0 8px;">' + out + '</div>' + _oiPngFtrOpt(optRelease);
  _oiPngRender(inner, 'OI_Options_'+comm+'_'+optRelease+'.png', 1060);
}

function clearOptHistory() {
  document.getElementById('optHResult').innerHTML = '';
  document.getElementById('optHStatus').textContent = '';
}

function _renderOptHistRows(allRows, status, result) {
  if (!allRows || !allRows.length) { status.textContent = 'No data found'; return; }
  status.textContent = allRows.length + ' rows';
  var GREEN='#22c55e', RED='#ef4444', DIM='#64748b', WHITE='#f1f5f9', DARK='#0d1520', ALT='#0a1018', GOLD='#E8C547';
  function cChg(v){ return v>0?GREEN:v<0?RED:DIM; }
  var html = '<div style="margin-bottom:12px;">';
  var groups = {};
  allRows.forEach(function(r){ if(!groups[r.sec])groups[r.sec]=[]; groups[r.sec].push(r); });
  Object.keys(groups).sort().forEach(function(sec) {
    var rows = groups[sec];
    /* separator bar between strike groups */
    html += '<div style="margin-bottom:16px;">'
          + '<div style="background:#0f1e35;padding:5px 14px 5px 10px;font-size:12px;font-weight:700;border-left:3px solid '+(rows[0].pc==='C'?GREEN:RED)+';color:'+(rows[0].pc==='C'?GREEN:RED)+';">'+sec+'</div>'
          + '<div style="display:grid;grid-template-columns:100px 1fr 1fr 1fr 1fr;background:#080f1a;">'
          + ['DATE','OPEN INT','OI CHG','SETTLE','VOLUME'].map(function(h){
              return '<div style="font-size:10px;font-weight:700;color:'+DIM+';text-align:right;padding:3px 8px;letter-spacing:.5px;">'+h+'</div>';
            }).join('')+'</div>';
    rows.forEach(function(r, idx) {
      html += '<div style="display:grid;grid-template-columns:100px 1fr 1fr 1fr 1fr;background:'+(idx%2===0?DARK:ALT)+';}">'
            + '<div style="font-size:13px;color:'+DIM+';padding:3px 8px;text-align:right;">'+r.date+'</div>'
            + '<div style="font-size:13px;font-weight:700;color:'+(r.oi>=5000?GOLD:r.oi>=1000?GREEN:WHITE)+';text-align:right;padding:3px 8px;">'+Number(r.oi).toLocaleString()+'</div>'
            + '<div style="font-size:13px;color:'+cChg(r.chg)+';text-align:right;padding:3px 8px;">'+(r.chg!==null&&r.chg!==''?(r.chg>=0?'+':'')+Number(r.chg).toLocaleString():'—')+'</div>'
            + '<div style="font-size:13px;color:'+WHITE+';text-align:right;padding:3px 8px;">'+(r.settle?Number(r.settle).toFixed(2):'—')+'</div>'
            + '<div style="font-size:13px;color:'+DIM+';text-align:right;padding:3px 8px;">'+(r.vol?Number(r.vol).toLocaleString():'—')+'</div>'
            + '</div>';
    });
    html += '</div>';
  });
  html += '</div>';
  result.innerHTML = html;
}

function runOptHistory() {
  var month  = document.getElementById('optHMonth').value;
  var pc     = document.getElementById('optHPC').value;
  var from   = document.getElementById('optHFrom').value;
  var to     = document.getElementById('optHTo').value;
  var status = document.getElementById('optHStatus');
  var result = document.getElementById('optHResult');

  /* multi-strike: read all selected options */
  var strikeEl = document.getElementById('optHStrike');
  var strikes  = Array.from(strikeEl.selectedOptions).map(function(o){ return o.value; });
  if (!strikes.length) strikes = [''];   /* no selection = all strikes */

  status.textContent = 'Loading...';
  result.innerHTML = '';

  var promises = strikes.map(function(strike) {
    return fetch('/api/options/history?month='+encodeURIComponent(month)
      +'&strike='+encodeURIComponent(strike)+'&pc='+encodeURIComponent(pc)
      +'&from='+encodeURIComponent(from)+'&to='+encodeURIComponent(to)
      +'&commodity='+encodeURIComponent(_optComm))
      .then(function(r){ return r.json(); });
  });
  Promise.all(promises)
    .then(function(results) {
      var allRows = [];
      results.forEach(function(d){ if(d&&d.rows) allRows = allRows.concat(d.rows); });
      /* stable sort: by sec then date desc */
      allRows.sort(function(a,b){ return a.sec<b.sec?-1:a.sec>b.sec?1:b.date<a.date?-1:1; });
      _renderOptHistRows(allRows, status, result);
    })
    .catch(function(e){ status.textContent = 'Error: '+e.message; });
}
</script>
</body>
</html>"""


# ── Full history API endpoint ────────────────────────────────────────────────────
@app.route('/api/options/data')
def api_options_data():
    from flask import request as freq
    comm = freq.args.get('commodity', 'CT')
    opts = load_options(comm=comm)
    return jsonify(opts)

@app.route('/api/options/history')
def api_options_history():
    from flask import request as freq
    month  = freq.args.get('month', '')
    strike = freq.args.get('strike', '')
    pc     = freq.args.get('pc', '')
    comm_filter = freq.args.get('commodity', '')
    strike_f = None
    if strike:
        try: strike_f = float(strike)
        except Exception: pass
    from_dt = freq.args.get('from', '')
    to_dt   = freq.args.get('to', '')

    def get_strike(r):
        """Get strike from strike_px column, or parse from security_des if blank."""
        sx = r.get('strike_px', '').strip()
        if sx:
            try: return float(sx)
            except Exception: pass
        # Parse from security_des e.g. 'CTN6C    90' -> 90.0
        try:
            sec = r.get('security_des', '').strip()
            # Find the number after the C/P character
            for i, ch in enumerate(sec):
                if i > 3 and ch in ('C', 'P'):
                    return float(sec[i+1:].strip())
        except Exception: pass
        return None

    def get_pc(r):
        """Get put/call from column, or parse from security_des."""
        pc_val = r.get('put_call', '').strip()
        if pc_val: return pc_val
        try:
            sec = r.get('security_des', '').strip()
            for i, ch in enumerate(sec):
                if i > 3 and ch in ('C', 'P'):
                    return ch
        except Exception: pass
        return ''

    rows = []
    if OPT_FILE.exists():
        with open(OPT_FILE, 'r', encoding='utf-8', newline='') as f:
            for r in csv.DictReader(f):
                try:
                    row_strike = get_strike(r)
                    row_pc     = get_pc(r)
                    if comm_filter and r.get('commodity','') != comm_filter: continue
                    if strike_f is not None:
                        if row_strike is None or abs(row_strike - strike_f) > 0.01: continue
                    if month and r['contract_month'] != month: continue
                    if pc and row_pc != pc: continue
                    if from_dt and r['date'] < from_dt: continue
                    if to_dt   and r['date'] > to_dt:   continue
                    rows.append({
                        'date':   r['date'],
                        'sec':    r['security_des'].strip(),
                        'pc':     row_pc,
                        'oi':     int(r['open_int'])    if r.get('open_int')   else 0,
                        'chg':    int(r['oi_chg'])      if r.get('oi_chg') and r['oi_chg'] not in ('','None') else None,
                        'settle': float(r['px_settle']) if r.get('px_settle') and r['px_settle'] != '' else None,
                        'vol':    int(r['px_volume'])   if r.get('px_volume') and r['px_volume'] not in ('','None') else 0,
                    })
                except Exception:
                    continue
    rows.sort(key=lambda x: x['date'], reverse=True)
    return jsonify({'rows': rows})


@app.route('/api/history/<comm>')
def api_history(comm):
    if not DATA_FILE.exists():
        return jsonify({}), 404
    from collections import defaultdict
    daily_agg = defaultdict(int)
    by_ticker = defaultdict(list)
    with open(DATA_FILE, 'r', encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            if r.get('commodity') == comm and r.get('open_int'):
                oi  = int(r['open_int'])
                dt  = r['date']
                tk  = r['bbg_ticker']
                daily_agg[dt] += oi
                by_ticker[tk].append({'date': dt, 'open_int': oi, 'contract': r.get('contract','')})
    sd = sorted(daily_agg.keys())
    th = {'Aggregate': [{'date': d, 'open_int': daily_agg[d]} for d in sd]}
    for tk, hist in by_ticker.items():
        th[tk.replace(' Comdty', '')] = sorted(hist, key=lambda x: x['date'])
    return jsonify({'daily_agg': {d: daily_agg[d] for d in sd}, 'ticker_history': th})


# ── Routes ───────────────────────────────────────────────────────────────────────
@app.route('/debug')
def debug():
    data = load_data()
    opts = load_options(comm='CT')
    if not data:
        return 'No data', 503
    out = {}
    for comm, cd in data['commodities'].items():
        out[comm] = {
            'agg_oi': cd['agg_oi'], 'lo5': cd['lo5'], 'hi5': cd['hi5'],
            'lo15': cd['lo15'], 'hi15': cd['hi15'],
            'sparkline_len': len(cd['sparkline']),
            'max_years': cd['max_years'],
        }
    return jsonify(out)

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'version': __version__, 'data_exists': DATA_FILE.exists()})

def load_options(comm='CT'):
    """Load options_oi.csv, return structured dict for the Options tab."""
    if not OPT_FILE.exists():
        return {}
    MC_SORT = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    def mo_key(m):
        try:
            mo, yr = m.split()
            return (int(yr), MC_SORT.index(mo))
        except Exception:
            return (9999, 0)
    rows = []
    try:
        with open(OPT_FILE, 'r', encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return {}
    if not rows: return {}
    # Filter to requested commodity
    rows = [r for r in rows if r.get('commodity', 'CT') == comm]
    if not rows: return {}
    last_date = max(r['date'] for r in rows)
    today_rows = [r for r in rows if r['date'] == last_date]
    # Build prev day OI for oi_chg display
    dates = sorted({r['date'] for r in rows})
    prev_date = dates[-2] if len(dates) >= 2 else None
    prev_oi = {}
    if prev_date:
        for r in rows:
            if r['date'] == prev_date and r.get('open_int'):
                prev_oi[r['security_des']] = int(r['open_int'])
    # Rebuild oi_chg from stored value (already computed) or fallback
    months = sorted({r['contract_month'] for r in today_rows}, key=mo_key)
    def parse_strike(r):
        sx = r.get('strike_px', '').strip()
        if sx:
            try: return float(sx)
            except Exception: pass
        try:
            sec = r.get('security_des', '').strip()
            for i, ch in enumerate(sec):
                if i > 3 and ch in ('C', 'P'):
                    return float(sec[i+1:].strip())
        except Exception: pass
        return 0.0

    def gnum(r, field):
        """Parse a stored Black-76 Greek/IV; None when the strike had no solve."""
        v = r.get(field, '')
        v = v.strip() if isinstance(v, str) else v
        if v in ('', 'None', None):
            return None
        try: return float(v)
        except (TypeError, ValueError): return None

    result = {'last_date': last_date, 'months': months, 'calls': {}, 'puts': {}}
    for m in months:
        for pc, key in (('C','calls'), ('P','puts')):
            if m not in result[key]:
                result[key][m] = []
            for r in today_rows:
                if r['contract_month'] == m and r['put_call'] == pc:
                    oi     = int(r['open_int'])   if r.get('open_int')   else 0
                    chg    = int(r['oi_chg'])      if r.get('oi_chg') and r['oi_chg'] not in ('','None') else None
                    settle = float(r['px_settle']) if r.get('px_settle') and r['px_settle'] != '' else None
                    vol    = int(r['px_volume'])   if r.get('px_volume') and r['px_volume'] not in ('','None') else 0
                    result[key][m].append({
                        'sec':    r['security_des'].strip(),
                        'strike': parse_strike(r),
                        'oi':     oi,
                        'chg':    chg,
                        'settle': settle,
                        'vol':    vol,
                        # Black-76 IV/Greeks (computed upstream at F_parity + base SOFR).
                        # Additive API fields; None where the strike had no IV solve.
                        'iv':     gnum(r, 'iv_pct'),
                        'delta':  gnum(r, 'delta'),
                        'gamma':  gnum(r, 'gamma'),
                        'vega':   gnum(r, 'vega'),
                        'theta':  gnum(r, 'theta'),
                    })
            result[key][m].sort(key=lambda x: x['strike'])
    return result


@app.route('/')
def index():
    data = load_data()
    if data is None:
        return ('<body style="background:#080b0f;color:#f8fafc;font-family:monospace;padding:40px;">'
                '<h2>oi_data.csv not found — run oi_bootstrap.py first.</h2></body>'), 503
    opts = load_options(comm='CT')
    css  = load_css()
    html = HTML.replace('%%CSS%%',       css)\
               .replace('%%DATA%%',      json.dumps(data).replace('</', '<\\/'))\
               .replace('%%OPTDATA%%',   json.dumps(opts).replace('</', '<\\/'))\
               .replace('%%VERSION%%',   __version__)
    return html

@app.route('/api/data', methods=['GET'])
def api_data():
    try:
        rows = []
        with open(DATA_FILE, 'r', encoding='utf-8', newline='') as f:
            for r in csv.DictReader(f):
                rows.append(dict(r))
        as_of = max((r['date'] for r in rows), default='unknown')
        resp = jsonify({'source': 'VLM Commodities — Open Interest Dashboard', 'as_of': as_of, 'data': rows})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
    except Exception as e:
        resp = jsonify({'error': str(e)})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp, 500
@app.route('/api/latest', methods=['GET'])
def api_latest():
    try:
        rows = []
        with open(DATA_FILE, 'r', encoding='utf-8', newline='') as f:
            for r in csv.DictReader(f):
                rows.append(dict(r))
        # Most recent date only
        if rows:
            last_date = max(r['date'] for r in rows)
            rows = [r for r in rows if r['date'] == last_date]
        as_of = last_date if rows else 'unknown'
        resp = jsonify({'source': 'VLM Commodities — Open Interest Dashboard', 'as_of': as_of, 'data': rows})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
    except Exception as e:
        resp = jsonify({'error': str(e)})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp, 500               
if __name__ == "__main__":
    print(f"VLM OI Monitor v{__version__} — http://127.0.0.1:8052")
    print(f"Data: {DATA_FILE}  exists={DATA_FILE.exists()}")
    app.run(debug=False, host="0.0.0.0", port=8052)
