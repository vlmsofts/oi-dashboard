# VLM Open Interest Dashboard — Full Handoff Document v2
**As of: May 1, 2026**

---

## 1. SYSTEM OVERVIEW

Live Bloomberg-powered dashboard displaying futures and options open interest for Cotton, Sugar, Coffee, and Cocoa.

**Live URL:** `dashboard.openinterest.vlmdata.com`
**Repo:** `vlmsofts/oi-dashboard`
**Hosting:** Railway (auto-deploys on every GitHub push)
**DNS:** IONOS → CNAME → Railway (Cloudflare proxy OFF)

---

## 2. FILE LOCATIONS

```
Desktop\Open interest dashboard\
├── app.py                        — Flask dashboard (Monitor/Seasonal/Table/Options tabs)
├── build_whatsapp_oi.py          — WhatsApp PNG generator (4 commodities)
├── oi_bootstrap.py               — One-time futures OI backfill
├── options_bootstrap.py          — One-time options OI backfill (run when needed)
├── migrate_options_commodity.py  — One-time migration (already run Apr 2026)
├── test_volume.py                — Bloomberg volume diagnostic (keep for reference)
├── test_opt_chain_all.py         — Bloomberg option chain diagnostic (keep for reference)
├── Procfile                      — gunicorn startup for Railway
├── railway.json                  — Railway build config (NIXPACKS)
└── data\
    ├── oi_data.csv               — Futures OI: 38 tickers × daily (2008–present)
    └── options_oi.csv            — Options OI: CT/KC/CC/SB daily (Jan 2025–present)

Desktop\VLM Data\
├── vlm_master_fetch.py           — Master daily fetcher (Jobs 1+2+3)
└── vlm_master_fetch.log          — Daily run log
```

---

## 3. DAILY AUTOMATION — CONFIRMED WORKING

**Windows Task Scheduler fires at 9:35 AM EST Mon-Fri:**
- Program: `C:\Users\Louis\AppData\Local\Programs\Python\Python314\pythonw.exe`
- Script: `vlm_master_fetch.py` in `Desktop\VLM Data\`
- Bloomberg Terminal must be open and logged in

### JOB 1 — Crop Dashboard Prices
- Pulls CT/Corn/Soy settle prices
- Patches `crop_app.py` → git push → Railway redeploys crop dashboard

### JOB 2 — Futures OI (38 tickers)
- Month-specific generics: CTMAR1/2, CTMAY1/2, CTJUL1/2, CTOCT1/2, CTDEC1/2 (CT)
- Same pattern for SB (4 months × 2), KC (5 months × 2), CC (5 months × 2)
- **NO numbered slots** (CT1/CT2 etc) — these were removed April 2026 (caused roll distortion)
- Second Bloomberg request pulls FND/LTD from actual underlying contracts
- Appends 38 rows to `oi_data.csv` → git push → Railway redeploys
- Dedup guard: skips if today already exists

### JOB 3 — Options OI (CT + KC + CC + SB)
- Step 1: `OPT_FUTURES_CHAIN_DATES` on CT1/KC1/CC1/SB1 → underlying contracts
  - CT: 14 contracts, KC: 9, CC: 8, SB: 11 = 42 total
- Step 2: `OPT_CHAIN` on each underlying → ~8,900 option tickers
  - **KEY:** `OPT_CHAIN` works with no override. `OPT_CHAIN_EXPIRE_DT` requires
    `EXPIRATION_DATE_OVERRIDE` or returns 0 tickers silently — do not use
- Step 3: Pull `OPEN_INT`, `OPEN_INT_CHANGE`, `PX_SETTLE`, `PREVIOUS_TOTAL_VOLUME`,
  `OPT_STRIKE_PX`, `OPT_PUT_CALL`, `SECURITY_DES` in batches of 100
- Appends ~1,865 rows to `options_oi.csv` → git push
- Dedup guard: skips if today already exists
- Takes ~90 seconds total
- Commodity stored in `commodity` column (CT/KC/CC/SB)

---

## 4. DASHBOARD TABS

### Monitor Tab
- 4 commodities: CT, SB, KC, CC
- Columns: Ticker | Fut Cont | Open Int | OI Chg | Settle Px | Aggte OI | Aggte OI Chg | Sparkline | 5yr Lo | 5yr Hi | 15yr Lo | 15yr Hi | 1st Notice
- FND dates from actual underlying contracts via second Bloomberg request

### Seasonal Tab
- HI/AVG/LO band + Individual Years mode
- 1–18 year slider
- Each commodity uses its own matching contract (SBMAY1, KCMAY1 etc)

### Table Tab
- Historical OI with date range, DAILY/WEEKLY/MONTHLY aggregation

### Options Tab
- Side-by-side CALLS / PUTS for Cotton only (pending: add KC/CC/SB selector)
- Grouped by contract month with subtotals
- History Search: strike dropdown, month, C/P, date range → `/api/options/history`
- Strike parsed from `security_des` when `strike_px` column blank (bootstrap rows)
- **PENDING:** Add commodity selector to Options tab for KC/CC/SB

---

## 5. OPTIONS DATA — CSV STRUCTURE

```
options_oi.csv columns:
date, commodity, security_des, contract_month, put_call,
strike_px, open_int, oi_chg, px_settle, px_volume
```

**Notes:**
- `commodity` column added May 2026 via `migrate_options_commodity.py`
- `strike_px` is blank for bootstrap rows (strike embedded in `security_des`)
- `px_volume` blank for dates before Apr 2026 (Bloomberg doesn't store historical options volume via BDH)
- Volume from Apr 2026 forward is accurate (`PREVIOUS_TOTAL_VOLUME` field)
- `OPEN_INT_CHANGE` calculated by Bloomberg directly — no manual diff needed

---

## 6. WHATSAPP IMAGE GENERATOR

**Script:** `build_whatsapp_oi.py`
**Output:** `output\whatsapp\YYYY-MM-DD\OI_Monitor_{CT/KC/CC/SB}_YYYY-MM-DD.png`
**Bat file:** `WHATSAPP O I.bat` on Desktop — double-click → generates → opens folder

### What Each Image Contains
**Section 1 — Futures OI** (sorted by first notice date — correct crop year order):
Ticker | Fut Cont | Open Int | OI Chg | Settle Px | Aggte O.I. | Aggte Chg | 1st Notice

**Section 2 — Top 10 Options OI Changes:**
Ticker | Month | P/C | Open Int | OI Chg | Settle | Volume

### 4 Images Generated
- `OI_Monitor_CT_YYYY-MM-DD.png` — Cotton
- `OI_Monitor_KC_YYYY-MM-DD.png` — Coffee
- `OI_Monitor_CC_YYYY-MM-DD.png` — Cocoa
- `OI_Monitor_SB_YYYY-MM-DD.png` — Sugar

### Data Source
Reads from local CSVs only — no Bloomberg needed. Run any time after 9:35 AM.

### PENDING: Website Post Functionality
Previously the COT dashboard bat posted images to a website. This needs to be
added back. Two options when ready:
- **Option A:** Post single Cotton image only (same as before)
- **Option B:** Post all 4 commodity images separately

When decided, add back the `requests.post()` call at the end of `build_whatsapp_oi.py`
after the `render_png()` calls.

---

## 7. KEY TECHNICAL DECISIONS & FIXES

### Apr 2026 — Numbered Slots Removed
CT1/CT2/SB1/SB2 etc were in the fetcher and CSV. These cause roll distortion
(OI appears to change by 130K+ on roll day). Replaced with month-specific generics
(CTJUL1 always = front July regardless of queue). Old rows cleaned from CSV with
month-name filter (keeps any ticker containing MAR/MAY/JUL/OCT/DEC/SEP).

### Apr 2026 — Options Bootstrap
Historical backfill Jan 2025 → Apr 2026 via BDH. Key learnings:
- `VOLUME` field + `periodicityAdjustment=ACTUAL` returns historical daily volume
- `PREVIOUS_TOTAL_VOLUME` is real-time only — not historizable
- Historical volume only available from ~Apr 2026 forward reliably

### Apr 2026 — OPT_CHAIN Fix
`OPT_CHAIN_EXPIRE_DT` silently returns 0 tickers without `EXPIRATION_DATE_OVERRIDE`.
`OPT_CHAIN` works with no override. Confirmed via field search on Bloomberg terminal.

### May 2026 — KC/CC/SB Options Added
`OPT_FUTURES_CHAIN_DATES` confirmed working on KC1/CC1/SB1 Comdty.
`commodity` column added to `options_oi.csv` via migration script.
Job 3 now pulls ~8,900 tickers across all 4 commodities in ~90 seconds.

### May 2026 — Crop Year Sort Fix
WhatsApp image was showing CTMAR1 (Mar 2027) before CTDEC1 (Dec 2026).
Fixed by sorting futures rows by `first_notice` date instead of ticker label order.

---

## 8. KNOWN ISSUES & POTENTIAL PROBLEMS

### Git Conflicts — Crop Dashboard
GitHub Actions also commits to crop-dashboard repo. Occasional push rejection.
Fix:
```powershell
cd "C:\Users\Louis\OneDrive - VLM Commodities LTD\Desktop\COT"
git pull --rebase
git push
```

### Partial Day Data
If fetcher runs but Bloomberg only returns some tickers (happened May 1 2026 —
only 17 tickers from old fetcher version), today's rows in CSV will be incomplete.
Fix: delete today's rows and rerun:
```powershell
python -c "
import csv, pathlib, shutil
f = pathlib.Path(r'data\oi_data.csv')
rows = list(csv.DictReader(f.open(encoding='utf-8')))
shutil.copy(f, f.with_suffix('.bak'))
clean = [r for r in rows if r['date'] != 'YYYY-MM-DD']
with open(f, 'w', newline='', encoding='utf-8') as out:
    w = csv.DictWriter(out, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(clean)
print('Done')
"
```
Same pattern works for `options_oi.csv`.

### Wrong Fetcher Version
On Apr 30 the correct fetcher was overwritten with an old simplified version
(missing "2" contracts and Job 3). Backup saved as:
`vlm_master_fetch good one b4 options.py` in Desktop\VLM Data\
Always keep a dated backup before replacing.

### Bloomberg Session Timeout
Job 3 takes ~90 seconds. If Bloomberg times out mid-run, options CSV gets no data
for that day. Check log for `ERROR` in Job 3. Manual fix: delete today's options
rows and rerun fetcher.

### Options CSV Growth
Currently ~1,865 rows/day × 252 trading days = ~470K rows/year.
At this rate the CSV scan becomes slow in 2–3 years.
Migrate `/api/options/history` to Supabase when file exceeds 20K rows/request latency.

---

## 9. PENDING WORK

### HIGH PRIORITY
1. **Options tab commodity selector** — Add KC/CC/SB tabs to Options tab in `app.py`.
   Data is already in `options_oi.csv` with `commodity` column.
   The `load_options()` function needs a `comm` parameter.
   The `buildOptions()` JS function needs commodity selector buttons.
   The `/api/options/history` endpoint needs `&commodity=KC` filter parameter.

2. **WhatsApp image website post** — Add back post functionality to bat file.
   Decide: single Cotton image or all 4 commodities.

### MEDIUM PRIORITY
3. **WASDE Dashboard fixes** — Still pending from earlier sessions:
   - Invisible topbar text (CSS only)
   - Truncated narrative summary
   - Broken release calendar with countdown
   - May 12: manual download of `wasde0526.xls` into WASDE DASHBOARD folder

4. **WASDE Email/WhatsApp auto-draft** — Pending after dashboard fixes

5. **Rain Dashboard** — `rain_scraper.py` complete. Next step:
   `python rain_scraper.py --history` to populate `rain_history.csv`
   Then wire station-level precip into `build_features.py`

### LOW PRIORITY
6. **Options historical backfill KC/CC/SB** — Run `options_bootstrap.py` for
   KC/CC/SB going back to Jan 2025. Cotton already done. Will take ~30 minutes.

7. **Options volume pre-Apr 2026** — Bloomberg doesn't store historical options
   volume via BDH reliably. Pre-Apr 2026 volume shows `—`. Building daily record
   going forward is the only solution.

---

## 10. BOOTSTRAP SCRIPTS

### oi_bootstrap.py — Futures (run if CSV corrupted)
```powershell
python oi_bootstrap.py --start 2008-01-01
```

### options_bootstrap.py — Options (run to extend history or recover data)
```powershell
python options_bootstrap.py --start 2025-01-01
```
Uses `VOLUME` field + `periodicityAdjustment=ACTUAL`. Updates volume on existing rows.
Takes ~10 minutes for Cotton (2,140 tickers). Multiply by 4 for all commodities.

---

## 11. DEPLOYMENT

- **Procfile:** `web: gunicorn app:server --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
- **railway.json:** NIXPACKS builder, ON_FAILURE restart
- Auto-redeploys 1-2 min after git push

---

## 12. DATA STORAGE OUTLOOK

| File | Current Size | Growth/Year | GitHub Limit Hit |
|------|-------------|-------------|-----------------|
| oi_data.csv | ~180K rows | ~10K rows | ~8 years |
| options_oi.csv | ~35K rows (CT only) | ~470K rows (all 4) | ~3 years |

**Action needed in ~2 years:** Migrate `options_oi.csv` to Supabase.
The `/api/options/history` endpoint is the only full-scan — one-function swap.

---

## 13. CONFIRMED DAILY PROCESS

```
9:35 AM EST Mon-Fri (Bloomberg open)
         ↓
vlm_master_fetch.py
         ↓
Job 1: CT/Corn/Soy prices → crop_app.py → git push → crop dashboard live
         ↓
Job 2: 38 futures tickers → oi_data.csv → git push → OI dashboard live
         ↓
Job 3: 4 commodities → ~8,900 option tickers → ~1,865 rows → options_oi.csv → git push
         ↓
Railway redeploys within 1-2 minutes
         ↓
Run WHATSAPP O I.bat any time after 9:35 → 4 PNGs generated → folder opens
```

*VLM Commodities LTD — Internal Use Only — May 2026*
