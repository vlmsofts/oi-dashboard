# Open interest dashboard — MEMORY

## 2026-08-04 — WhatsApp auto-send was never scheduled + freshness guard added

**What happened:** Lou double-clicked `OPEN INTEREST.bat` manually and WhatsApp images sent
fine (as always) — but the *unattended* run he expected didn't happen. Root cause: only the
data pipeline (`vlm master fetch`, Task Scheduler, daily 09:30, `VLM Data/vlm_master_fetch.py`)
was ever automated. `build_whatsapp_oi.py` (WhatsApp PNGs + site post) had **zero** scheduled
task — `OPEN INTEREST.bat`/`Run_OI_Update.bat` on the Desktop were always manual-only Explorer
shortcuts. Commit b53c902 (Aug 3) made the *manual* run auto-send instead of y/N-prompting;
it never added scheduling. `gh run list` confirmed zero GitHub Actions runs ever (not the
mechanism) — all "auto:" commits are local Task Scheduler + local git identity.

**Fix (Lou-approved):**
1. New Task Scheduler task `VLM OI WhatsApp AutoSend` — Mon–Fri 09:35 EST (5 min after
   `vlm master fetch` finishes, confirmed by today's commit timestamps ending 09:33:19),
   runs `build_whatsapp_oi.py` via `pythonw.exe`, mirrors `vlm master fetch`'s action pattern.
2. **Freshness guard** added: `check_freshness()` in `build_whatsapp_oi.py`. Two checks, both
   must pass or the SEND (not image generation) is skipped:
   - `oi_data.csv` mtime must be today's calendar date — proves master fetch actually touched
     it today (catches a failed/late fetch still serving yesterday's row). NOTE: cannot compare
     `trade_date == today` directly — OI is legitimately T+1, so a healthy 09:35 run always
     shows yesterday's completed session; mtime is what proves freshness, not the date value.
   - `.sent_<date>` marker file in the dated output folder — blocks a duplicate send if the
     task ever double-fires. Written only on full send success (not on partial/total failure,
     so a real Twilio/site outage stays retryable).
   - `--force` flag bypasses both for manual override. Manual re-runs always regenerate PNGs
     regardless of guard state — only the send step is gated.

**Verify:** guard can't be proven live until the next scheduled fire (08-05 09:35) — check
`Get-ScheduledTaskInfo -TaskName "VLM OI WhatsApp AutoSend"` for `LastTaskResult: 0` and that
WhatsApp images actually landed.

## 2026-07-19 — Seasonal PNG mirrors on-screen layout (merge 3dc3aca)

Lou reported: exporting the Seasonal PNG while GRID was selected produced the spaghetti
chart, not the grid. Root cause = my own deferral: `exportSeasPng` always rendered the
single multi-line card (I'd left "static grid PNG is a later enhancement" in a comment).
Lou's rule: **whatever layout is chosen on the site IS the PNG default.**

**Fix:** `exportSeasPng` now routes on `seasLayout`/`seasMode` (the same on-screen state):
- GRID → renders one light-mode small-multiple panel per prior year (gold current-year +
  blue prior-year line, shared y-scale via `seasYRange`) laid out 5-up — mirrors
  `buildSeasGrid()`. Title label → "INDIVIDUAL YRS · GRID". Wider canvas (1500px).
- BAND / SPAGHETTI → unchanged single-card render.
- `lightChartImg()` gained optional `{yMin,yMax,fontSize,maxYTicks,maxXTicks}` so grid
  panels share the y-scale; existing single-card calls pass no opt (defaults = old behavior).

**Audit (Sonnet) clean** on fidelity/back-compat/scope/promise/no-regression. Caught + fixed:
prior-year line was #1e6fd4 in PNG vs screen's #5ba3e8 → matched for true parity; removed
dead `fmtSm`. Both branches write `OI_Seasonal_<date>.png`.

## 2026-07-19 — Monitor B3 OI-vs-price conviction tag (merge 514b16a)

Shipped the previously-deferred B3. Lou challenged the deferral ("don't we have this data?")
and was RIGHT — the settle day-change was derivable from data ALREADY loaded, not a producer
change as I first claimed. Corrected: `load_data` builds a per-ticker `history` list of
{date,open_int,oi_chg,settle}; the prior settle is right there. But two things had hidden it
from the row: the main payload drops `history` per ticker, and `/api/history` only carries
{date,open_int} (no settle). Fix = compute it server-side where history is already in hand.

**What:** `load_data` now computes `settle_chg` per ticker = settle[-1]-settle[-2] over the two
latest PRICED history rows (None if <2 priced sessions). Client `convictionTag(oiChg,settleChg)`
renders a small tag INSIDE the existing OI CHG cell (grid stays 11 cols): NL new longs (OI↑px↑,
grn) / NS new shorts (OI↑px↓, red) / SC short covering (OI↓px↑, grn) / LL long liquidation
(OI↓px↓, red), with a descriptive tooltip. Empty when oiChg or settleChg is 0/null/undefined/NaN.

**Additive-only:** `settle_chg` is a new ticker key; the payload whitelists-by-exclusion (drops
only `history`), so new keys pass through — no route/shape change. Sonnet audit clean 6/6
(quadrants, guards airtight incl. +null=0 caught by ===0, grid=11, blast-radius additive,
tooltip no-conflict, OI CHG numeric+color unchanged). Verified real CTDEC1 (OI -2115, px -2.25)
reads LL.

**Lesson:** "the field isn't on the row" ≠ "the data doesn't exist." Check whether it's
derivable from what load_data already holds before calling something a producer/CSV change.

## 2026-07-19 — Seasonal SPAGHETTI|GRID redesign + Monitor B1/B2/B4 (merge ffcae02)

**What (all in app.py's INDEX_HTML template — display-only, zero data-layer change):**
Seasonal tab: dropped the STACKED all-commodities view + the `seasView` var entirely; the
tab is now ALWAYS single-commodity via an always-on dropdown. Added a `SPAGHETTI|GRID` layout
toggle (`seasLayout` state, `setSeasLayout`), shown only in INDIVIDUAL-YEARS mode (hidden for
HI/AVG/LO band). SPAGHETTI = multi-line chart + gold crosshair (`seasCrosshair` plugin) + a
hover "rail" (`attachSeasRail`) reading every year's OI at the hovered month. GRID =
small-multiples (`buildSeasGrid`), one panel per prior year, shared y-scale, synced hover.
`buildSeasonal` rewritten to a 3-way route (band card / spaghetti / grid). Band mode +
`computeIndividual`/`getSeasHist`/`computeBand`/`buildSeasCard` UNCHANGED. PNG export
collapsed to single-commodity; filename `OI_Seasonal_Single_*` → unified `OI_Seasonal_*`.

Monitor tab: **B1** child (expanded) rows now show per-tenor share-of-aggregate
(`shareCell` = open_int/agg_oi %, tiny inline bar) instead of the repeated aggregate figure;
parent row keeps the real aggregate. **B2** merged the 5yr Hi/Lo + 15yr Hi/Lo columns into
ONE range bar (`rangeBar`: 15yr faint track, 5yr band, gold current marker); exact numbers to
hover tooltip. `.G` grid 12→11 cols; header/parent/child rows all realigned to 11 cells.
**B4** faint current-value label at the 1yr sparkline right endpoint.

**Deferred (with reasons):**
- **B3** (OI×price conviction: new-longs/new-shorts/short-cover/long-liq glyph) — BLOCKED:
  needs a per-row settle DAY-CHANGE sign, which does not exist in the Monitor row data today
  (rows carry settle LEVEL + oi_chg only). That's a producer/data-layer change → out of scope.
- **A5** (Playwright PNG port) — de-scoped: the spec assumed the export screenshots the DARK
  DOM, but `_oiPngRender` already injects a full LIGHT-palette token set into an offscreen
  clone, so palette is already correct. The port is fidelity-only (scale:2 html2canvas vs 3×),
  a later nice-to-have, not required.
- Spec's B4 `tension:0` was a non-issue: the sparkline is hand-drawn SVG polyline, not Chart.js.

**Process:** built on branch `feat/seasonal-redesign-monitor-refine`, smoke-tested (app boots,
`/`→200, APIs 200, JS brackets balanced), then 3-agent Sonnet audit — correctness (grid cells
all 11, division/NaN guards solid, chart lifecycle clean, no off-by-one hover), no-regression
(all 7 categories clean), blast-radius (ZERO — `build_whatsapp_oi.py` reads CSVs + renders its
OWN html, never scrapes this dashboard; no API/CSV/route touched). Merged to main ffcae02.
Stale doc `options_dashboard_DOCS/VLM_OI_Dashboard_Handoff_v2.md` still lists the old columns.

## 2026-07-18 — oi_data.csv historical field backfill (OHLC + FND/LTD), commit b008ed3

**What:** filled previously-EMPTY `high`/`low`/`open` + `first_notice`/`last_trade` across
the full 2008→2026 history of `data/oi_data.csv` (5 columns, 752,480 cells). FILL-EMPTY-ONLY:
0 existing cells overwritten, row count + keys identical (178,817 rows). volume/open_int
were already as full as BBG serves — 0 filled. Builder: `VLM Data/backfill_oi_fields.py`.
Backup: `data/oi_data.backup_pre_fieldfill_2026-07-18.csv` (gitignored `*.backup_*`).

**FND/LTD method (no formula):** BBG serves `FUT_CUR_GEN_TICKER` as a HISTORICAL per-date
field on the generic — CTMAR1 on 2008-01-02 → 'CTH08'. So resolve the dated contract the
generic pointed at PER DATE, then pull that contract's real `FUT_NOTICE_FIRST`/
`LAST_TRADEABLE_DT`. Exchange-authoritative for all 4 commodities. A cotton FDD-5bd/LDD-10bd
formula is 15/15 on FND but only 13/15 on LTD (fails March) and is flat WRONG for softs
(KC LTD ~1mo after FND; SB LTD BEFORE FND, cash-settle) — so formula is diagnostic-only,
never writes. Weekday-holiday rows (BBG republishes prior OI with NO pointer) inherit the
prior trading session's dated contract, guarded to ≤4 calendar days (`prior_dated`,
`MAX_INHERIT_GAP_DAYS`). Real max gap = 3d, 0 roll crossings, guard never fired.

**Verified (3-agent Sonnet audit + Haiku scan):** CT FND/LTD matches ICE ProductSpec ref
CSV exactly; 0 high<low; softs LTD<FND correct; corrected CSV byte-identical across two
runs (md5 df073d1b).

**BLAST RADIUS — Lou-approved as a CORRECTION, not additive:** `CTA MONITOR/cta_scraper.py`
`build_front_month` used `infer_expiry()` ("15th of month") when FND/LTD were blank →
rolled LATE, holding contracts INTO their notice period. Real dates fix this. Front-month
selection changed on CT 289 / SB 75 / KC 596 / CC 741 days (2-16%) — all cases where the
OLD pick was wrong (held a contract past FND). Proven on samples (2008-04-14 CT: old held
CTMAY1 at FND-10d, new correctly rolls to CTJUL1). CTA `*_prices.csv` regenerated locally
(they are gitignored build artifacts — `data/*_prices.csv` — NOT committed). Other CTA
consumers (app.py/snapshot.py/build_whatsapp_oi.py sort by `first_notice or '9999'`) only
ever render TODAY's row, already populated by the daily job → cosmetically unaffected.

**Go-forward:** the daily job (`vlm_master_fetch.py`) already writes these 5 fields for new
rows, so this backfill is a one-off closing the historical gap; no producer change needed.

## 🔴 ICE EXPIRY AUTHORITY (Lou 2026-07-16) — the overarching truth for ALL expiry
ICE's own /expiry pages are the UNDISPUTED authority for every product's expiry/FTD/LTD.
Any stored date differing from ICE = OUR data is wrong. The 8 sources:
CT fut /products/254 · opt /products/1027 · CC fut /products/7 · opt /products/8 ·
KC fut /products/15 · opt /products/14 · SB fut /products/23 · opt /products/22
(all `ice.com/products/{id}/{Name}/{Futures|Options}/expiry`).
Served at **`vlmapi.vlmdata.com/v1/expiry/{CT|CC|KC|SB}/{futures|options}`**, refreshed
monthly. THIS REPO'S angle: `contract_expiries.json` (manual annual snapshot →
`get_opt_exp`/`contract_dates.py` → `options_oi.csv` expire_dt/days_to_exp) is the
drift-prone root the endpoint is meant to REPLACE — migrate `_build_calendar_from_json`
onto the endpoint behind the stable getter interface (get_fnd/get_ltd/get_opt_exp). MUST
preserve serial-month codes (F/U/X for CT — the fix at contract_dates.py serial-month
patch). NOT yet wired — additive migration, its own change.

## 2026-07-15 — Futures/options date-convention desync (off-by-one join + phantom holiday row)

**Defect class:** `oi_data.csv` (futures) is TRADE-date stamped; `options_oi.csv` is
RELEASE-date stamped (= trade date + 1 business day). Any code that joins/compares the two,
or assumes both share the same "latest date", is off-by-one. Became live when the prior
session made futures trade-date-stamped (they used to coincide by luck).

**Three surfaces fixed:**
1. **`build_whatsapp_oi.py` (commit def6f5d):** the client PNG joined options on the futures
   TRADE date → showed the PRIOR session's options next to current futures (missed the day's
   biggest OI moves — Lou caught the missing 14-Jul CTZ6 90/80 call flow). Fix: `_next_bday(trade_date)`
   targets the options RELEASE row; applied to PNG path + site-publish path; loud STALE fallback.
   Verified vs ICE DMR + WebICE blotter to the digit.
2. **`app.py` (commit bade6c8):** `exportOptionsPng` stamped the options PNG with `DATA.last_date`
   (FUTURES date) in banner+filename while the section header used the options release date =
   two dates on one image. Fix: options PNG + on-page options tab show RELEASE date as headline
   + "as of trade date X" subtext (Lou's convention). Added `_prevBday`/`_oiPngHdrOpt`/`_oiPngFtrOpt`;
   shared `_oiPngHdr/_oiPngFtr` left alone (other 3 exports correct). Live web-dashboard main page
   was NOT buggy (futures/options in separate tabs with separate stamps).
3. **`VLM Data/vlm_master_fetch.py` (commit df6fe31, LOCAL — VLM Data has NO git remote):** ROOT
   cause. Options job stamped rows with run calendar date (`today_str`) + computed Black-76 `trade_dt`
   via blind `today - 1 weekday`. Neither holiday-aware. Futures never had this (trade date from
   real Bloomberg session via `fetch_prior_session_finals`, filters holiday zero-bars). Fix:
   `run_options_append(session, raw, today_str, oi_trade_date)` — derives `release_date =
   oi_trade_date + 1 bday`, `trade_dt = oi_trade_date`, dedup keys on release_date, refuses to
   write if oi_trade_date is None.

**Phantom cleaned (in bade6c8):** Fri 2026-07-03 = July 4 holiday; BBG published the 07-02
session stamped 07-03 (LEGIT). Mon 07-06 (real trading day) ran with NO holiday guard and
RE-published the same 07-02 data stamped 07-06 — byte-identical dup, no futures session maps
to it. Removed the 1,930 rows dated **07-06** (kept legit 07-03). Backup
`data/options_oi.backup_pre_0706clean_2026-07-15.csv` (gitignored). 136,053→134,123 lines.

**Blast radius:** `data/options_oi.csv` is a shared contract (gateway + WhatsApp + backfill +
web dashboard). Cleanup was gated (Lou sign-off, backup, verified only 07-06 removed, schema
17 cols, neighbors intact). Verified: full desync sweep found 07-06 was the ONLY true phantom;
05-26/06-22 "orphans" are real holiday-boundary sessions (Memorial Day / Juneteenth), not dups.

## 2026-07-07 — Serial-month options got no expiry/IV (contract_dates.py structural fix)

**Defect:** `options_oi.csv` left `expire_dt`/`days_to_exp`/`iv_pct` BLANK for every
serial-month option tenor (CT: Jan=F, Sep=U, Nov=X; CC/KC: Q,V; SB: F,Q,U) across ALL
FOUR commodities — settle prices present, but no Black-76 IV. On 07-06: CT 162, KC 139,
CC 129, SB 169 blank-IV rows traced to this cause.

**Root cause (structural, not cotton-specific):** `contract_dates._build_calendar_from_json`
built `_D` by walking the FUTURES records and attaching a matching option. Serial months
have NO listed future, so they never got a `_D` entry and `get_opt_exp()` returned None →
no DTE → the daily job (`VLM Data/vlm_master_fetch.py`, which reads `get_opt_exp(security_des[:4])`)
skipped IV/greeks. The serial expiries were already IN `contract_expiries.json` as OPTION
records (e.g. CTU26 OPT_LTD 2026-08-21) — just never read for future-less codes.

**Fix (this repo, `contract_dates.py`):** after the futures walk, add any option-only code
(not already in `_D`) as an OPTION-ONLY entry {fnd:None, ltd:None, opt_exp:OPT_LTD}. Plus a
completeness guard that warns at build if a listed option has no parseable OPT_LTD.
- ADDITIVE: only fills codes absent from `_D` → cannot alter any quarterly.
- **Blast radius verified ZERO:** `_D` 46→64 (+18 serials); BBG generic↔ICE slot maps
  (`_BBG_TO_ICE`/`ice_to_bbg`) byte-identical (0 diffs) — no consumer generic requests a
  serial token, so serials stay out of slot resolution; only surface via direct get_opt_exp.
- **End-to-end proof:** get_opt_exp('CTU6')=2026-08-21; real Black-76 → Sep 2026 ATM IV
  19.51% (parity-clean, DTE 46, F_parity 77.12) where it was blank. This gives the file its
  first genuine ~30-day cotton ATM IV (Oct=70DTE is house-excluded; Dec=133DTE too long).

**Shared contract:** `contract_dates.py` is imported by BOTH this dashboard AND the daily
job. Lou approved the permanent fix 2026-07-07.

**Not fixed by this (logged, separate, tiny):**
- 2 SBX6 (Nov sugar) blank rows in options_oi.csv — SB Nov is NOT a listed ICE option
  (SB serials are F/Q/U), yet the Bloomberg feed returned settle/OI for it. Feed edge case,
  not an expiry-calendar bug. Guard correctly stays silent (nothing to warn — code absent, not blank).
- ~8 CC far-dated blank-IV rows (CCK7/CCN7/CCZ7) — these ARE in `_D` with valid opt_exp;
  blank because 0–1 shared call/put strikes → F_parity can't resolve a forward. Correct
  behavior (illiquid far contract SHOULD be blank, not carry a garbage IV).

**IV reconciliation (sandbox 22.7% vs our 19.51%):** NOT a bug — a tenor mismatch. Sandbox's
"~30-day" constant-maturity path (`skew_history._rolling_series`) filters to standard letters
H/K/N/Z, EXCLUDING serials, and falls back to the front standard tenor = Dec (CTZ6, ATM IV
~21.6–22.7%). Our 19.51% is the true Sep serial; the sandbox's own Sep solve agrees (~19.5–20.1%).
See `SERIAL_MONTH_EXPIRY_REFERENCE.md` in repo for full serial/expiry logic (Rule 10.51/10.54).


## 2026-06-12 — vlm_signal_backfill.csv carry month-gap repair (commit eae8e25)

`data/signals/vlm_signal_backfill.csv` was rebuilt to fix a carry month-gap bug: the
producer (`append_backfill.py` in the market-intelligence repo) had hardcoded a 2-month
CT1→CT2 gap, which is only correct for MAR→MAY / MAY→JUL. JUL→DEC (5 months) and DEC→MAR
(3 months) front pairs had inflated `si_carry_approx` / `pct_si_approx` (and therefore
`pct_si_zscore_1yr`).

What changed in this commit (only 3 columns + one date relabel + one new row):
- **870 rows**: `si_carry_approx` / `pct_si_approx` recomputed with the true gap.
- **All rows**: `pct_si_zscore_1yr` rebuilt over the corrected series (252-row window).
- **2026-06-08 → 2026-06-09**: relabeled (snapshot date-shift victim; prices matched 06-09).
- **2026-06-11**: reconstructed (was swallowed by the date-shift + duplicate-date guard);
  futures from local history, IV/HV from clean 06-11 options data, gap 5.
- All other columns byte-identical. Row count 1,474 → 1,475.

**OPEN:** 12 rows **2026-05-04 → 05-19** were NOT corrected (could not be matched to
standard contracts; suspected Bloomberg-generic bootstrap source) — pending Lou's
terminal verification and a follow-up pass. They retain their original (likely-wrong)
values.

Going forward the producer derives the gap from contract tickers in the EOD snapshot,
and the snapshot is written by `settle_watcher` only after both futures and options
settle — so this class of bug should not recur.

## 2026-07-05 — oi_data.csv volume backfill + settle/date correction (commit 67e970c)

**What was decided (Lou-approved):** `oi_data.csv` fully corrected and backfilled.
- `volume` backfilled 2008→present from Bloomberg `PX_VOLUME` finals (145,891 cells);
  `efp/efs/block` 2012-09-17→present (event-sparse: empty = no activity, real).
- Daily-era rows (2026-05-05→) had TWO defects: stamped with the 09:30 RUN date
  (T+1) and carrying 09:30 in-progress PARTIAL values (volume 18–56% of finals;
  settle a live price). Fixed: dates restated to TRADE dates, values replaced
  with official session finals. Holiday-duplicate rows (05-25/06-19/07-03 runs)
  collapsed. Uniform contract now: **every row = trade date + official finals.**
- KCMAY2 2026-05-04→05-15 sourced from dated KCK27 (generic is stale at BBG);
  (2026-05-18, KCMAY2) intentionally absent — May-2028 not listed (Lou's EXS).
- Gate before any write: settle series proved EXACT vs Bloomberg (6,518/6,518)
  + 3/3 Dec roll boundaries (2015/2019/2023). Builder:
  `VLM Data/backfill_oi_volume.py`. Audit: `data/oi_volfill_PROVENANCE.csv`
  (218,988 entries, local/untracked). Backup:
  `data/oi_data.backup_pre_volfill_2026-07-05.csv` (gitignored).
- Go-forward: `vlm_master_fetch.py` (VLM Data repo, commit 39d9a49) now pulls
  prior-session FINALS via HistoricalDataRequest (PX_SETTLE-preferred), stamps
  trade dates, dedups on trade date. RT OPEN_INT capture unchanged (was always
  official). Gateway docs updated (vlm-data-gateway commit 2aa2816).

**Why:** three-way reconciliation (ICE tape vs Bloomberg vs this file) exposed
the 09:30 snapshot defect; the seasonal/session-volume engine needs deep daily
finals. **Rejected:** leaving era-B partials (permanent semantic seam); keeping
run-date stamping (violated this file's documented trade-date convention).

**RESOLVES the 2026-06-12 OPEN item:** the 12 suspect rows 2026-05-04→05-19 were
exactly this defect class; all daily-era rows are now trade-date-stamped finals.

**Consumer note:** `oi_last_date` now reads the completed session (one day
earlier than the old run-date label). Verified live post-push: gateway serves
2026-07-02, CTDEC1 settle 77.12 / vol 18,267 / efp 47 / efs 207.

## 2026-07-06 — INFORMATIONAL: ICE session-volume engine now co-tenants our Supabase project (additive, non-breaking)

**This is a heads-up, not a change request. Nothing this repo consumes changed.**

A separate subsystem — the **ICE session-volume engine** (lives at
`…\Desktop\VLM_Session_Volume_Project\ice_timesales_engine`, NOT in this repo) —
is migrating its database into the **same shared Supabase project this repo uses**:
project ref `luhvqxneulzqsyltcluh` ("marco@vlmsofts.com's Project"), which also hosts
`vlm_newsletters`, `vlm_users`, `vlm_posts`, etc.

**What the engine is:** it reads ICE tick CSVs from `C:\Ice eod records\` (READ-ONLY —
the folder captured daily by the Windows Task Scheduler jobs: VLM ICE Cotton Blotter 14:22,
VLM ICE Softs Blotter 13:35, VLM ICE All Surface 16:00), classifies trades, and archives
5-minute session-volume buckets for CT/KC/SB/CC. It shares upstream data lineage with us
(our `oi_data.csv` also derives from Bloomberg/ICE for the same commodities), but it is a
distinct producer writing distinct tables.

**Six NEW tables added to the shared project (all additive, RLS disabled to match the
project's other non-RLS tables). None collide with, rename, or modify any table, column,
endpoint, or date convention this repo uses:**

| Table | PK | Rows | Purpose |
|---|---|---|---|
| `ticks` | (commodity, session_date, ice_code, seq_num) | ~130,490 | every ICE trade tick, permanent |
| `minute_agg` | (commodity, session_date, ice_code, minute_ts, primary_type) | ~15,249 | 1-min buckets per contract/type |
| `bar5m` | (source, commodity, session_date, ice_code, bucket_ts, primary_type) | ~195,188 | durable 5-min archive, source-labeled |
| `ingest_log` | (commodity, session_date, ice_code, file_name) | ~25 | per-file ingest audit |
| `reconcile_flags` | (commodity, session_date, ice_code) | ~25 | tape-vs-settle reconciliation |
| `block_supplement` | (commodity, session_date, ice_code, source) | ~2 | block-trade supplement |

Key column notes:
- `ticks`: commodity TEXT, session_date TEXT 'YYYY-MM-DD', ice_code TEXT (e.g. 'CTZ6'),
  generic_code TEXT nullable (e.g. 'CTDEC1'), exchange_time TEXT (ISO naive ET),
  price/size DOUBLE PRECISION, primary_type TEXT, conditions_raw TEXT, seq_num BIGINT,
  window_preset TEXT (night|day|other), ingested_at TEXT.
- `bar5m` (THE archive): source TEXT `'ice'|'bloomberg'` (never mixed in a query),
  commodity, session_date, ice_code, generic_code nullable, bucket_ts TEXT (ISO naive ET,
  floored to 5min), window_preset (night|day|other), primary_type, sum_size DOUBLE
  PRECISION, trade_count INTEGER. Bloomberg intraday seed covers 2025-12-22 forward
  (~6.4 months) as source='bloomberg'; ICE captures are source='ice' going forward.
- primary_type ladder (trade classification): efs_delete > efp > efs > block > leg > outright.
- Window presets: cotton session spans 2 calendar dates (9pm ET prior evening → 2:20pm ET
  session day); boundaries 21:00 / 07:00 / 14:20 ET all fall on 5-min marks.

**Blast radius for THIS repo:** none. The six tables are named distinctly and are purely
additive to the shared project. As of this notice, **no code in this repo reads any of the
six engine tables** (verified: no reference to bar5m / minute_agg / ingest_log /
reconcile_flags / block_supplement / engine `ticks` / project ref anywhere here). This
entry exists only so future Supabase work in this repo is aware of the new neighbor tables
and the shared-project relationship.
