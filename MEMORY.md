# Open interest dashboard — MEMORY

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
