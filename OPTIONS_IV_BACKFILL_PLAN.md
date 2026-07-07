# Options IV/Expiry Historical Backfill — PLAN (for adversarial review before any code)

**Goal:** retro-fill the BLANK `expire_dt` / `days_to_exp` / `iv_pct` (+ greeks:
delta/gamma/vega/theta) cells in `data/options_oi.csv` for the ~last year
(2025-01-03 → 2026-07-06, 388 trade dates), **correctly and only where the value
is genuinely recoverable.** Read-only w.r.t. all other data; fill-empty-only; full
provenance; backup first. This mirrors the shipped go-forward fix (commit 3ce649e)
applied retroactively.

---

## 0. Non-negotiables (house rules)
- **Fill-empty-only.** NEVER overwrite a populated `iv_pct`/`expire_dt`/greek. Touch a
  cell only if it is currently blank.
- **Never touch** settle/open_int/oi_chg/strike/volume or ANY futures row. Options rows only.
- **Backup** `oi_data.csv`-style: copy `options_oi.csv` → `options_oi.backup_pre_ivfill_<date>.csv` first.
- **Provenance CSV** logging every cell written: (date, commodity, security_des, field,
  old=blank, new, source, forward_used, F_value, dte).
- **Idempotent + additive**: re-running changes nothing already filled. Row count unchanged.
- **Schema frozen**: 17 cols, same order. No new columns, no reordered columns.
- Blast radius: `options_oi.csv` is served by the gateway (`/v1/openinterest/{cmd}/options`).
  Additive fills only → no consumer breaks. STOP for Lou before promoting the filled file.

---

## 1. The exact universe (measured 2026-07-07, not estimated)

Blank-IV rows total by cause (121,948 file rows, 388 dates):

| Cause | Rows | Action |
|---|---|---|
| **6 RECOVERABLE** — has expiry (post-fix `_D`), DTE≥2, settle>0, ≥4 shared C/P strikes | **38,576** | **FILL** |
| 5 no forward — <4 shared call/put strikes (illiquid) | 25,838 | LEAVE (correct blank) |
| 1 no expiry in JSON — 8 expired front contracts rolled off forward calendar | 24,295 | **FILL after adding historical expiries** (see §3) |
| 4 no settle price | 599 | LEAVE (nothing to solve) |

The "no expiry" 24,295 is NOT random: **8 codes** — CTN6 (13,603), KCN6 (3,673),
CCN6 (3,615), SBN6 (2,051), SBM6 (590), KCM6 (411), CCM6 (348), SBX6 (4). All are
Jun/Jul-2026 (and SBX6 phantom) contracts that **expired and dropped from the forward
JSON**. Their true expiries exist in `DOCS_SANDBOX/*ProductSpecExpiryDates*.csv` and
`ICE_Expiry_Dates_ALL_COMMODITIES.md` (e.g. CT Jul26 opt LTD 2026-06-12).

**Net fillable target: 38,576 (recoverable now) + ~24,291 (after historical expiry
graft, excl. 4 SBX6 phantoms) ≈ 62,867 rows.** Everything else stays blank *by design*.

---

## 2. Method — per recoverable row, reproduce the daily job EXACTLY

The go-forward pipeline (`VLM Data/vlm_master_fetch.py` lines 877–944) is the ground
truth. Backfill MUST use the identical math so history == future semantics:

1. `ice_code = security_des[:4]`; `opt_exp = get_opt_exp(ice_code)` (post-fix module).
2. `dte = (opt_exp - trade_date).days` (CALENDAR days; trade_date = the row's own `date`).
   Guard: `dte >= MIN_DTE_DAYS (=2)`.
3. **Forward F, same priority as the job:**
   a. `F_parity` = median over shared call/put strikes of `K + (C−P)·e^{rT}`, needs
      **≥ MIN_PARITY_PAIRS (=4)** shared strikes in that (date, commodity, contract_month).
   b. else ICE contract settle (from oi_data.csv futures for that trade date) — the
      historical analogue of `px_by_ice`.
   c. else generic-slot last — NOT available historically; if a/b fail → leave blank.
4. `rfr = 0.0365` (base SOFR, no credit spread) — matches job line 849/851 fallback.
   **Open question for review:** the live job uses live SOFR when available; for history
   should we use flat 0.0365, or a historical SOFR series? (0.0365 is the documented
   monthly fallback; SOFR moved <0.5% over the window → sub-0.3 vol impact. Propose flat.)
5. `iv = implied_vol(F, K, T, rfr, settle, is_call)` using the job's OWN
   `black76_validator.implied_vol` (import it — do NOT reimplement).
6. If iv solves: write `iv_pct = round(iv*100,4)`, `expire_dt = str(opt_exp)`,
   `days_to_exp = dte`, and greeks from `black76_greeks(...)` (signed delta, same as job).
7. Any step fails → leave ALL of that row's target cells blank (never partial-garbage).

---

## 3. Historical expiries for the 8 rolled-off contracts (§1 row "1")
- Source: `DOCS_SANDBOX/CT_ProductSpecExpiryDates_REFERENCE.csv` (CT) +
  `ICE_Expiry_Dates_ALL_COMMODITIES.md` (CC/KC/SB) — both carry Jul-2026/Jun-2026
  OPTION LTDs. Cross-check the two sources agree before use.
- Build a small static `HIST_OPT_EXP = {'CTN6': date(2026,6,12), 'KCN6': ..., ...}`
  (8 codes) with a comment citing the source line. Feed it as a fallback when
  `get_opt_exp` returns None. **SBX6 stays blank** (phantom, not a real ICE option).
- This is a one-time historical graft, NOT a change to contract_dates.py (that module
  is a forward calendar by design; grafting expired contracts into it would be wrong).

---

## 4. Verification (before promote)
- **Anchor test:** pick a KNOWN-GOOD populated quarterly row (e.g. CTZ6 on a 2026-06
  date) and confirm the backfill, run on a *blanked copy* of it, reproduces the existing
  `iv_pct` within rounding. If it can't reproduce a value the job already wrote, the
  method is wrong — STOP.
- **Sep-2026 serial spot-check:** CTU6 2026-07-06 must land ~19.5% (matches the
  end-to-end proof from the go-forward fix).
- Row count identical pre/post. Schema diff empty. Zero populated cells changed
  (assert old==blank on every write). Provenance row count == cells written.
- IV band: the live job (`vlm_master_fetch.py:934-965`) has NO sane-band filter — it writes
  whatever `implied_vol` returns. To keep history == future semantics, backfill MUST do the
  same: DO NOT withhold or clamp. Instead LOG (in provenance) any solve with extreme moneyness
  (|ln(F/K)| > 0.25) AND dte < 10 — the degenerate deep-ITM/near-expiry corner where IV is
  hyper-sensitive to settle rounding (e.g. CT Jul26 K100 P @4DTE: 0.1 settle → ~35 vol). These
  are the ONLY rows expected to diverge from a re-solve; flag for eyeball, don't drop.

---

## 5. Deliverables
1. `backfill_options_iv.py` (read source CSVs, write `options_oi_IVFILL.csv` +
   `options_ivfill_PROVENANCE.csv`; NEVER writes live file).
2. QA report: counts by cause, anchor-test result, spot-checks, band check.
3. Promotion is a SEPARATE gated step after Lou reviews the QA report.

---

## === v3 SCOPE DECISION (post CODE adversarial review, 2026-07-07) ===

**The ~3,523 thin-liquidity rows CANNOT be faithfully backfilled — and that is final.**
The live job's forward priority is F_parity -> `px_by_ice` -> generic-slot. `px_by_ice` =
`fetch_ice_forwards(session, ...)` = a LIVE Bloomberg `PX_LAST` pull at run-time
(`vlm_master_fetch.py:868`), with **zero persistence**. There is no historical record of what
it returned on any past date. So for rows in parity groups with <4 shared strikes (F_parity
fails), the forward the job actually used is **unrecoverable**. Substituting oi_data.csv's
futures settle would inject a DIFFERENT number (code comment line 871: "0.7-2.2% off the
forward") and for serials the wrong contract entirely — a fidelity violation.

**DECISION: honest scope = "reproduce every row the job filled via F_parity."** The ~3,523
thin rows are LEFT BLANK by design and logged as such. This is not an under-fill bug; it is the
correct call — backfilling them with a substitute forward would fabricate values that never
matched the live pipeline. `EXTREME_CORNER`-style transparency: the QA report will state the
exact count left blank for this reason. **Needs Lou's one-line sign-off that this scope is
acceptable (vs. leaving those rows for a future live-reconstruction, which would require a
Bloomberg pull).**

**SBX6 fix applied** (adversarial review caught it): was wrongly omitted from HIST_OPT_EXP;
Nov IS a valid SB serial, LTD 2026-10-15. Now fills its 9 rows.

## === v2 CORRECTIONS (post adversarial review + cross-check, 2026-07-07) ===

All numbers below re-measured with the two blocker fixes applied. These SUPERSEDE §1/§2/§3.

### C1. trade_dt shift (BLOCKER 1 — confirmed)
The daily job (`vlm_master_fetch.py:855-859`) computes `trade_dt = row_release_date − 1 day,
then skip back over Sat/Sun ONLY` (no holiday calendar). Proven: stored `days_to_exp`=70 for
CTV6 @release 2026-07-06 = opt_exp(09-11) − 2026-07-03 (trade), not − 07-06 (=67). Backfill
MUST use this exact weekend-only shift. dte/T measured from trade_dt, never the row date.

### C2. strike from security_des (BLOCKER 2 — confirmed + parser validated)
`strike_px` is BLANK for 100% of rows before 2026-04-25 (all 2025 + Q1-26). True strike is in
`security_des` (e.g. 'CTN6P    62' → 62). Parser `^[A-Z]{2}[A-Z0-9]{2}[CP]\s+([0-9.]+)$`
validated: **91,539 rows where both exist agree, 0 mismatch; 0 unparseable when strike_px blank.**
Backfill: use strike_px if present, else parse security_des. Mandatory before any F_parity.

### C3-FINAL (2026-07-07, supersedes C3 below) — ALL parse fixes + validated serial map
Three parse fixes total: (a) trade_dt weekend-shift, (b) strike from security_des, (c) put_call
from security_des[4] (blank in column pre-2026-04-25). Plus validated rolled-off expiries
(KCM6/CCM6=2026-05-08, SBM6=2026-05-15, CTN6/KCN6/CCN6=2026-06-12, SBN6=2026-06-15 — all
data-validated, see ICE_OPTIONS_SERIAL_EXPIRY_AUTHORITY.md). File = 123,960 rows (grew as daily
job appends).

| fill action | rows | reconciled |
|---|---|---|
| **expire_dt + days_to_exp** | **88,213** | 88,213 + 11 leave = 88,224 blank expire_dt ✓ |
| **iv_pct + greeks** (full B76) | **69,130** | 69,130 + 19,974 expiry-only + 278 leave = 89,382 blank iv ✓ |
| leave blank — SBX6 phantom (no ICE option) | 11 | genuinely uncoverable |
| leave blank — past-expiry / under-min-dte | 267 | correct |
| expiry-only (no forward: <4 shared, or no settle) | 19,974 | get expire+dte, no iv (correct) |

Both totals reconcile exactly against raw blank counts (no silent drops). The two write-units
(expire/dte vs iv/greeks) are independent per C4-resolved.

### C3. (SUPERSEDED by C3-FINAL) earlier classification (C1+C2 only)
| bucket | rows | action |
|---|---|---|
| 6 RECOVERABLE (expiry+DTE≥2+settle+≥4 shared strikes) | **51,917** | FILL iv+greeks |
| 5 no forward (<4 shared C/P strikes, illiquid) | 18,110 | leave blank (correct) |
| 0 no put_call field | 18,664 | leave (not an option leg) |
| 4 no settle | 346 | leave |
| 2 past/under-min-dte | 267 | leave (correct) |
| 1 no expiry uncoverable | 4 (SBX6 only) | leave |
(Recoverable jumped 38,576 → **51,917** once strikes were parsed correctly — the old count
under-counted the pre-04-25 era. This is the true fill target for iv/greeks.)

### C4. Historical expiry map — DERIVED FROM THE FILE ITSELF, not hardcoded guesses
Anchor test caught a wrong hardcoded expiry (I'd used CTN6=06-12 from a mis-read tab; then a
single stale row made me wrongly think it was ambiguous). RESOLVED: each rolled-off code has
ONE authoritative expiry = the in-file MAJORITY by distinct dates. Lou confirmed CTN6=06-12.
| code | authoritative opt_exp | evidence |
|---|---|---|
| CTN6 | **2026-06-12** | 7 of 8 dates in-file + Lou + ProductSpec Jul26 06/12 |
| KCN6 | **2026-06-12** | 8/8 dates in-file, no anomaly |
| CCN6 | **2026-06-12** | 7 of 8 dates in-file |
| SBN6 | **2026-06-15** | 8 of 9 dates in-file |
| SBM6/KCM6/CCM6 | **UNRESOLVED** | "Jun 2026" options, real settle+OI, no in-file expiry; ProductSpec tabs don't list June for SB/KC/CC. NEEDS Lou/ICE-rule before fill (~1,349 rows). |
| SBX6 | none (phantom) | SB has no Nov option; 4 rows, leave blank |

### C5. PRE-EXISTING DATA ERROR found (separate finding, NOT touched by fill-empty-only)
~356 already-POPULATED rows carry a wrong single-day expiry: CTN6/CCN6 stamped 2026-06-19 on
2026-06-03 only; SBN6 stamped 2026-06-26 on one date. These are the true expiry + ~1 week.
Populated → the backfill will NOT overwrite them (fill-empty-only). Flagged for Lou: decide
separately whether to correct these existing wrong stamps (would be an overwrite, gated).

### C6. Anchor test RESULT (the real gate) — method VALIDATED
13 of 15 sampled populated rows reproduced to ≤0.68 vol (most exactly 0.0000) with the
corrected method. The 2 outliers were BOTH the 2026-06-03 CTN6/CCN6 rows carrying the wrong
06-19 stamp (C5) — i.e. the method is right; those stored values are the ones that are wrong.
Re-running the anchor excluding known-bad-stamp rows → worst |mine−stored| well under 0.5 vol.

## OPEN QUESTIONS FOR ADVERSARIAL REVIEW
1. Flat rfr=0.0365 vs historical SOFR series — material? (I claim no, <0.3 vol.)
2. For forward fallback (b), is oi_data.csv's futures settle for that trade date the
   right F, given oi_data has its own two-era semantics (bootstrap finals vs 09:30
   snapshots)? Could a snapshot-era futures settle inject a wrong forward → wrong IV?
   **This is the highest-risk assumption.** Does F_parity (a) cover enough rows that (b)
   rarely fires, making it moot?
3. Should greeks be filled at all, or only iv_pct+expire+dte? (Job fills greeks; parity
   suggests yes, but greeks depend on F which is the risky input.)
4. RESOLVED (measured 2026-07-07): **1,091 rows have expire_dt populated but iv_pct
   blank** (DTE known, IV solve failed on forward/liquidity); **0 rows** have iv without
   expire. => "fill all four together" is WRONG. Backfill MUST fill expire_dt/days_to_exp
   as one unit (whenever opt_exp resolves) and iv_pct/greeks as a SEPARATE unit (only when
   the solve succeeds). A row can legitimately get expiry+DTE but no IV. Provenance must
   log the two units independently.
5. MIN_PARITY_PAIRS=4 and MIN_DTE=2 — inherited from the job. Correct for history too?
