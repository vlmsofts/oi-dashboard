# Open interest dashboard — MEMORY

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
