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
