# COORDINATION — CTN7 (and all N7 softs) corrupt IV/Greeks in options_oi.csv

**For:** OI-dashboard owner. **From:** options-flow-analyzer session, 2026-06-27.
**Status:** ROOT CAUSE CONFIRMED (read-only diagnosis). **No code changed.** Fix proposed below for your authorization — `vlm_master_fetch.py` is the protected master; nothing touched without your OK.

---

## Symptom
In `data/options_oi.csv`, the **CTN7 (July 2027)** options have corrupt `iv_pct` / `delta` / `gamma` / `vega` / `theta`. Calls and puts are priced against different forwards — a put-call-parity violation:
- Same-strike call vs put IV diverge ~20+ vol points (strike 75 on 2026-06-25: call `iv_pct` 32.06 vs put 8.69).
- 50-delta call sits at strike ~69, 50-delta put at ~73-74 — the two wings imply forwards ~5 apart.
- `CTN7P 78`: `iv_pct` 3.7322, delta -0.9361 — a 3.7-vol put is impossible; symptom of a too-low forward.
- Every OTHER CT contract (Z6/H7/K7/U6/X6/Z7) is internally consistent and agrees with the ICE settled surface to ~0.000-0.001. **Only N7 is broken.**

## Root cause (verified end-to-end)
The master prices CTN7 options against the **dying post-FND CTN6 stub**, not real CTN7.

1. `vlm_master_fetch.py:739` — the options forward is read verbatim from the resolver-chosen Bloomberg generic slot:
   ```python
   bbg_slot = get_bbg_slot(ice_code)                           # 'CTN7' -> 'CTJUL1'
   F        = raw.get(bbg_slot + ' Comdty', {}).get('px_last') # = CTJUL1 px_last
   ```
   `ice_code = sec[:4]` is identical for calls and puts, so F is the same for both — the split is NOT puts/calls drawing different fields; **F itself is wrong.**

2. `contract_dates.py` `resolve_generic_to_ice` (line ~250) rolls generics on **OPTION expiry** (`opt_exp > as_of`), and `_calendar_for` (line ~222) DROPS contracts whose `opt_exp is None`. CTN6's option already expired (~2026-06-12) so CTN6 is excluded → the resolver maps **`CTJUL1 → CTN7`**.

3. But **Bloomberg rolls `CTJUL1` on the FUTURES lifecycle, not the option's.** CTN6's future is still alive until LTD 2026-07-09, so Bloomberg's live `CTJUL1 Comdty` is still **CTN6**. Verified in `oi_data.csv`:
   | date | generic | settle | FND | LTD | open_int |
   |---|---|---|---|---|---|
   | 2026-06-25 | CTJUL1 | **72.09** | 2026-06-24 | 2026-07-09 | 97 (dying) |
   | 2026-06-25 | CTJUL2 | **78.75** | 2027-06-24 | 2027-07-08 | 10,321 (live = real CTN7) |

   So the code feeds **72.09 (post-FND dying CTN6)** as CTN7's forward — ~7 points too low.

**The FND link (the precise trigger):** CTN6 entered delivery at FND 2026-06-24 — its future became an illiquid dying stub (OI 97, price sliding 74→72→70) but stayed in the generic chain as `CTJUL1` until futures LTD. The corruption lives in the **gap between CTN6's option expiry (~06-12) and its futures LTD (07-09)**; FND is when the stub price goes badly wrong.

**Reproduction (repo's own `black76_validator.implied_vol`, T=352/365, rfr=0.0515):**
F=72.09 reproduces the corrupt CSV to the decimal; F=78.75 collapses calls+puts onto one consistent ~18-20.5 smile. The forward is the only thing wrong.

## Scope
- **All four softs' N7** — CT/KC/CC/SB N7 are corrupted identically on 2026-06-25 (verified). Same resolver feeds all four. (Their N6 options already expired → only N7 shows it.)
- **Recurs every cycle** — every front month hits the option-dead/future-alive window once per year, per commodity.

### Historical-scope scan (added 2026-06-27 — 110,537 master rows scanned)
Detecting the defect signature (per-contract MEDIAN same-strike call/put IV gap > 0.05 — robust to legitimate wing skew, which inflates only the wings, not the median):
- **133 flagged (date, commodity, contract) tuples across 17 trading days, 2026-06-03 .. 2026-06-26, ALL FOUR commodities.**
- **NOT just N7 — it's whichever month is the dying front contract in its option-dead/future-alive window.** Worst offenders are the **N6** contracts crossing their own window in early-mid June: `CCN6` median gap **3.03** (06-12), `KCN6` **1.59**, `CTN6` **1.04** — far worse than N7 because they were the actively-dying front month then. Flagged month-letters span N/U/Z/V/H/K (every near-dated month rolls through it).
- **Magnitude scales with how dead the contract is:** CCN6 3.03 vol (deep in window) vs CTN7 ~0.09 (early in its window). 0.5–3.0 vol gaps = 50–300 vol points of call/put disagreement — unmistakably a wrong forward, not skew.
- **Healthy contracts** have median gaps ~0, confirming the signature is specific.

**Implication:** the defect is systematic, recurring, multi-commodity, every-front-month — it has silently published wrong `iv_pct`/Greeks for the dying front month on at least 17 days in June alone. The Option-A fix corrects ALL of it at once (single shared root cause). Worth checking how long the gateway has served these before this scan window (scan only covered the dates present in the current master file).

## Proposed fix (Option A — narrowest; for your review, NOT implemented)
**File:** `vlm_master_fetch.py`, `run_options_append`, ~line 739.
Resolve the options forward by the **actual ICE contract** using `FUT_CUR_GEN_TICKER` (`cur_ticker`), which the code ALREADY fetches (lines ~274-275, 291) but currently uses only for the futures FND/LTD pass:
- Build a one-time `{ice_code: px_last}` map from `raw` (each generic's `cur_ticker` → its `px_last`).
- `F = px_by_ice.get(ice_code)` instead of reading the generic slot's price.
This keys CTN7's forward to CTN7's real price (78.75), immune to the slot/option-roll mismatch, fixes all four commodities, and leaves the shared `get_bbg_slot` contract untouched.

Alternatives (wider blast radius — see full diagnosis): (B) roll the resolver on futures LTD not opt_exp (touches `get_bbg_slot`, also used by `oi_ice_fetcher.py:251`); (C) post-solve put-call-parity sanity gate that blanks Greeks on divergence (defensive backstop, pair with A).

## Blast radius of the DEFECT (consumers reading the wrong N7 Greeks)
- `options_oi.csv` `iv_pct/delta/gamma/vega/theta` for N7 → gateway `/v1/openinterest/{cmd}/options`.
- Downstream: GEX/flow analyzer, IV-surface/skew cards, EOD-brief vol enrichment, any straddle/skew analytics on the July contract.
- **NOT affected:** open_int, oi_chg, px_settle, px_volume, strikes, dates; `oi_data.csv` futures (correct).

## Interim workaround (flow analyzer side, already decided)
The delta-adj-vol loader (S1) will prefer the ICE settled surface (internally consistent, single forward) for any contract failing a call/put-parity check, and the OI master for all others — so the flow signal stays honest while the master fix is coordinated.

**Confidence:** ~95% on root cause (code path read, resolver run live, data verified, corrupt values reproduced to the decimal). Not validated against Bloomberg Terminal (fixes described, not run).
