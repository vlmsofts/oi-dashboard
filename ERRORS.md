# Open interest dashboard — ERRORS

> Log approaches that took >2 attempts or a user correction. Check this before
> suggesting approaches to similar problems.

## 2026-07-15 — Which options date was the "phantom" (07-06 vs 07-03)

**What didn't work:** When told "clean the 07/03," I initially planned to delete 07-03 —
and separately first framed 07-06 as the phantom by naming-convention reasoning alone. Both
were guess-first, not data-first.

**What worked instead:** Compare the actual row values across adjacent dates. The phantom is
the BYTE-IDENTICAL duplicate, provable by diffing OI/settle/vol per strike. Result: **07-03
is LEGIT** (correct release of the real 07-02 holiday-eve session, distinct values); **07-06
is the phantom** (byte-identical dup of 07-03, no futures session maps to it via trade+1bday).

**Note for next time:** For any "which row is bad" question in these CSVs, prove it with a
per-strike value comparison across neighboring dates FIRST. The true phantom is always the
byte-identical duplicate with no corresponding futures session. Don't infer from the date
label alone — and if the user names a date that the data contradicts, flag it before acting.

## 2026-07-15 — Desync audit script mislabels holiday-boundary sessions as "orphans"

**What didn't work:** A weekend-only `prev_bday`/`next_bday` desync audit flags legitimate
post-holiday options releases (05-26 after Memorial Day, 06-22 after Juneteenth, 01-02 after
New Year) as "orphans" because the shifted date lands on the holiday, not the real session.

**What worked instead:** Add the byte-identical-duplicate test. A real desync/phantom is a
DUPLICATE of a neighbor; a holiday-boundary session is DISTINCT data. Only flag as a defect
if the row is byte-identical to an adjacent date. In this file, 07-06 was the only true dup.

**Note for next time:** Weekend-only date shifts (`_next_bday`/`_prevBday`/`trade_dt_of`) are
holiday-blind everywhere in this chain — correct on normal days, wrong across market holidays.
They become holiday-correct only once producer-stamped dates (from the real Bloomberg session)
flow through. Any audit built on them will false-positive around holidays; gate on the
duplicate test, not the date arithmetic.
