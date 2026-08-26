# Open interest dashboard — ERRORS

> Log approaches that took >2 attempts or a user correction. Check this before
> suggesting approaches to similar problems.

## 2026-08-25 — "Watcher failed" that was actually a missed trigger

**What didn't work:** treating a reported prelim-watcher failure as a watcher/
scheduler/credential problem. The log was full of clean "no unread PRELIM OI
messages" polls and Task Scheduler showed exit 0 — which is EXACTLY what a
missed trigger looks like, not proof of health.

**What worked instead:** read the MAILBOX, not the log. A read-only IMAP listing
showed the email had arrived on time with the CSV attached and an **empty
subject**, so the `(UNSEEN SUBJECT "prelim")` search never matched it.

**Note for next time:** for any "the watcher didn't fire" report, list the source
mailbox/folder FIRST and confirm the trigger condition actually matched. A green
log only proves the poll ran, never that the input was seen.

## 2026-08-25 — Two bugs that only a full render would catch

**What didn't work:** verifying a card change by unit-testing the data layer.
`load_futures()` returned perfect numbers for all 4 commodities while
`build_html()` was still broken: `color_chg()` threw TypeError on the `None`
from a missing baseline. CT and KC had already written PNGs, so the failure was
PARTIAL — 2 cards and no report, and under `pythonw.exe` (no console) it would
have been silent at 09:35.

**Also missed by my own testing:** a row with OI but BLANK dates rendered as a
nameless `—` row whose OI still inflated the total (8,624 such rows exist in the
2008-era history). Found by an independent Sonnet audit, not by me — my crash
replay only covered blank OI, not blank DATES.

**Note for next time:** for anything that renders, ALWAYS generate the actual
image before committing — data-layer tests cannot catch formatter crashes. And
when enumerating messy-data cases, treat each nullable column as an INDEPENDENT
axis (blank OI, blank dates, blank both), not one "bad row" case. Spawn a
second-opinion audit on anything unattended; it caught what I missed.

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

## 2026-08-26 — Prelim OI built but never delivered; log looked clean

**What didn't work:** Reading the watcher log as if a missing "WhatsApp sent" line meant the
build failed. It did not. The log's LAST line was `XLSX : ...` and then nothing — no error,
no traceback. That absence is the tell, and it is easy to misread as a silent build failure.

**What worked instead:** `Get-ScheduledTaskInfo` → `LastTaskResult : 267014` (0x41306, task
terminated). The 05:15 run BUILT the PNG (395,608 bytes) and XLSX successfully, then Task
Scheduler killed it at 05:25 on the 10-min `ExecutionTimeLimit` — mid-WhatsApp-send. The
report existed on disk the whole time; only delivery was lost.

Root cause of the slowness: the build normally takes 4s (08-24: 4s, 08-25: 3s) but took 253s
and 306s on 08-26 under machine load. Re-timing the identical render afterwards gave 34s total
with the screenshot itself at 0.2s — so the render code is NOT the problem; CPU contention was.
Note the box had ~9 concurrent python processes including a DUPLICATE `pull_loop.py CC` pair.

Fix: `ExecutionTimeLimit` 10min → 20min. The watcher's own build timeout (480s) was documented
as sitting "safely under" the scheduler limit — true, but the scheduler budget must cover
build AND SEND. That gap is what dropped the report.

**Note for next time:** Two different failures on consecutive days, DIFFERENT causes — do not
assume a repeat. 08-25 was a DETECTION failure (blank subject, automated polls found nothing,
only a manual run delivered). 08-26 was a TIMEOUT/kill failure. Also: an exit code of 267014
on any VLM scheduled task means "scheduler killed it," never "the script errored" — check
`LastTaskResult` BEFORE reading a traceback that isn't there. And a report that is built but
undelivered must fail LOUDLY; silence read as success for two mornings running.

## 2026-08-26 — OI cards missing: hung process reported exit code 0

**What didn't work:** Checking the scheduled task's status to decide whether the
morning fetch ran. `vlm master fetch` showed `Last Result: 0` and `Status: Ready`
after the reboot — that zero is the **reboot-kill exit, not success**. The task had
actually hung mid-run and written nothing to oi_data.csv. Reading task status alone,
this failure is indistinguishable from a clean run. Also misleading: the process
reported `Responding: True` the entire time it was wedged.

**What worked instead:** Sample CPU twice, a few seconds apart. Identical values
(`0.796875` both times) prove *frozen*, not *slow* — a working process moves. Then
diff the log's own timing against prior days: `grep "JOB 2"` showed today starting
at 09:33:53 vs ~09:30:4x on all 11 previous runs, exposing a stall before the hang.
The decisive artifact was file mtime: oi_data.csv still stamped the PRIOR day while
the task claimed success.

**Root cause: OneDrive file-lock contention, not code.** Proof is the re-run delta —
after reboot, the identical JOB 2 against the identical 17MB file finished in **1
second** (09:56:44 -> 09:56:45) versus 22+ minutes hung. Nothing in the code changed.
OneDrive (PID 13408) had accumulated ~58,654 CPU-seconds since Aug 22.

**Note for next time:** This is the THIRD consecutive silent failure on this box
(08-25 detection, 08-26 05:15 scheduler kill, 08-26 09:30 lock hang) — all three
under CPU/lock contention, all three reading as success. Distinct exit codes,
distinct causes, same symptom: nothing arrived and nothing complained. Rules that
generalize: (1) on any VLM scheduled task, verify the EFFECT (file mtime), never the
status field — `0` can mean killed and `267014` means scheduler-killed, never script
error; (2) two identical CPU samples = hung, and `Responding: True` means nothing;
(3) any long unguarded read of a large file inside the OneDrive sync root is a
single point of failure for the whole daily chain. `build_whatsapp_oi.py`'s
`check_freshness()` mtime guard was the ONLY thing that caught this — it is load-
bearing, do not weaken it to a trade_date comparison (OI is T+1, so trade_date is
legitimately yesterday even on a good day; mtime is what proves today's fetch ran).
