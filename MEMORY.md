# Open interest dashboard — MEMORY

## 2026-08-26 — DECISION: migrate all repos OFF OneDrive (deferred to a quiet market period)

**Decided (Lou, 2026-08-26):** every repo moves out of the OneDrive sync root to a
plain local path. GitHub stays the source of truth, Railway runs the services, and
LOCAL becomes the redundancy layer — inverting today's arrangement where OneDrive
is doing a redundancy job that GitHub+Railway already do better.

**Why:** OneDrive file-locking is the root cause of the 2026-08-26 09:30 master-fetch
hang (JOB 2 wedged 22+ min on a 17MB read; identical read took 1s after reboot). It is
a latent single point of failure under EVERY scheduled job on this box, not just OI.

**NOT a design error, and worth remembering as such:** OneDrive was the ONLY off-machine
redundancy when this ecosystem started. It earned its place. It became pure downside only
once 20 repos were on GitHub and services were on Railway — the backup value is now
duplicated, the locking cost is not. This is outgrowing a tool, not misusing one.

**Timing — deliberate, do NOT opportunistically start this:** Lou's call is to run the
migration only during a QUIET MARKET PERIOD with a few weeks of runway. It touches every
repo, every absolute path and 54 scheduled tasks simultaneously; a half-finished migration
under live markets is far worse than the status quo, which is a KNOWN and now-instrumented
flaw.

**Measured scope (2026-08-26, verified by scan — not an estimate):**
  * 20 git repos under `Desktop/`, ALL already have GitHub remotes on `vlmsofts`
  * 84 hardcoded `OneDrive - VLM Commodities` paths across 63 .py files in 12 repos
  * Worst offenders: VLM Data (17 hits/11 files), Options_flow_analyzer (16/12),
    OTEXA DASHBOARD (11/10), CertStocks (10/4), export-sales-repo (9/7)
  * **54 of 62** VLM scheduled tasks reference a OneDrive path
  * Cross-repo writer: vlm_master_fetch.py writes into THREE repo trees
    (Open interest dashboard, CTA MONITOR, + its own) — migration cannot be done
    one-repo-at-a-time without breaking it mid-flight

**Sequencing agreed:** Tier 1 (make failures loud — see below) lands FIRST and is
independent of the migration. It is what makes the status quo survivable until the
window opens. The migration is Tier 2 and supersedes the per-read retry workaround.

**Master fetch defects found while diagnosing (all still OPEN, none actioned):**
  1. `return 0` is UNCONDITIONAL even when `summary['errors']` is populated — the script
     is structurally incapable of reporting failure to Task Scheduler. This is WHY exit
     code 0 lied on 08-26.
  2. `ExecutionTimeLimit=PT72H` — a hang gets THREE DAYS to squat and block every
     subsequent daily run. (Compare: prelim job was capped at 10min.)
  3. All five jobs share ONE try/except — a throw in spreads silently skips macro,
     signals AND options. No per-job isolation.
  4. ZERO alerting (grep-confirmed: no twilio/smtp/webhook). `VLM Daily Data Watchdog`
     runs 17:15 — ~8h after the damage — and does not check oi_data.csv at all.


## 2026-08-26 (later) — OI cards missed: master fetch hung on JOB 2, guard held

**Session summary.** Lou: "o.i. cards never came via whatsapp." Cause was UPSTREAM,
not the send path. `vlm master fetch` (09:30:38) hung at `--- JOB 2: OI dashboard
data append ---`, log frozen at 09:33:53 and nothing after. It never wrote
`oi_data.csv`, so at 09:35 `build_whatsapp_oi.py` hit `check_freshness()`, saw
yesterday's mtime, and refused to send stale data. **The guard did its job — it
caught the failure, it did not cause it.**

**How the hang was PROVEN (not inferred):** CPU sampled twice 5s apart was
byte-identical at `0.796875` — frozen, not slow. `Responding: True` the whole
time, so the process looked healthy. Today's JOB 2 also started at 09:33:53 vs
~09:30:4x on all 11 prior days (grep of the log), i.e. a ~3min stall BEFORE the
hang. Both PIDs (128552 master fetch, 136492 build_whatsapp) sat wedged.

**Fix was Lou's reboot, then a plain re-run.** After reboot: JOB 2 cleared in
**1 second** (09:56:44 -> 09:56:45, 36 rows appended) vs 22+ min hung. Same code,
same file, same size. That delta is the proof it was **file-lock contention, not
a code bug** — nothing was changed to make it work. Full run completed 09:58:54,
all jobs, all pushed (c076d1d, 042d7a0, d89c71a, 0cd7379).

**Verified effects, not statuses:** oi_data.csv/options_oi.csv/spread_ohlc.csv all
re-stamped 2026-08-26; oi_data max trade_date 2026-08-25 @36 rows (T+1 correct);
options_oi max 2026-08-26 (release = trade+1bday, correct); WhatsApp 4 sent /
0 failed (HTTP 201 each); site slug `open-interest-monitor-2026-08-25-2026-08-26`;
`.sent_2026-08-25` marker written 09:59:40. CT PNG was READ back to confirm real
content (Dec '26 198,614, -1,520 DoD; total 374,857; options panel populated) —
a PNG existing on disk is not evidence it rendered data.

**LIAR OF THE DAY:** the hung task reported `Last Result: 0` after the reboot
killed it. Zero is the reboot-kill exit, NOT success. Checked alone, a hang of
this class reads as a clean run. **`oi_data.csv` mtime is the only honest signal.**

**Open / NOT actioned (Lou's call, deliberately untouched):** master fetch reads
the 17MB oi_data.csv twice with no timeout — dedup guard at vlm_master_fetch.py
:514-516 and `load_yesterday_oi()` at :521. Both sit inside the OneDrive sync
root. A OneDrive stall there silently takes down the whole daily chain with no
alert; Lou only found out because cards didn't arrive. Options floated, none
implemented: (a) read timeout, (b) move data dir outside the sync root,
(c) watchdog alerting if oi_data.csv mtime isn't today by 09:45.


## 2026-08-26 — Prelim OI moved to Railway; gateway-backed baselines

**Session summary.** Two consecutive missed prelim deliveries (08-25, 08-26) with
DIFFERENT causes. 08-25 was DETECTION (blank subject; only a manual run
delivered). 08-26 was a KILL: the 05:15 run BUILT the PNG (395,608 bytes) and
XLSX, then Task Scheduler killed it at 05:25 on the 10-min `ExecutionTimeLimit`
mid-WhatsApp-send. `LastTaskResult 267014` = scheduler-killed, NOT a script
error. The log's last line was a normal-looking `XLSX : ...` — **silence read as
success for two mornings.** Build had gone 4s -> 306s under machine load (9
python processes, incl. a DUPLICATE `pull_loop.py CC` pair); re-timed later at
34s total with the screenshot itself 0.2s, so the render code was never at fault.

**Decided:** move the job to a Railway cron service, stateless, reading official
OI from the gateway instead of a bundled CSV.

**New endpoint** `GET /v1/openinterest/history?from=&to=` (gateway 6ffce45,
LIVE). Full rows, same field names as `/daily`. Rationale: `/daily` already
fetched the WHOLE `oi_data.csv` via `github_reader.read_file` and discarded all
but the newest date — this is the same read with a date filter. **Rejected** a
per-range payload cache: `MemoryCache.get()` returns None past expiry but NEVER
evicts, so keying on from/to would mint a permanent entry per range. It now
reuses read_file's single `github:oi-dashboard/data/oi_data.csv` entry (proven:
cache keys stayed at 1 across 27 distinct ranges).

**`OI_DATA_SOURCE` defaults to 'local', deliberately.** The Windows job runs the
file with no env overrides, so the DEFAULT is what runs at 05:00. An agent had
set it to 'gateway' before the endpoint existed — that would have broken the
next morning's run (proven: HTTP 400). Railway sets 'gateway' explicitly.

**Gateway swap proven equivalent, not argued:** an adversarial audit diffed all
4,765 sessions x 36 contracts between modes — ZERO differences including types
(both paths parse via `csv.DictReader`, so every field is a str). Rendered HTML
byte-identical bar the footer, which bakes `datetime.now()` and so differs
between ANY two runs.

**RAILWAY CONFIG-AS-CODE DOES NOT WORK — the trap of the day.** With Config Path
set to `/railway.prelim.toml`, deploy 438215a7 reported **SUCCESS** while having
built via RAILPACK from `requirements.txt` (flask+gunicorn, no Playwright, no
Chromium) and booted `gunicorn ... :8080` — a DUPLICATE OF THE DASHBOARD WEB APP.
It would never have polled email or built a PNG. Railway called it SUCCESS
because gunicorn started; only the BUILD LOG revealed it. Config-as-Code is
deprecated (dies 2026-12-01). Settings are now set DIRECTLY on the service and
are authoritative: builder=DOCKERFILE, dockerfilePath=Dockerfile.prelim,
startCommand, cron `*/5 7-13 * * 1-5`, restartPolicyType=NEVER (UPPERCASE enum).
`railway.prelim.toml` is retained as documentation only — **editing it changes
nothing.** Image size is the tell: 1.02GB (Playwright) vs 418MB (gunicorn).

Also cleared `watchPatterns`, which had been set to "Dockerfile.prelim" — that is
a redeploy TRIGGER filter, so changes to the .py files would never have deployed.

**Failure alerts are LOG-ONLY** (Lou's call: "I will be the alert"). A persistent
send failure would otherwise fire one WhatsApp per 5-min poll. `alert_failure()`
writes a banner-delimited `DELIVERY FAILURE: <reason>` block. The audit caught
that two docs still CLAIMED a Twilio alert — the 08-26 trap inverted, with the
operator waiting on a message that cannot arrive (fixed, 751d319).

**Fonts — reasoned, NOT observed.** The report declares ONE stack,
`Arial,sans-serif`. Arial does not exist on Linux; Liberation Sans is metrically
identical (same advance widths) so the fixed-width table cannot reflow;
`fonts-liberation` is now installed explicitly. Docker and WSL are BOTH ABSENT
from this machine, so the image has NEVER been built or run locally.

**State at close (2026-08-27 is the live test):** Windows task `VLM Prelim OI
Watcher` DISABLED (PT20M preserved); Railway `prelim-oi-watcher` live, 12 vars,
cron 03:00-09:55 ET. Both watchers race the same UNSEEN email and mark `\Seen`
only after a successful send, so only ONE can deliver. If Railway fails before
sending, the email stays unread — re-enable Windows and run manually.

**CLOSED — 54MB payload + no retry (commit below).** `_fetch_gateway_history()`
now sends `?from=today-400d`: 54MB -> 3.5MB (94% smaller), 280 sessions where
MoM needs 21. 400 CALENDAR days is a deliberate margin over 21 TRADING sessions
(absorbs holidays/closures); it is a FLOOR, not a window — pick_baselines()
still walks back N real sessions, so over-fetching is free and under-fetching is
the only risk. Retries: 3 attempts, linear backoff, transport+5xx only — a 4xx
is a real answer (bad key/params) and fails immediately rather than burning the
clock. Added an empty-rows guard: no rows now RAISES instead of building a
report with every baseline dashed. Verified both modes: identical baselines
(08-24/08-18/07-27), 36/36 reconciled, PNG body pixel-identical.

**NO OPEN ITEMS.**

**CLOSED — image trim (~1.02GB is fine, do not chase it).** The idea was to drop
"the unused headless shell" (272MB). That was BACKWARDS: `p.chromium.launch()`
(build_prelim_oi.py:453) defaults to headless=True, so the 272MB headless shell
is the one actually USED and full Chromium (428MB) is the spare. Removing full
Chromium means abandoning `mcr.microsoft.com/playwright/python:v1.58.0-noble`
and hand-building an image — reintroducing exactly the browser/library version
drift the Dockerfile header warns about — to save ~400MB on an image pulled ONCE
PER BUILD and cached across all 84 daily runs. If container cost ever matters,
narrow the CRON WINDOW (fewer starts), do not shrink the image.

**`pull_loop` contention — SUSPECT, NOT PROVEN.** Correlation only: 9 python
processes during the 05:10-05:25 window incl. a DUPLICATE `pull_loop.py CC` pair
(started 05:37:54 and 05:38:19), and a build that went 4s -> 306s. Re-checked
same day 07:50: exactly four pull_loops, one per commodity (KC/CC/CT/SB) — clean,
so the duplicate was TRANSIENT, not a persistent leak. Plausible mechanism, best
suspect, never demonstrated. Do not record it as the established cause.


## 2026-08-25 — OI cards: dated-contract labels + WoW/MoM (commit ac87cf7)

**Decided:** the futures table on the WhatsApp/site cards keys on the DATED
contract (`Oct '26`), derived from each row's own FND/LTD via the same rule as
`build_prelim_oi.py:contract_key()` (anchor `last_trade` for CT/CC/KC,
`first_notice` for SB — cash-settle, LTD lands the month BEFORE delivery). The
old TICKER + FUT CONT pair is gone; WoW (T-5) and MoM (T-21) columns added,
walked back in TRADING SESSIONS not calendar days.

**Why — the load-bearing insight:** generic tickers are POINTERS that re-aim at
a different real contract on roll days (~31 events/yr across CT/KC/CC/SB), so
differencing a generic subtracts two different instruments. Proven on real data:
2026-07-21 KCJUL1 moved Jul-26 -> Jul-27 and the stored `oi_chg` column reported
**+4,817 on a 4,899-lot contract**. Lou pulled Bloomberg KCN7 history (4,945 ->
4,899) which independently confirms the true flow was **-46**. DoD is therefore
now COMPUTED like WoW/MoM instead of read from `oi_chg` — verified identical to
the stored column on every non-roll session (1,166/1,167 over 30 sessions).

**TOTAL renamed "TOTAL (SHOWN)"** (Lou's choice of 3 options). A Bloomberg SB1
screenshot showed the chain carries 12 months (Aggr OI 1,222,053) vs the 8 we
display (1,210,576) — the 11,477 gap is Oct28/Mar29/May29/Jul29, which we do not
fetch. The card's total was ALWAYS a subtotal; the label now says so. Rejected:
leaving it ambiguous, and fetching the missing months (separate job, own blast
radius). **Do not "fix" the 11,477 discrepancy — it is expected.**

**Two bugs caught in preview, not in production** (this is why the render-before-
commit step exists — the 09:35 run is `pythonw.exe`, no console, so a crash is a
SILENT no-delivery): (1) `color_chg()` took `None` from a missing baseline and
threw TypeError — CT/KC had already rendered, so it would have shipped 2 cards
and no report; (2) a row with OI but BLANK dates rendered as a nameless `—` whose
OI still inflated the total — 8,624 such rows exist in the 2008-era history.
Both now guarded; rows that cannot be dated are dropped.

**Verification:** SB matched a Bloomberg screenshot 8/8 exact; 2026-07-21 replayed
(KC Jul '27 now -46, blank-OI row skipped, no crash); independent Sonnet audit
found no CRITICAL/MAJOR (it caught bug 2, which my own testing missed); layout
1040px vs 1080px canvas. Trigger/delivery untouched — same filenames, folder,
`as_of`, commodity order, freshness guard, send/post logic.

**Known-and-accepted:** a contract younger than its window shows `—` (CC Jul '28
MoM today); a total excluding such a contract is summed and marked `*`.

**Follow-up (commit 6e27575):** Lou spotted the options TICKER header floating at
the right edge instead of over its data. Cause: options DATA rows render ticker
left-aligned and P/C centered, but the header loop right-aligned all 7 labels
uniformly — worst on TICKER since its 252px column gave the label the most room
to drift. Fixed both, and replaced the options header's hardcoded widths with the
shared `OCOLS` var (they were a duplicate copy of the row widths — exactly how a
header silently drifts out of sync). Futures header already used its `FCOLS` var.

**Options panel: DELIBERATELY LEFT ALONE (Lou, 2026-08-25).** It renders
`Oct 2026` (straight from `options_oi.csv:contract_month`) while futures now
render `Oct '26`. Considered and REJECTED as not worth the change — the panel
reads correctly as-is. Do NOT "fix" this mismatch as tidy-up in a later session.
Had it been unified, the month SETS would still differ legitimately, since
options carry serial months with no future of their own (SB Jan/Sep/Nov/Dec).

## 2026-08-25 — prelim watcher "failure": blank subject, not a broken watcher (commit 95e8423)

**Worked on / completed:** Lou reported the prelim OI watcher failed on the
morning of 2026-08-25. It hadn't — Task Scheduler fired normally (last run
05:25, exit 0) and the log showed clean polls. The email itself (04:57, CSV
`PreliminaryOpenInterestFutures (6).csv` attached) had an **empty subject**, so
the `(UNSEEN SUBJECT "prelim")` search never matched it. Lesson: a healthy log
full of "no unread PRELIM OI messages" is exactly what a missed trigger looks
like — check the mailbox itself (read-only IMAP listing) before suspecting the
watcher, scheduler, creds, or Twilio.

**Recovery:** processed the email manually through the watcher's own functions
(import `prelim_oi_watcher`, target the UID): built session 2026-08-24,
reconciled 36/36 vs ICE's change column, R2 upload + WhatsApp HTTP 201, marked
read only after the successful send — identical semantics to the automated path.

**Decision — fallback trigger added (95e8423, pushed):** unread mail FROM
`thecottonkid@gmail.com` now also triggers, but ONLY if an attachment filename
starts with `PreliminaryOpenInterest` and ends `.csv`. The filename gate is
load-bearing: Lou self-sends other mail (e.g. "CCF Pipeline") that must never
start a build; verified against the real mailbox that the blank-subject prelim
matches and the CCF mail is rejected, plus a `--dry-run` with no false
triggers. **Rejected:** matching any unseen mail with any CSV (too loose) and
leaving behavior as-is with "just remember the subject" (already failed once).
Subject-token path unchanged. See [[project-prelim-oi-report]].

**Next session:** nothing pending here; first automated exercise of the
fallback will be whenever Lou next sends a subject-less prelim.

## 2026-08-18 — prelim OI TOTAL row contrast fix (commit 67d1f93)

Lou flagged the DoD/WoW/MoM figures on each commodity's TOTAL row as hard to
read against the navy `DKROW` (#2c3e50) band. Root cause: `GREEN`/`RED`
(#15803d/#c0392b) are tuned for contrast on WHITE data rows, and go muddy on
dark slate — the same problem the OI column had already solved (`OI_COL_TOT`
light-blue swap for the totals row) but that fix was never extended to the
delta colors. Added `GREEN_TOT`/`RED_TOT`/`LGRAY_TOT` (brighter variants) and
gave `cc()` an `on_dark` flag, applied only at the TOTAL row. Numbers/logic
unchanged — reconciliation still 41/41 vs ICE's own change column, 0
mismatch. Verified visually via WhatsApp render before commit. See
[[project-prelim-oi-report]] for the system this belongs to.

## 2026-08-17 — stray uncommitted files cleaned up; bad hv30/hv60 rewrite discarded

**What happened:** repo had 14 uncommitted items sitting in the working tree,
none from this session. Diagnosed and resolved each:
- 9 handoff `.md` files (BLPAPI_BACKFILL_ROUND2_*, RECON_PROBE1_*,
  ROLL_ALIGNMENT_RECON_SPEC, SESSION_BRIEF_2026-07-05_CORRECTION_SHIPPED,
  VOLFILL_*) — all dated Jul 5, all inter-agent scratch notes from the volume
  backfill program. The brief explicitly said "SHIPPED... MEMORY.md already
  carries the decision log (commit `2335003`)." **DELETED.**
- `basis_DARKFIX.png` / `carry_DARKFIX.png` (Jul 21) — standalone render
  artifacts from a dark-mode contrast fix, referenced by no code. **DELETED.**
- `vlm_signal_backfill_FRESH.csv` / `kc_signal_backfill_FRESH.csv` /
  `vlm_signal_backfill.csv.BEFORE_ICE_REBUILD.csv` (Jul 23) — `_FRESH` proved to
  be an EARLIER, less-complete draft (505 rows, blank IV/skew columns) than the
  live working file (1,499 rows, populated) — not a superior rebuild as the
  naming implied. Nothing in the repo's `.py` code references any of the three.
  **DELETED.**

**The one real issue — `vlm_signal_backfill.csv` modified in place, DISCARDED:**
working copy had ~500 historical rows (2024-07-11 → 2025-12-31) rewritten, but
ONLY in `hv30`/`hv60`/`iv_hv30_ratio`/`iv_hv30_ratio_zscore` — price/IV/spread
columns untouched. Traced the actual values: HEAD's `hv30` is a smooth,
monotone-decaying series (38.27→36.27→34.92→33.94→32.44→30.26...), textbook
30-day rolling stdev decay. The working copy matches HEAD EXACTLY through
2024-07-10 (34.9143 both), then drops 34.91→19.85 on 07-11 and **never rejoins
the old curve** — a permanent level shift, not a one-day correction that snaps
back. That is the signature of two different calcs/sources SPLICED together at
07-11, not a bug fix. No code in the repo computes this file (0 grep hits), so
the correct formula couldn't be independently re-derived to confirm which side
was right — reverted to HEAD (`git checkout HEAD --`) as the safer default
rather than commit an unverifiable rewrite. **If hv30/hv60 for pre-2026 dates
ever needs revisiting, start here — check for a lookback-window or source
change exactly at 2024-07-11 before trusting any future "fresher" rebuild.**

**Why this matters:** the live daily appender (`backfill: append EOD row`
commits, one row/day) was building on top of this uncommitted rewrite for 25
days without anyone noticing the historical section had silently diverged from
HEAD. Worth a periodic `git status` sanity check on data-signal repos with a
daily auto-appender, since new rows can mask a stale historical edit sitting
underneath.

## 2026-08-17 — ICE preliminary OI report + 3rd-cycle contract backfill

**Worked on:** a daily report over ICE's *Preliminary Open Interest - Futures* CSV
(CT/KC/CC/SB), showing DoD/WoW/MoM per contract month and per commodity, delivered
to WhatsApp. Commits `4ffbc56` (feature) and `f8f00fe` (post-audit hardening), both
pushed to origin/main.

**Completed**
- `build_prelim_oi.py` — joins the prelim CSV to official OI in `oi_data.csv`,
  emits a 2×2 PNG (VLM master palette) + xlsx. Reconciles its own DoD against ICE's
  published change column every run: **41/41 contracts agree, 0 mismatch**.
- `prelim_oi_watcher.py` — IMAP trigger. Tested end-to-end under pythonw.exe.
- Task Scheduler `VLM Prelim OI Watcher`, Mon–Fri 04:30–07:00, every 5 min.
- `oi_data.csv` backfilled +1,902 rows (179,565 → 181,467): SBMAR3/MAY3/JUL3/OCT3 +
  CTDEC3, 2025-01-02 → 2026-08-14. Backup `data/oi_data.csv.BEFORE_CYCLE3`.

**Decisions made**
- *Prelim is T+1, not "missing a baseline."* The prelim for session T publishes
  BEFORE official T exists (master fetch is 09:30), so the baseline is the PRIOR
  session, already on disk. A 5am run and a 10am run give identical numbers. Lou
  corrected an earlier wrong framing of this; do not re-introduce window shifting.
- *No unattended ICE download, ever.* `metadata/114` returns
  `"recaptchaRequired": true` and `/criteria` 409s without a token. Rejected —
  it would mean defeating an access control, robots.txt disallows
  `/report-partial/`, and the report is subscriber-only. Sanctioned route is an
  ICE Report Center subscription.
- *Contract mapping is commodity-specific*: `last_trade` for CT/CC/KC, but
  `first_notice` for SB (cash-settle — LTD falls the month BEFORE delivery).
  Using one rule for all four fails ~13 contracts. Rejected a single-field rule.
- *Partial totals sum what has a baseline and carry `*`* rather than blanking —
  nulling cotton's headline DoD over a 2-lot back month was the worse trade.
- *KC/CC 3rd cycle NOT added.* CC cycle-3 is INVALID in Bloomberg; `KCJUL3`
  ALIASES to `'KC' JUL MONTHLY 1` and would double-count OI already in the file.
- *Sugar's +10,440-lot aggregate step at 2025-01-02 accepted* (Lou's call).
  Bloomberg has no cycle-3 history before then, so it cannot be backfilled away.
  Sub-1% of SB's ~1.146M total.

**Traps found (two audit rounds — all fixed, worth remembering)**
1. Sorting the merged file by `first_notice` repositioned 161,943 of 179,529
   existing rows. The file's real order is `(date, commodity)` + `OI_TICKERS`
   INSERTION order. Splice new rows in; never re-sort.
2. A generic ticker carries NO FND/LTD (even `SBOCT1`). They live on the DATED
   contract, resolved per-date via historical `FUT_CUR_GEN_TICKER`.
3. Master fetch's session test drops holiday bars, where Bloomberg publishes
   OPEN_INT with no settle/volume. That OI-only shape is the file's own convention
   (2,550 rows / 72 dates back to 2008). Resolve their contract from the NEAREST
   populated gen, preferring the FOLLOWING one.
4. Roll-boundary collision: two generics can share an LTD and map to the same
   month (CCMAY1/CCMAY2 on 2026-05-13). Keep the lower generation.

**Next session priorities**
- Verify the first real unattended fire (2026-08-18 04:30) — `LastTaskResult: 0`
  and a PNG on WhatsApp. Nothing has fired on the schedule yet.
- Optional: a duplicate WhatsApp is possible if the process dies between a
  successful send and the IMAP mark-read. Left open — needs a pre-send marker,
  and the failure is a duplicate message, not wrong data.
- Still open from 2026-07-23: CC/SB signal appenders were never built.

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
