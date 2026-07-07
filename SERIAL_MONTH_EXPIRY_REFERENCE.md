# Serial-Month & Options-Expiry Reference — CT / CC / KC / SB

> Built 2026-07-07 from `Desktop/options sandbox/DOCS_SANDBOX/` (authoritative ICE
> product specs + rules). This is the reference behind the **serial-month IV blank**
> defect in `options_oi.csv` (Sep/Nov/Jan tenors have settle prices but no
> `expire_dt`/`days_to_exp`/`iv_pct`). Source-of-truth files:
> - `ICE_Cotton_No2_Rules_KEY_DATES.md` — CT Rules Ch.10 (10.51 serial authority, 10.54 LTD formula)
> - `ICE_Expiry_Dates_ALL_COMMODITIES.md` — FND/LTD/FDD tables, all 4 softs
> - `SANDBOX_GUIDE.md` §Contract Specifications (serial maps per commodity)
> - `ProductSpecExpiryDates*.csv` — per-contract OPTIONS LTD (incl. serials)

---

## THE DEFECT (why this doc exists)

`options_oi.csv` populates `expire_dt`/`days_to_exp`/`iv_pct` **only for tenors that
map to a listed FUTURE** (H/K/N/V/Z quarterlies). **Serial option months
(Jan=F, Sep=U, Nov=X) have no own future**, so `contract_dates.py`'s `_D` builder —
which walks the FUTURES records and attaches a matching option — **drops them**.
No expiry → no DTE → the Black-76 solver in `vlm_master_fetch.py` skips IV/greeks.

**Key fact: the serial option expiries ARE already in `contract_expiries.json`**
(as OPTION records: CTU26 OPT_LTD 2026-08-21, CTX26 2026-10-16, CTF27 2026-12-18…).
They are present but unused because `_D` is keyed on the futures code. The fix is to
carry option-only codes into `_D` (additive; see plan below), NOT to fetch new data.

---

## SERIAL-MONTH MAPS (which option months exist, and their underlying future)

### CT — Cotton No. 2  (Rule 10.51, verified vs SANDBOX_GUIDE:745)
Futures months: **H=Mar, K=May, N=Jul, V=Oct, Z=Dec** (Oct excluded by house rule everywhere).
Option months = the 5 futures months **PLUS serials**:

| Option month | Code | Underlying future | Type |
|---|---|---|---|
| Jan | F | Mar (H) | **serial** → `F→H` |
| Feb | G | Mar (H) | serial (rarely listed) → `G→H` |
| Mar | H | Mar (H) | quarterly |
| May | K | May (K) | quarterly |
| Jul | N | Jul (N) | quarterly |
| **Sep** | **U** | **Dec (Z)** | **serial** → `U→Z` |
| Oct | V | Oct (V) | quarterly (house-excluded) |
| **Nov** | **X** | **Dec (Z)** | **serial** → `X→Z` |
| Dec | Z | Dec (Z) | quarterly |

`_CT_SERIAL_FWD` (engine): **U→Z, X→Z, F→H** (and G→H). NO Apr/Jun/Aug serials.

### CC — Cocoa
Futures months: **H, K, N, U, Z** (Mar, May, Jul, **Sep**, Dec — Sep IS a real future here).
Serial options: Jan→Mar, Nov→Dec (and the standard prev-month serials). Sep is a quarterly, not a serial.

### KC — Coffee C
Futures months: **H, K, N, U, Z** (Mar, May, Jul, **Sep**, Dec — Sep is a real future).
Serial options: **Jan→Mar, Feb→Mar, Sep→Dec, Nov→Dec** (SANDBOX_GUIDE:753). Note KC also lists
Aug/Oct serial options in some ProductSpec pulls (Aug26 OPT_LTD 07/10, Oct26 09/11).

### SB — Sugar No. 11
Futures months: **H, K, N, V** (Mar, May, Jul, Oct — **NO December** future/option, special rule).
FND pattern is different (FND ≈ FDD ≈ day after LTD — cash-settle style). Options LTD = 15th of prior month.

---

## OPTIONS LTD (expiry) FORMULA — CT, Rule 10.54

> Option LTD = **the last Friday which precedes the FND of the underlying future by
> at least 5 business days.**

Special cases (Dec-future serials): 
- Dec-future option expiring prev **Aug** (U/Sep-adjacent) → 3rd Friday of prev August
- Dec-future option expiring prev **Oct** → 3rd Friday of prev October
- Mar-future option expiring prev **Dec** (Jan serial) → 3rd Friday of prev December

FND itself (CT, Rule 10.02): **FND = First Delivery Day − 5 US business days**, where
FDD = 1st business day of the delivery month. Holiday-aware (9 US federal holidays,
Sat→Fri / Sun→Mon observance). Verified 100% vs ProductSpec CSV for all 2026–2029 contracts.
**FND↔FDD gap is COMMODITY-SPECIFIC — do NOT reuse CT's derivation for CC/KC/SB.**

---

## CT OPTIONS LTD TABLE (from ProductSpecExpiryDates.csv — the CT options file)

Serials in **bold**. This is the ground truth `iv_pct` needs.

| Contract | OPTIONS LTD | Serial? | Underlying |
|---|---|---|---|
| Jul 2026 | 06/12/2026 | | N |
| **Sep 2026** | **08/21/2026** | **✔ serial** | Z (Dec) |
| Oct 2026 | 09/11/2026 | | V |
| **Nov 2026** | **10/16/2026** | **✔ serial** | Z (Dec) |
| Dec 2026 | 11/13/2026 | | Z |
| **Jan 2027** | **12/18/2026** | **✔ serial** | H (Mar) |
| Mar 2027 | 02/05/2027 | | H |
| May 2027 | 04/16/2027 | | K |
| Jul 2027 | 06/11/2027 | | N |
| **Sep 2027** | **08/20/2027** | **✔ serial** | Z (Dec) |
| Oct 2027 | 09/10/2027 | | V |
| **Nov 2027** | **10/15/2027** | **✔ serial** | Z (Dec) |
| Dec 2027 | 11/12/2027 | | Z |
| **Jan 2028** | **12/17/2027** | **✔ serial** | H (Mar) |
| Mar 2028 | 02/11/2028 | | H |
| May 2028 | 04/13/2028 | | K |
| Jul 2028 | 06/09/2028 | | N |

These match `contract_expiries.json` OPT_LTD exactly (CTU26=2026-08-21, CTX26=2026-10-16,
CTF27=2026-12-18, CTU27=2027-08-20, CTX27=2027-10-15). **Data is present; only the
`_D` build loop drops it.**

---

## WHY SEP MATTERS (the analytical need)

The **Sep serial** (hooked on Dec, LTD ~08/21) is the only short tenor giving a genuine
**~30-day ATM implied vol** for cotton. October (V, ~70 DTE) is the only other short tenor
with IV currently populated — and **Oct is house-excluded from all cotton scoring**. So with
Sep blank, there is **no usable ~30d IV in the file**. Sandbox engine ATM-IV for that tenor ≈ **22.7%**.

---

## CONTRACT SPEC QUICK-REF (all commodities)

| Comm | Futures months | Codes | Opt LTD rule | Strike grid | Settle time |
|---|---|---|---|---|---|
| CT | Mar,May,Jul,Oct,Dec | H,K,N,V,Z | 2nd Fri prior mo (Rule 10.54) | 1.00¢ | ~16:00–16:30 ET |
| KC | Mar,May,Jul,Sep,Dec | H,K,N,U,Z | 2nd Fri prior mo | 2.5¢ | ~12:25 ET |
| CC | Mar,May,Jul,Sep,Dec | H,K,N,U,Z | 2nd Fri prior mo | $1.00/MT | — |
| SB | Mar,May,Jul,Oct,Dec* | H,K,N,V | 15th of prior mo | 0.01/lb | — |

*SB has **no December** option/future (special rule).

Month codes: F=Jan G=Feb H=Mar J=Apr K=May M=Jun N=Jul Q=Aug U=Sep V=Oct X=Nov Z=Dec.

---

## FIX DIRECTION (for contract_dates.py — this repo)

`_build_calendar_from_json` (contract_dates.py:106) walks FUTURES and attaches options
by code match. **After that walk, add any option code not already in `_D` as an
option-only entry** (`fnd`/`ltd` = None, `opt_exp` = OPT_LTD). Additive → cannot change
any existing quarterly entry. Must confirm option-only codes do NOT leak into
`_calendar_for` / `resolve_generic_to_ice` slot resolution (no CT consumer generic
requests U/X/F, so they shouldn't — verify before ship). The daily job only needs
`get_opt_exp` for these; None `fnd`/`ltd` is correct (serials have no futures FND/LTD).

**Blast radius: `contract_dates.py` is imported by BOTH this dashboard AND
`VLM Data/vlm_master_fetch.py` (the daily job).** Shared contract → STOP-and-confirm
before shipping (per house blast-radius rule).
