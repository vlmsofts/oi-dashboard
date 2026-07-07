# ICE OPTIONS — Serial Months & Expiry: AUTHORITATIVE REFERENCE

> **HARD RULE:** Whenever ANY expiry-date or serial-month-mapping question arises for
> CT / CC / KC / SB options, consult THIS file first, and the live ICE expiry links below.
> DO NOT guess, do not rely on stale sandbox notes. Fetched direct from ICE product pages
> 2026-07-07. Refresh from the source pages when in doubt.

## Source pages (spec) + expiry-detail links (per-contract dated calendars)
| Comm | Options spec page | Expiry-details link |
|---|---|---|
| **CT** Cotton No.2 | https://www.ice.com/products/1027/Cotton-No-2-Options | https://www.ice.com/products/1027/Cotton-No-2-Options/expiry |
| **CC** Cocoa | https://www.ice.com/products/8/Cocoa-Options | https://www.ice.com/products/8/Cocoa-Options/expiry |
| **KC** Coffee C | https://www.ice.com/products/14/Coffee-C-Options | https://www.ice.com/products/14/Coffee-C-Options/expiry |
| **SB** Sugar No.11 | https://www.ice.com/products/22/Sugar-No-11-Options | https://www.ice.com/products/22/Sugar-No-11-Options/expiry |
| General | — | https://www.ice.com/expiry-calendar |

**When a specific dated expiry is needed → fetch the per-commodity `/expiry` link above.**
The spec pages (this file) give the RULE; the `/expiry` links give the actual dates per contract.

---

## CT — Cotton No. 2 Options
- **Regular months:** March, May, July, October, December (H, K, N, V, Z)
- **Serial months:** **January, September, November** (F, U, X)
- **Serial → underlying future (VERBATIM):** "The underlying future for the September and
  November serial options is the December futures contract; the underlying future for the
  January serial option is the March futures contract."
  → **U→Z (Dec), X→Z (Dec), F→H (Mar).**
- **LTD:** Regular = "Last Friday preceding the first notice day for the underlying futures by
  at least 5 business days." Serial = "**Third Friday of the month in which the option expires.**"
- **Strike increment:** 1-cent, all months.
- NOTE: CT is the ONLY one of the four with a SPARSE serial set (just F/U/X). No Jun/Aug/etc.

## CC — Cocoa Options
- **Regular months:** March, May, July, September, December (H, K, N, U, Z)
- **Serial months:** **January, February, April, June, August, October, November**
  (F, G, J, **M**, Q, V, X) — near-continuous.
- **Serial → underlying (VERBATIM):** "For a serial option, the underlying future is the next
  Regular futures contract month." → each serial settles into the NEXT regular month
  (e.g. Jun→Jul(N), Jan→Mar(H), Apr→May(K), Aug→Sep(U), Oct/Nov→Dec(Z), Feb→Mar(H)).
- **LTD:** Through May-2025 = "first Friday of the preceding calendar month." **June-2025 onward
  = second Friday of the month preceding the option month**, with ≥4 trading days between option
  LTD and the future's FND. Expiry 17:00 ET; auto-exercise ≥1 tick ITM.
- **Strike increment:** $50/ton.

## KC — Coffee C Options
- **Regular months:** March, May, July, September, December (H, K, N, U, Z)
- **Serial months:** **January, February, April, June, August, October, November**
  (F, G, J, **M**, Q, V, X) — near-continuous (same set as CC).
- **Serial → underlying (VERBATIM):** "For a serial option, the underlying future is the next
  Regular futures contract month." → Jun→Jul(N), Jan→Mar(H), Apr→May(K), Aug→Sep(U), Oct/Nov→Dec(Z).
- **LTD:** "Second Friday of the calendar month preceding such regular or serial option month;
  ... minimum of four trading days between the option LTD and the future's FND." 17:00 ET; auto-exercise.
- **Strike increment:** $0.025/lb (2.5 cents).

## SB — Sugar No. 11 Options
- **Regular months:** **January, March, May, July, October** (F, H, K, N, V)
- **Serial months:** **February, April, June, August, September, November, December**
  (G, J, **M**, Q, **U**, X, **Z**) — near-continuous; NOTE Sep(U), Dec(Z) are SERIAL for sugar.
- **Serial → underlying (VERBATIM):** "For the January regular option, the March contract is the
  underlying future. For serial options, the underlying future is the next Regular futures
  contract month." → each serial settles into the next regular month (Jun→Jul(N), Feb→Mar(H),
  Apr→May(K), Aug/Sep→Oct(V), Nov/Dec→Jan-next(F)).
- **LTD:** "**15th calendar day of the month that precedes the options trading month**, or the
  first business day after the 15th if that is a weekend/Exchange holiday." 17:00 ET; auto-exercise.
- **Strike increment:** $0.25 cents at all levels.

---

## CRITICAL CORRECTIONS this reference makes to prior work (2026-07-07)
1. **JUNE (M) IS a real serial option for CC, KC, and SB** (settles into July). Earlier work
   wrongly treated CCM6/KCM6/SBM6 as "phantom" based on a stale sandbox note. They are REAL
   options with real expiries — the June serial's LTD is derivable from each commodity's rule
   above (SB: 15th of May; CC/KC: 2nd Friday of May, ≥4 biz days before Jul FND).
2. **CT serials are ONLY F/U/X** (Jan/Sep/Nov) — CT has NO Jun/Aug serials, unlike CC/KC/SB.
   So "no Apr/Jun/Aug serials" is a COTTON-only statement, NOT true for the other three.
3. **SB regular months include January and October; SB Sep and Dec are SERIALS** (not regular).
4. SBX6 (Nov sugar): Nov IS a valid SB serial (X) → NOT a phantom after all. Its LTD = 15th of Oct.
   (Re-check the 4 SBX6 rows against this — earlier "phantom" call was also wrong.)

## Serial → underlying quick table (for forward selection)
| Comm | Serial months (codes) | Rule |
|---|---|---|
| CT | F,U,X | F→H, U→Z, X→Z (explicit) |
| CC | F,G,J,M,Q,V,X | → next regular (H/K/N/U/Z) |
| KC | F,G,J,M,Q,V,X | → next regular (H/K/N/U/Z) |
| SB | G,J,M,Q,U,X,Z | → next regular (H/K/N/V/F) |

Month codes: F=Jan G=Feb H=Mar J=Apr K=May M=Jun N=Jul Q=Aug U=Sep V=Oct X=Nov Z=Dec.

---

## Derived + DATA-VALIDATED expiries for rolled-off contracts (2026-07-07)
The live `/expiry` calendars drop EXPIRED contracts (earliest 2026 row = Aug), so June-2026
serials aren't listed there. Derived from the LTD rules above and VALIDATED by solving Black-76
on the in-file settle data (call/put ATM IV agreed to <0.02 vol on every sampled date — a wrong
expiry would split them):
| code | commodity | contract | opt LTD | rule used | validation |
|---|---|---|---|---|---|
| KCM6 | KC | Jun-2026 serial | **2026-05-08** | 2nd Fri of May | IV 33.8→50.2% into expiry, parity-clean ✓ |
| CCM6 | CC | Jun-2026 serial | **2026-05-08** | 2nd Fri of May | IV 47→74.5%, parity-clean ✓ |
| SBM6 | SB | Jun-2026 serial | **2026-05-15** | 15th of May (Fri) | IV 30→40%, parity-clean ✓ |
| CTN6 | CT | Jul-2026 | **2026-06-12** | (in-file majority + Lou) | — |
| KCN6/CCN6 | KC/CC | Jul-2026 | **2026-06-12** | in-file majority | — |
| SBN6 | SB | Jul-2026 | **2026-06-15** | in-file majority | — |
For exact dates when a contract is still live, ALWAYS prefer the `/expiry` link over derivation.
