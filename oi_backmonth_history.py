#!/usr/bin/env python3
"""
VLM — Cotton back-month open interest share, 20-year seasonal comparison
=========================================================================
Answers: is the current deferred-OI share unusual for this point in the season,
or is it just the continuation of a 20-year structural trend?

WHY THE TREND MATTERS (read this before using the output):
    Deferred OI share has plausibly been rising for two decades as foreign
    growth increasingly gets hedged on the US board. If so, comparing 2026's
    level to a 20-year MEAN just rediscovers that trend. The meaningful test
    is the RESIDUAL against the fitted trend, which this script reports.

SEASONAL ALIGNMENT:
    Years are aligned on trading days before the December contract's FND,
    NOT on calendar date. Aug 20 sits at a different point in the roll cycle
    each year; anchoring on Dec FND keeps the comparison honest.

REQUIRES: blpapi (Bloomberg Terminal running), pandas, numpy
USAGE:    python oi_backmonth_history.py --start-year 2006 --end-year 2026
"""

import argparse
import datetime as dt
import sys

import numpy as np
import pandas as pd

try:
    import blpapi
except ImportError:
    sys.exit("blpapi not installed. Run: pip install blpapi "
             "(needs Bloomberg Terminal running)")

# ---------------------------------------------------------------------------
# Contract construction
# ---------------------------------------------------------------------------
# Cotton No.2 futures months: H(Mar) K(May) N(Jul) V(Oct) Z(Dec)
# For crop year Y, the "new crop" December is Z of year Y.
# Deferred = everything past that December: H, K, N of year Y+1, plus Z of Y+1.

FRONT_MONTHS = ["Z"]              # the Dec that anchors the crop year
DEFERRED = [("H", 1), ("K", 1), ("N", 1), ("Z", 1)]  # (month code, year offset)


def ticker(month_code: str, year: int) -> str:
    """Bloomberg ticker for a cotton contract, e.g. CTZ6 Comdty."""
    yy = year % 100
    # Bloomberg uses 1-digit year for near contracts, 2-digit for far.
    # 2-digit form is unambiguous and accepted for history.
    return f"CT{month_code}{yy:02d} Comdty"


def dec_fnd(year: int) -> dt.date:
    """
    Approximate FND for the December cotton contract.
    Spec: five business days before the first business day of the month.
    Good enough for seasonal alignment (we only need day-count consistency).
    """
    first = dt.date(year, 12, 1)
    while first.weekday() >= 5:
        first += dt.timedelta(days=1)
    d, back = first, 0
    while back < 5:
        d -= dt.timedelta(days=1)
        if d.weekday() < 5:
            back += 1
    return d


# ---------------------------------------------------------------------------
# Bloomberg
# ---------------------------------------------------------------------------
def bbg_history(securities, field, start, end):
    """Pull daily history for a list of securities. Returns {sec: DataFrame}."""
    opts = blpapi.SessionOptions()
    opts.setServerHost("localhost")
    opts.setServerPort(8194)
    session = blpapi.Session(opts)
    if not session.start():
        sys.exit("Could not start Bloomberg session. Is the Terminal running?")
    if not session.openService("//blp/refdata"):
        sys.exit("Could not open //blp/refdata")

    svc = session.getService("//blp/refdata")
    out = {}

    # chunk to stay well inside request limits
    CHUNK = 25
    for i in range(0, len(securities), CHUNK):
        batch = securities[i:i + CHUNK]
        req = svc.createRequest("HistoricalDataRequest")
        for s in batch:
            req.getElement("securities").appendValue(s)
        req.getElement("fields").appendValue(field)
        req.set("startDate", start.strftime("%Y%m%d"))
        req.set("endDate", end.strftime("%Y%m%d"))
        req.set("periodicitySelection", "DAILY")
        req.set("nonTradingDayFillOption", "ACTIVE_DAYS_ONLY")
        session.sendRequest(req)

        while True:
            ev = session.nextEvent(30000)
            for msg in ev:
                if not msg.hasElement("securityData"):
                    continue
                sd = msg.getElement("securityData")
                sec = sd.getElementAsString("security")
                if sd.hasElement("securityError"):
                    out[sec] = pd.DataFrame()
                    continue
                fd = sd.getElement("fieldData")
                rows = []
                for j in range(fd.numValues()):
                    e = fd.getValueAsElement(j)
                    if e.hasElement(field):
                        rows.append({
                            "date": e.getElementAsDatetime("date").date(),
                            "oi": e.getElementAsFloat(field),
                        })
                out[sec] = pd.DataFrame(rows)
            if ev.eventType() == blpapi.Event.RESPONSE:
                break

    session.stop()
    return out


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def build_series(start_year, end_year, days_before_fnd, field="PX_OPEN_INT"):
    """For each crop year, deferred share at the seasonal anchor point."""
    secs, meta = [], {}
    for y in range(start_year, end_year + 1):
        front = ticker("Z", y)
        secs.append(front)
        meta.setdefault(y, {"front": front, "deferred": []})
        for code, off in DEFERRED:
            t = ticker(code, y + off)
            secs.append(t)
            meta[y]["deferred"].append(t)

    secs = sorted(set(secs))
    lo = dt.date(start_year, 1, 1)
    hi = min(dt.date(end_year + 1, 12, 31), dt.date.today())
    print(f"Pulling {field} for {len(secs)} contracts, {lo} to {hi} ...")
    hist = bbg_history(secs, field, lo, hi)

    rows = []
    for y in range(start_year, end_year + 1):
        anchor = dec_fnd(y)
        # step back N trading days from Dec FND
        d, n = anchor, 0
        while n < days_before_fnd:
            d -= dt.timedelta(days=1)
            if d.weekday() < 5:
                n += 1
        target = d

        def oi_on(sec):
            df = hist.get(sec)
            if df is None or df.empty:
                return np.nan
            df = df[df["date"] <= target]
            if df.empty:
                return np.nan
            # nearest prior observation, guard against stale quotes
            last = df.iloc[-1]
            if (target - last["date"]).days > 7:
                return np.nan
            return last["oi"]

        f = oi_on(meta[y]["front"])
        dfr = [oi_on(s) for s in meta[y]["deferred"]]
        dsum = np.nansum(dfr) if not all(np.isnan(x) for x in dfr) else np.nan
        if np.isnan(f) or np.isnan(dsum) or (f + dsum) == 0:
            print(f"  {y}: incomplete, skipped")
            continue
        rows.append({
            "crop_year": y,
            "anchor_date": target,
            "front_dec_oi": f,
            "deferred_oi": dsum,
            "total_oi": f + dsum,
            "deferred_share": dsum / (f + dsum),
        })
    return pd.DataFrame(rows)


def analyze(df):
    if df.empty or len(df) < 5:
        print("Not enough complete years to fit a trend.")
        return
    x = df["crop_year"].values.astype(float)
    y = df["deferred_share"].values * 100.0
    slope, intercept = np.polyfit(x, y, 1)
    df["trend_pct"] = slope * x + intercept
    df["residual_pp"] = y - df["trend_pct"]
    sd = df["residual_pp"].std(ddof=1)
    df["resid_z"] = df["residual_pp"] / sd

    pd.set_option("display.width", 160)
    show = df[["crop_year", "anchor_date", "front_dec_oi", "deferred_oi",
               "deferred_share", "trend_pct", "residual_pp", "resid_z"]].copy()
    show["deferred_share"] = (show["deferred_share"] * 100).round(1)
    for c in ["trend_pct", "residual_pp", "resid_z"]:
        show[c] = show[c].round(2)
    print("\n" + "=" * 100)
    print("DEFERRED OPEN INTEREST SHARE AT SEASONAL ANCHOR")
    print("=" * 100)
    print(show.to_string(index=False))

    print("\n" + "-" * 100)
    print(f"Trend: {slope:+.2f} percentage points per year "
          f"({'RISING' if slope > 0 else 'FALLING'} structurally)")
    print(f"Residual std dev: {sd:.2f} pp")
    cur = df.iloc[-1]
    print(f"\nMOST RECENT YEAR ({int(cur['crop_year'])}):")
    print(f"  deferred share   : {cur['deferred_share']*100:.1f}%")
    print(f"  trend expects    : {cur['trend_pct']:.1f}%")
    print(f"  residual         : {cur['residual_pp']:+.1f} pp  "
          f"({cur['resid_z']:+.2f} sd)")
    verdict = ("UNUSUAL vs its own trend" if abs(cur["resid_z"]) >= 1.5
               else "IN LINE with its own trend")
    print(f"  verdict          : {verdict}")
    print("\nNOTE: a high LEVEL with a near-zero residual means the structure "
          "kept growing, not that this year is special. Read the residual, "
          "not the level.")
    print("-" * 100)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2006)
    ap.add_argument("--end-year", type=int, default=2026)
    ap.add_argument("--days-before-fnd", type=int, default=70,
                    help="trading days before Dec FND to snapshot "
                         "(70 ~ late August)")
    ap.add_argument("--field", default="PX_OPEN_INT")
    ap.add_argument("--csv", default="cotton_deferred_oi_share.csv")
    a = ap.parse_args()

    d = build_series(a.start_year, a.end_year, a.days_before_fnd, a.field)
    if not d.empty:
        analyze(d)
        d.to_csv(a.csv, index=False)
        print(f"\nWrote {a.csv}")
