"""gta6-trophy-rarity-forecast-01 entry (Python 3.13, standard library only).

Forecasts every GTA VI PS5 trophy's earn rate on Exophase at the case's target.

Model (see README.md for the full reasoning):
  * The page's rates are counts over players tracked: rate = earners / tracked.
    Earners only grow, tracked only grows in jumps at Exophase's refreshes, so
    the program models the two counts separately and divides, instead of
    extrapolating the sawtooth rate series directly.
  * When the whole history is flat (one unchanged page, as on launch night),
    no trend can be fitted. Exophase refreshes a game's counts every few hours
    and a refresh jumps the tracked count several-fold while the newly tracked
    players have earned almost nothing yet, so the forecast is a refresh
    mixture: the old cohort's rate diluted by the jump, plus what the new
    cohort earns in its first hours. The new cohort has had only minutes of
    play, so only the first-minutes activities register with it at all: its
    earn rate is steep in the trophy's earliness, near zero for anything that
    needs hours of story or a grind.
  * When the history shows real movement, each trophy's earn rate is fitted
    as a line in logit space against log(1 + hours since the first read) on
    the RECENT window of the history, and the line is read at the target with
    the move beyond the history damped. The rate fitted is the de-sawtoothed
    one (earners over the trend of the tracked count): the raw rate dips at
    every refresh, when the tracked count jumps but the earners do not.
    Fitting the whole history would let the launch surge (tracked count
    exploding, every rate falling) dominate; the recent window is the regime
    the target lives in. The spread combines the fit residual with a random
    walk term that grows with the horizon.

Quantiles are the normal quantiles of the forecast logit, rounded to 4
decimals (half up, as the starter rounds them).
"""

import json
import math
import os
import re
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a module shipped beside this file (kept for parity with the starter).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# --- refresh-mixture prior, from the rules' own account of launch night ---
# The page sat unchanged (638 players tracked) from 23:48 to 04:00 UTC and was
# refreshed by the 04:30 read: 3,868 players, every rate but the 100% one
# lower. A refresh therefore multiplies the tracked count several-fold while
# the newly tracked players have earned almost nothing yet.
REFRESH_JUMP = 3868 / 638  # ~6.06
# New-cohort earn rate, as a fraction of the old cohort's rate, for a trophy
# of earliness e (0 = latest, 1 = earliest). The new cohort at a refresh has
# had only minutes to a few hours of play, so only the first-minutes
# activities register with it at all: the fraction is steep in earliness,
# near zero for anything that needs hours of story or a grind.
NEW_COHORT_FRACTION = 0.5


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    return math.log(clamp(pct) / (100 - clamp(pct)))


def inv_logit(lg: float) -> float:
    return 100 / (1 + math.exp(-lg))


def round4(x: float):
    """x rounded to 4 decimals, half up, as JavaScript's Math.round rounds it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def earliness(trophy: dict, last_rate: float) -> float:
    """0..1: how early in a player's first hours this trophy can be earned.

    From the case alone: the launch-night rate (more early earners means an
    earlier/easier trophy), the description (chapter order, final missions,
    grindy counts) and the tier. The new cohort at a refresh has had only
    minutes to a few hours of play, so early trophies are the ones it earns.
    """
    desc = trophy["description"].lower()
    lo, hi = 0.63, 2.04  # the launch night's observed rate range, ex-100%
    e = (last_rate - lo) / (hi - lo) if last_rate < 100 else 1.0
    if "chapter 1" in desc:
        e += 0.35
    elif "chapter 2" in desc:
        e += 0.20
    elif "chapter 3" in desc:
        e += 0.10
    elif "chapter 4" in desc:
        e += 0.05
    if "final mission" in desc:
        e -= 0.25
    if "100% completion" in desc:
        e -= 0.50
    if trophy["tier"] == "platinum":
        e -= 0.50
    if trophy["tier"] == "gold":
        e -= 0.15
    if trophy["tier"] == "silver":
        e -= 0.05
    if re.search(r"(250|100 locations|50 challenges|8 mission strands|3 collectible)", desc):
        e -= 0.30
    if ("try each type of activity" in desc or "arcade game" in desc
            or "fishing challenge" in desc or "gone fishing" in desc):
        e += 0.30
    return max(0.0, min(1.0, e))


def quantiles(median_pct: float, spread: float) -> list:
    lg = logit(median_pct)
    return [round4(inv_logit(lg + z * spread)) for z in Z]


def parse_time(text: str) -> float:
    """ISO 8601 UTC timestamp -> hours since the Unix epoch (fractional)."""
    # 2026-11-19T03:00:00Z, or with fractional seconds: ...:00.309Z
    text = text.rstrip("Z")
    date, clock = text.split("T")
    y, mo, d = (int(v) for v in date.split("-"))
    parts = clock.split(":")
    h, mi = int(parts[0]), int(parts[1])
    sec = float(parts[2])
    import datetime
    dt = datetime.datetime(y, mo, d, h, mi, tzinfo=datetime.timezone.utc) \
        + datetime.timedelta(seconds=sec)
    return dt.timestamp() / 3600.0


def least_squares(xs, ys):
    """Slope and intercept of y ~ a + b*x; returns (a, b, resid_sd)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    sd = math.sqrt(sum(r * r for r in resid) / max(1, n - 2)) if n > 2 else 0.0
    return a, b, sd


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0
    last = history[-1]
    last_rates = [r if r is not None else 0.0 for r in last["rates"]]

    # Is the history flat? One unchanged page (same players, same rates).
    seen = set()
    for read in history:
        seen.add((read["players_tracked"], tuple(read["rates"])))
    flat = len(seen) <= 1

    rows = []
    if flat:
        # --- refresh-mixture path: the page is stale, a refresh is due ---
        # Median: the old cohort's rate diluted by the jump, plus what the
        # new cohort earns in its first hours (more for early trophies).
        # Spread: the refresh's size and timing are uncertain, so the band is
        # wider than the residual of the median alone, but not so wide that it
        # pays at every trophy.
        spread = 0.22
        for i, trophy in enumerate(trophies):
            r = last_rates[i]
            if r >= 100:
                rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
                continue
            f = NEW_COHORT_FRACTION * earliness(trophy, r) ** 2
            median = r * (1 + (REFRESH_JUMP - 1) * f) / REFRESH_JUMP
            rows.append({"id": trophy["id"], "q": quantiles(median, spread)})
        return {"trophies": rows}

    # --- trend path: logit(rate) linear in log time, on the recent window ---
    # Fit the de-sawtoothed rate: earners over the TREND of the tracked count
    # (the raw rate dips at every refresh, when the tracked count jumps but
    # the earners do not; dividing by the trend removes the sawtooth while
    # keeping the counts' information). The logit form saturates at 100%, so
    # a fast-rising trophy cannot extrapolate past it the way separate
    # power laws for earners and tracked can.
    t0 = parse_time(history[0]["at"])
    target_h = parse_time(case["target"])
    # Distinct reads only: flat stretches between refreshes carry no signal.
    distinct = []
    for read in history:
        if not distinct or (read["players_tracked"], tuple(read["rates"])) != \
                (distinct[-1]["players_tracked"], tuple(distinct[-1]["rates"])):
            distinct.append(read)
    xs = [math.log1p(parse_time(r["at"]) - t0) for r in distinct]
    x_target = math.log1p(target_h - t0)
    x_max = xs[-1]
    # Fit on the RECENT window only (the last 40% of the history's log-time
    # span). Early on, the tracked count explodes with the launch surge and
    # every rate falls; the launch regime does not extrapolate. The recent
    # window is the regime the target lives in: the surge spent, players
    # progressing, rates rising.
    x_cut = x_max - 0.40 * (x_max - xs[0])
    recent = [k for k in range(len(xs)) if xs[k] >= x_cut]
    if len(recent) < 3:  # too little history to window: use it all
        recent = list(range(len(xs)))
    fx = [xs[k] for k in recent]
    # Damping beyond the observed range: growth slows, so shrink the
    # extrapolated move the further past the history the target lies.
    over = max(0.0, x_target - x_max)
    damp = 1.0 / (1.0 + 0.20 * over)

    # Tracked-count trend (log N vs log time) on the recent window.
    log_n = [math.log(max(1, distinct[k]["players_tracked"])) for k in recent]
    a_n, b_n, _ = least_squares(fx, log_n)

    for i, trophy in enumerate(trophies):
        r_last = last_rates[i]
        if r_last >= 100:
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue
        # De-sawtoothed logit rate on the recent window.
        ys = []
        for k in recent:
            r = distinct[k]
            rate = r["rates"][i]
            earners = max(0.5, (rate if rate is not None else 0.0)
                          * r["players_tracked"] / 100.0)
            n_trend = math.exp(a_n + b_n * xs[k])
            ys.append(logit(100.0 * earners / n_trend))
        a_r, b_r, sd_r = least_squares(fx, ys)
        median = clamp(inv_logit(a_r + b_r * (x_max + damp * over)))
        # Spread: the fit's residual plus a horizon term for trend-bending and
        # refresh-timing noise at the target, floored so a too-clean fit does
        # not claim false precision, with a penalty when the recent window
        # holds so few reads that the residual says nothing (a line through
        # two points fits exactly). (The starter's 0.5-per-root-day random
        # walk is far too wide once the median is a real trend: it pays at
        # every trophy.)
        small = 0.30 if len(recent) < 5 else 0.0
        spread = math.sqrt(max(0.25, sd_r) ** 2 + (0.22 * math.sqrt(horizon_days)) ** 2 + small ** 2)
        rows.append({"id": trophy["id"], "q": quantiles(median, spread)})
    return {"trophies": rows}


def fallback_forecast(case: dict) -> dict:
    """Last-resort forecast: every quantile equals the trophy's last rate.

    Used only if the model fails on a case, so one bad case cannot take down
    the run's other cases. It scores exactly as an invalid case would (the
    rules score an invalid case as its last rates), but the other cases stay
    valid.
    """
    rows = []
    history = case.get("history") or []
    last_rates = history[-1]["rates"] if history else []
    for i, trophy in enumerate(case["trophies"]):
        rate = last_rates[i] if i < len(last_rates) and last_rates[i] is not None else 1.0
        rate = clamp(rate)
        rows.append({"id": trophy["id"], "q": [round4(rate)] * 5})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case, path.name)
        except Exception:
            forecast = fallback_forecast(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
