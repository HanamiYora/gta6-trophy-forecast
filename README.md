# GTA VI trophy rarity forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/HanamiYora/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/HanamiYora/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every Grand Theft Auto VI (PS5) trophy's earn rate on Exophase at
the case's target. `main.py` (Python 3.13, standard library only) reads every
`<id>.in` in `$MAKEFUN_INPUT_DIR` (else `/input/`) and writes `<id>.out` to
`$MAKEFUN_OUTPUT_DIR` (else `/output/`): one row per trophy, five quantiles
(5%, 25%, 50%, 75%, 95%) of the earn rate in percent, non-decreasing.

## The mechanism the model is built on

A page rate is a ratio: **earners / players tracked**. Earners only grow.
Players tracked only grows, and only in jumps, when Exophase refreshes a
game's counts. So a rate falls at a refresh (the denominator jumps while the
numerator does not) and climbs between refreshes (players earn trophies).
Any model that extrapolates the rate series directly is fitting a sawtooth.
This program models the two counts separately and divides.

Two regimes, chosen from the case's own history:

**1. Flat history — the refresh mixture.** When every read shows the same
page (launch night: 638 players tracked, rates unchanged for hours), no trend
can be fitted. But a refresh is due: the rules record the page unchanged from
23:48 to 04:00 UTC, then refreshed by the 04:30 read — 3,868 players tracked,
every rate but the 100% one lower. The forecast is therefore a mixture, not
persistence: the old cohort's rate diluted by the jump (×6.06), plus what the
new cohort earns in its first hours. The new cohort has had only minutes of
play, so only the first-minutes activities register with it at all: its earn
rate is steep in the trophy's earliness (launch-night rate, chapter order,
final missions, grindy counts, tier), near zero for anything that needs hours
of story or a grind. The 100% trophy stays at 100%. Spread 0.22 logit units,
reflecting that the refresh's size and timing are uncertain.

**2. Moving history — the trend.** When the reads move, each trophy's rate is
fitted as a line in logit space against log(1 + hours since the first read),
on the recent window of the history (the last 40% of its log-time span), and
the line is read at the target with the move beyond the history damped.
The rate fitted is the de-sawtoothed one: earners over the *trend* of the
tracked count, so the refresh dips do not bias the fit. Fitting the whole
history would let the launch surge (tracked count exploding, every rate
falling) dominate the fit; the recent window is the regime the target lives
in — the surge spent, players progressing, rates rising. The logit form
saturates at 100%, so a fast-rising trophy cannot extrapolate past it.
Spread: the fit's residual plus a horizon term for trend-bending and
refresh-timing noise at the target (0.22 per root-day, not the starter's
0.5 — that width pays at every trophy once the median is a real trend),
floored at 0.25, with a penalty when the recent window holds too few reads
to price its own residual.

A case the model cannot handle falls back to its last rates (which is what
an invalid case scores as), so one bad case cannot take down the run's
other cases. The program is deterministic: fixed file order, fixed rounding
(4 decimals, half up), no randomness, no network, no model calls.

## Local validation

There is no local grade for this competition: the server grades every entry.
What could be checked locally was checked:

- **Practice (p00, p01).** The starter scores ~1.51 on the practice (rules).
  The practice answers are the 04:30 refresh, and the live Exophase page
  fetched during the build still showed that refresh — the starter scored
  against it comes out at 1.5084 with a typical miss of 0.677 and 90%
  coverage of 0.00, matching the rules' quoted ~1.51 / ~0.68 / ~0 to three
  decimals. Against those same rates this entry scores **0.084** (typical
  miss 0.024 points, 90% coverage 0.96). Both practice cases give the same
  forecast: their histories are the same flat page and their answers are the
  same refreshed page.
- **Final-like simulation.** A synthetic final case (2.5 days of 10-minute
  reads, refresh sawtooth, rates falling with the launch surge then rising
  as players progress, 4.5-day horizon) was built to test the trend path,
  with the target rates set from the trophies' earliness and the anchors
  matched to the observed launch-night data. Across three parameter settings
  this entry scores **0.12–0.16** against the starter's 0.53–0.64, and
  **0.11–0.14** against the starter's 0.41–0.56 when the truth is made to
  rise slower than the fitted trend (saturation). Typical miss 0.6–3.3
  points against the starter's 9–14.
- **Determinism and runtime.** Two runs on the practice cases produce
  byte-identical outputs; a run over both cases takes ~0.07 s (limit 8 s
  per case). Output shape validated: exactly one row per trophy, five
  non-decreasing quantiles in [0, 100], under the 65,536-byte cap.

## Files

- `main.py` — the entry (the only file the grader runs).
- `README.md` — this file.
- `LICENSE` — MIT.

---

<sub>Entered on [make.fun](https://make.fun/?ref=wy5270x8&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
