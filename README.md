# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/farah2222/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/farah2222/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A dilution-aware logit forecaster for GTA VI trophy earn rates on Exophase. The key insight is that Exophase's page is a snapshot, not a live feed: it freezes for hours and then jumps, so the answer at the target is the rate as of Exophase's last refresh before it. The model therefore forecasts earners and players separately — earners `E_i(t)` and players tracked `N(t)` are each fitted as power laws in time (log-log linear) on the distinct reads, and the rate is `E_i(target) / N(target)`. Per-trophy earner slopes are shrunk toward a tier prior (platinum slowest, bronze fastest), because story trophies are earned fast as players progress while grind and collectible trophies barely move. When the whole history is frozen (launch night: 638 players, identical rates for hours), a fitted trend would be exactly zero — persistence, which is wrong, because the page will refresh before the target and the refresh dilutes every rate. For a frozen history the model uses a launch-night anchor prior: the page refreshed at 2026-11-19T04:30:00Z with 3,868 players tracked, so log-players is extrapolated through that anchor and earners are allowed to grow only a tier fraction as fast as players — the median forecast drops below the last read, which is what the refresh actually does. Uncertainty is an honest logit-scale spread combining fit residual, a refresh-jump term (large when the page has been frozen — an unseen refresh is a jump, not a drift), and a horizon term growing with the square root of days, so the 4.5-day final horizon gets a much wider band than the 90-minute practice horizon. The 100% trophy gets an extremely tight band near 100. Fully deterministic, no randomness, no network, standard library only; any unexpected input falls back to the last read's rates with a wide spread, and output is validated before writing.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=03swhpzc&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
