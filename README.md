# NBA_project
scraping and using NBA data

## Actual playoff participation and roster/coach continuity

Both championship scripts now support **actual prior playoff experience** as an
optional replacement for the six older playoff-roster proxies:

```bash
python3 nba_download_playoff_history.py
python3 nba_compare_playoff_experience.py --output-dir nba_playoff_experience_comparison
python3 nba_predict_champions_xgboost_2010_2026.py \
  --train-start 1983 --train-end 2002 --predict-start 2010 --predict-end 2026 \
  --include-playoff-experience --no-playoff-continuity \
  --output-dir nba_champion_predictions_actual_experience
```

`--include-playoff-experience` adds the actual experience and continuity groups;
`--no-playoff-continuity` excludes the five continuity inputs while retaining
actual player/coach history. `--no-playoff-experience` is the default and
preserves previous behavior. `--playoff-history-dir` selects the prepared cache.
Enabling actual history requires the advanced data and both selected leaders;
it automatically replaces the old first-time proxies. This prevents the old
franchise playoff-appearance count from remaining in the new model.

The 17 actual-participation inputs are prior NBA playoff games, recorded minutes,
participated series wins, Conference Finals appearances, and NBA Finals
appearances for each selected leader (ten), current-minute-weighted roster
averages of those statistics (five), the current coach's prior playoff games,
and current minutes belonging to players with no observed prior playoff games.
Roster weights use all regular-season team stints, including players who left
in-season; they are not a transaction-verified playoff-opening roster. Coach
games count head-coaching experience, not assistant-coaching assignments.
A player's experience follows their NBA identity across trades. Series/round
credit requires participation in that series, not simply membership on a team
that advanced. Conference Finals start in 1971; earlier division finals are not
relabeled. Games/minutes include those earlier NBA playoffs where covered.

The five continuity inputs measure last-season playoff-roster overlap, an
unchanged coach, and weighted prior series wins/Conference Finals/Finals over
five seasons. Overlap is `sum(min(current_regular_minutes_share,
past_playoff_minutes_share))` across shared players. Historical success is
multiplied by that overlap, a coach factor (1 unchanged, 0.5 changed), and
`0.5 ** ((years_ago - 1) / 3)`. These are fixed experimental feature definitions;
the classifier learns their effect. A completely different roster transfers
zero team success. The audit compares **all earlier playoff rosters of that
franchise**, including distant eras, and marks which fall inside the model's
five-year window. Team names/codes are still not predictive features.

Actual experience gets additive feature slots: up to 17 beyond the existing
base cap without continuity, or 22 with it. Optional depth independently adds
up to five slots. The selected columns, effective cap, actual/proxy
configuration, and command-line arguments are saved in `model_details.json`.
The ordinary cap can still limit basic statistics; the comparison freezes all
73 existing statistical inputs to isolate the experience changes.

The downloaded game/player archive covers verified playoff seasons **1954–2026**.
Earlier archive years are incomplete. Season totals from Basketball Reference
and the public playoff API improve minute precision and cross-check games;
coaching game counts provide a team-level completeness check. A few published
game-count disagreements are preserved and checked against actual player-series
tables. Missing 2022 team IDs are recovered from explicit home/away game records.
Nickname aliases, exact season age, and NBA game-verified identities prevent
name collisions. Unknown/censored history is excluded from known-history
averages, with at least 95% current minute coverage and resolved leading players
required. Counts cover NBA games, not ABA or international playoffs.

The source cache is in `nba_updated/playoff_experience_sources/`; compact
verified files are in `nba_updated/playoff_history/`. The downloader reuses
caches, makes no authentication-required requests, and waits at least six
seconds between new Basketball Reference pages. Predictors use local files only.
Its sources are [Eoin Moore's historical game archive](https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores),
[Basketball Reference](https://www.basketball-reference.com/),
[historical coaching records](https://github.com/spoonertaylor/NBA_Coaches), and
[the playoff statistics API](https://github.com/nprasad2077/nbaStats).

The controlled comparison trains 1983–2002, validates 2003–2009, and separately
backtests 2010–2026. The later period has already been explored and is not an
untouched holdout. Final results are in
`nba_playoff_experience_comparison_verified_1983_2002/REPORT.md`.
For XGBoost, actual experience improved validation correct picks from 4/7 to
5/7; adding continuity reduced it to 4/7. Backtest correct picks were 8/17,
8/17, and 7/17 respectively. Actual experience alone is the more promising
winner-ranking experiment; continuity remains available for comparison.
See the report for logistic results, top-three accuracy, MRR, and per-season
predictions. Seven validation seasons provide limited evidence.

Each enabled run exports `actual_playoff_experience.csv`,
`player_playoff_experience.csv`, `playoff_roster_similarity.csv`, and
`playoff_identity_coverage.csv`, plus the feature selection audit. The comparison
preserves original outputs and records exact feature lists and source hashes.

## Optional roster depth and controlled comparison

Both championship predictors support `--include-roster-depth` and
`--no-roster-depth` (the default). Existing predictions and feature selection
are unchanged when depth is disabled. When enabled, the scripts append up to
five eligible inputs **in addition to** the base `--max-features` limit: the
default is up to 59 base + 5 depth = 64 columns. Depth never displaces an
existing base column. Training-only coverage/variation rules still apply;
missing historical inputs are not forced into a fit.

The five inputs are:

| Feature | Calculation |
| --- | --- |
| `Third_Player_BPM` | BPM of the third player in the existing leader ranking |
| `Fourth_Player_BPM` | BPM of the fourth player in the same ranking |
| `Supporting_Cast_BPM` | Minute-weighted BPM of qualifying players ranked 3–8 |
| `Rotation_Depth` | Count of players with at least 12 team-stint minutes per team game |
| `Top_2_Minutes_Share` | Existing top-two players' team-stint minutes / all team player minutes |

The shared ordering uses BPM and the existing ten-minutes-per-team-game
threshold, with minutes/name tie-breaks and the existing minutes fallback for
remaining ranked positions. Supporting cast includes only qualifying ranked
players, so fewer than six can contribute; no qualifying support means missing.
Before 1974, PER is used only for ranking: the three BPM inputs remain missing.
Combined-team records are excluded, conflicting duplicate stints are rejected,
and missing BPM is never zero-filled. `Top_2_Minutes_Share` is an explicitly
requested optional distribution measure; old HHI/scoring-share features remain
excluded. Legacy `--no-roster` and `--no-third-player` remain compatibility
flags and do not control this separate optional group.

Use a new output directory to preserve older results:

```bash
python3 nba_predict_champions_xgboost_2010_2026.py \
  --train-start 1983 --train-end 2002 --predict-start 2003 --predict-end 2026 \
  --include-roster-depth --output-dir nba_champion_predictions_depth
python3 nba_compare_roster_depth.py --output-dir nba_roster_depth_experiments
```

The comparison refuses to overwrite an existing experiment directory. It
freezes Model A's actual saved feature list from
`nba_champion_predictions_xgboost/model_details.json`; B adds the 17
second-player statistics and six experience proxies; C adds all five depth
features. For the current baseline that is **56 / 79 / 84 columns**. Explicit
additive lists retain every baseline feature, rather than replacing basic
statistics under a shared cap. All configurations use the same XGBoost settings
and train on 1983–2002, validate on 2003–2009, and backtest on 2010–2026.
No 2010+ championship labels enter any fit. Ten individual/leave-one-out depth
ablations use validation only. The backtest was previously explored and is
not an untouched holdout.

The completed comparison is in `nba_roster_depth_experiments_1983_2002/`.
See its `REPORT.md`, `model_comparison.csv`, `season_predictions.csv`, and
`validation_depth_ablations.csv`. Adding all five depth inputs reduced
validation top-one accuracy from 4/7 to 3/7; B and C both scored 8/17 in the
backtest. No individual depth addition improved validation accuracy or MRR.
Depth therefore remains disabled by default; this is an actual measured result,
not a promised accuracy improvement.

Each enabled predictor run exports `roster_depth_features.csv` (including
third/fourth names, minutes, missingness diagnostics, and selected flags),
`roster_depth_player_rankings.csv`, and `feature_selection_audit.csv`.
`model_details.json` records actual selected columns, depth configuration,
effective cap, and command-line arguments. The experiment also exports all
1,498 available team-seasons from 1970–2026 with their depth diagnostics.
The old saved XGBoost run omitted second-player/experience groups despite
their priority; its invocation was not recorded, so the exact original flags
cannot be recovered. New runs record flags explicitly.

## Date-window search and leaderboard

```bash
python3 nba_champion_window_search.py
python3 render_champion_leaderboard.py
```

This searches **every eligible training start, training cutoff, and validation
end** between 1973 and 2025, for both logistic regression and XGBoost. At least
ten training seasons and one validation season are required. Defaults produce
**1,892 fitted models covering 28,380 date-window combinations**. The model
remains fixed after its training cutoff; it is not retrained each prediction
year. Identical fits are reused across validation end dates.

For example, one configuration trains on 1973–2009, validates on 2010–2020,
and tests the same model on 2021–2026. Another validates that fit through 2025
and tests on 2026. Each fit selects features, fits the imputer, and learns model
parameters using only its own training rows. Same-season regular-season stats
are used to predict that season. Experience uses only earlier seasons.

The leaderboard ranks by **validation** accuracy’s 95% Wilson lower bound, then
top-one accuracy, top-three accuracy and mean reciprocal rank. Sample counts
are shown: one perfect prediction does not outrank substantial historical
evidence automatically. Validation periods differ, so ranks are an exploratory
comparison rather than proof of which model generalizes best.

**2026 never affects the ranking or window selection.** Earlier “later-test”
periods can overlap other candidates’ validation periods; only 2026 is a global
holdout in the default search. Searching many windows can overfit validation,
and one final holdout season is not enough to establish reliable accuracy.

New-feature outputs are in `nba_champion_window_search_top2_ratings/`. The earlier
`nba_champion_window_search/` leaderboard is preserved as the two-player baseline.
Run the search again to produce a full leaderboard for the new feature set:

- `all_windows_leaderboard.csv`: every date combination, validation and later-test scores.
- `final_holdout_leaderboard.csv`: candidates validated through 2025, ranked without 2026 results.
- `winners_by_validation_end.csv`: the best setup selected using information available at each validation end.
- `all_season_predictions.csv`: top-three predictions and actual-champion rank for every fitted window.
- `search_manifest.json`: settings, fingerprints, counts and selected final setup.
- `leaderboard.html`: optional local report with model and validation-length filters, generated by the renderer.

Use `--min-validation-seasons 3` to require a longer validation period,
`--models xgboost` to search one algorithm, or `--workers 2` to control concurrent
fits. `--first-year`, `--last-validation-year`, and `--test-end` control date
bounds (selection ends by 2025; final test ends in 2026). The script saves each
completed fit, so the same command resumes from cache. Changed data, code, or
model settings require a new `--output-dir` to avoid mixing experiments.
The search writes to its own directory and preserves ordinary predictor outputs.

## Champion prediction with advanced stats

Both champion predictors now load the same cached regular-season advanced-stat
CSV as the MVP models:

```bash
python3 nba_download_advanced.py --start 1970 --end 2026
python3 nba_download_team_ratings.py --start 1970 --end 2026
python3 nba_predict_champions_2010_2026.py
python3 nba_predict_champions_xgboost_2010_2026.py
```

Use `--train-start`, `--train-end`, `--predict-start`, and `--predict-end` to
control each script's windows; the actual windows are saved in
`model_details.json`. Training must finish before prediction starts.
Use `--advanced-file PATH` for a specific regular-season advanced CSV, or
`--no-advanced` for the original team-stat baseline.

Advanced team features are:

| Player statistic | Team aggregation |
| --- | --- |
| PER, BPM, OBPM, DBPM, WS/48 | Average weighted by each player's minutes with that team |
| WS, OWS, DWS, VORP | Sum of player stint contributions divided by team games |
| TS% | Team PTS / (2 × (FGA + 0.44 × FTA)) |

These are roster aggregate features, not official team PER/BPM ratings.
Combined `TOT`/`2TM`/`3TM` rows are excluded; traded players contribute only
their individual team stints. No concentration, top-player-share, or other team
distribution features are added. Advanced stats are prioritized under the feature
limit alongside the leading-player features and first-time contender proxies.
Awards and postseason outcomes are excluded from the feature aggregation.

Both models also include **the two best players' own stats**. Selection ranks
players by regular-season BPM among players who logged at least 10 minutes per
team game (820 minutes in an 82-game season). Ties use minutes played, then
normalized name. Before 1974, player ranking uses PER because BPM is unavailable.
The selection metric is recorded in `advanced_stat_coverage.csv`.
If no player qualifies, the most-used player with a finite rating
is selected and the fallback is recorded in the coverage report. The same order
selects the second distinct player: qualified players first, then fallback
candidates by minutes if fewer than two qualify.

Each selected player's PER, BPM, OBPM, DBPM, WS/48, WS, VORP, TS%, USG%, games,
minutes per game, and points/rebounds/assists/steals/blocks/turnovers per game
enter the model. All stats come from that same player's stint with the team.
Selection uses regular-season information only, without MVP voting or playoff
outcomes. `top_two_player_features.csv` records both players and 34 input
stats for each team-season; `best_player_features.csv` retains the first player
for compatibility. `model_details.json` lists which features were actually used.
The default **59-feature cap** leaves room for both players, actual team
ratings, and first-time contender features. Lower `--max-features` values can exclude inputs.
`--no-best-player` excludes both players; `--no-second-player` retains only the
first. Third-player features have been removed; `--no-third-player` is retained
as a compatibility flag. `--stats-file PATH` overrides regular-season totals for their box scores.
`--no-advanced` disables advanced team and leading-player additions.

Actual team ratings are imported from the regular-season Team Summaries table
in the same public historical dataset. `nba_download_team_ratings.py` caches the
source as `team_summaries_source.csv` and writes `NBA_team_ratings_1970_2026.csv`.
**Offensive and defensive rating are team points scored/allowed per 100
possessions; net rating is offensive minus defensive rating.** These features
are independent of the minute-weighted player PER/BPM proxies above. All three
are prioritized under the feature limit and listed in `model_details.json`.
`team_rating_coverage.csv` records the joined values; missing team-seasons stop
the run. Use `--team-ratings-file PATH` to select a CSV or `--no-team-ratings` to
exclude them. To reproduce the old box-score-only baseline, also disable roster
advanced stats and experience features with the relevant exclusion flags.

First-time contender features describe a specific hypothesis about inexperience:

- Observed prior playoff-roster seasons for each of the two selected players.
- Whether the leader, or both leaders, have no observed prior experience.
- Team playoff appearances in the previous three seasons.
- A new-contender flag: no team appearance in those three seasons and no prior
  observed playoff-roster experience for the leader.

These are **proxies**, not actual player playoff appearances: player credit
means they logged regular-season minutes for a team that made the playoffs.
Traded players receive at most one credit per season. History follows players
across team moves using normalized names and approximate birth year; known
team code aliases are normalized. It uses only seasons strictly before the
season being predicted, including history before the training window. It does
not read that season's playoff result or assume the team will qualify. The
models learn the effect; no team-specific or fixed penalty is applied.
`first_time_contender_features.csv` records these inputs. `--no-first-time`
excludes them; they are also disabled if either leading-player feature group is
disabled. This targeted addition does not restore the broader roster/deep-run
experience features previously removed.

Each output directory includes `advanced_stat_coverage.csv` and lists the
advanced features actually selected in `model_details.json`. A run stops if
player stint minutes fall outside 90–110% of the team minutes in the original
totals, or any imported metric covers less than 90% of those player minutes.
The metric requirement applies only in eras where the source records it: BPM,
OBPM, DBPM, VORP, steals, and blocks from 1974; usage and turnovers from 1978.
Earlier values remain missing and are exempt from the completeness check.
`Era_BPM_Available` and `Era_TOV_USG_Available` indicate those eras when they vary
within training. The existing imputer fits its replacement values on training
data only. Unexpected missing metrics in later years still cause an error;
`Failure_Reasons` in the coverage report identifies which checks failed.
This catches incomplete advanced-stat seasons instead of silently imputing
entire teams. Results and diagnostics remain in `nba_champion_predictions/`
and `nba_champion_predictions_xgboost/`.

## MVP prediction

Install dependencies with `python3 -m pip install pandas numpy scikit-learn xgboost requests`.
Use the regular-season **totals** CSV generated by the scraper, plus advanced stats:

```bash
python3 nba_download_advanced.py --start 1970 --end 2026
python3 nba_predict_mvp_linear.py
python3 nba_predict_mvp_xgboost.py
```

Both default to training on seasons ending **1980–2022** and predicting
**2023–2026** from those seasons' regular-season stats. Override the windows with
`--train-start`, `--train-end`, `--predict-start`, and `--predict-end`.
Training must end before prediction starts. `--stats-file` selects a specific
regular-season totals CSV; `--team-file` selects the team results CSV;
`--output-dir` changes the destination.

The models use player per-game stats, shooting percentages, derived TS% and
three-point attempt rate, age, games played, team wins, and team win percentage.
Advanced features include **PER, WS, WS/48, OWS, DWS, BPM, OBPM, DBPM, VORP,
TS%, usage rate, assist rate, turnover rate, rebound rates, steal/block rates,
three-point attempt rate, and free-throw attempt rate**. These are imported
regular-season metrics; PER and BPM are not approximated from the totals.
Past MVP wins add defending MVP status, career wins, consecutive wins,
wins in the last three seasons, and
years since the last win (zero for players without a prior win). These features
let the models learn historical associations such as voter fatigue; no fatigue
penalty is imposed. Previous non-winning voting finishes are excluded from the
features: finishing second or ninth produces the same win-history features.
Stats are standardized within each season.
Same-season awards are used only for the target and evaluation;
postseason stats and team distribution features are excluded. Traded players use
the source's combined `2TM`/`3TM`/`TOT` row rather than counting each stint.
Their team wins and win percentage are weighted by games played in each stint;
unmatched teams are left missing and imputed using training data. Only wins and
losses are read from the team results, never playoff outcomes.
The source lacks player IDs, so names and age identify aggregate rows; voting
history matches on names and approximate birth year (`Season - Age`). Distinct
players with identical names and birth years, or inconsistent source ages, can
make history ambiguous.

Voting features use **strictly earlier seasons**, including award history before
1980 when available; those earlier seasons are not training examples. For a 2026
ranking, 2025 awards are treated as already known even when model fitting ended
earlier. This evaluates successive seasons with their available history, not a
forecast of all future seasons from one fixed information cutoff.

The advanced-stat downloader uses Sumitro Datta's public regular-season
[historical dataset](https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats),
filtering to NBA rows. It can fetch newer seasons from the third-party
[NBA Stats API](https://github.com/nprasad2077/nbaStats) when absent from that
dataset. Both sources are independent of the NBA organization. Source URLs and
download timestamps are saved with the data. The historical source is cached
as `nba_updated/advanced_historical_source.csv`; individual season caches are
`advanced_YEAR.csv`. Completed ranges are saved as `NBA_advanced_START_END.csv`.
Failed runs retain season caches and write `advanced_download_report.json`;
rerun to resume. `--refresh` rebuilds season files using the historical source
cache; remove that source cache first to download a newer historical snapshot.

The downloader defaults to **1970–2026**, including every intervening season.
It writes `advanced_metric_coverage_START_END.csv` to show metric availability
by year. PER and win shares are available in 1970–1973, but BPM, OBPM, DBPM,
and VORP are missing until 1974 in this source. Those values remain missing;
they are not estimated or set to zero. Extending the dataset does not change
model training windows. Champion predictors support training from 1970 onward
with the era-aware policy described above. MVP predictors still require 90%
coverage of PER/WS/BPM/VORP in every selected season and reject 1970–1973.

Both predictors require advanced stats by default. `--advanced-file PATH`
selects a particular regular-season CSV; `--no-advanced` runs the earlier
box-score baseline for comparisons. Advanced joins use season, normalized player
name, age, and team, including combined-team rows and known team aliases.
Current-source name encoding issues are normalized for matching. Imported award
columns are never passed to the models. Each model saves
`advanced_stat_coverage.csv`, with matches and per-metric availability by season.
A run stops when any training or prediction season has less than 90% coverage
for PER, WS, BPM, or VORP, so whole missing seasons cannot be silently imputed.

The regression target is **reciprocal MVP voting rank** (`1`, `1/2`, `1/3`, …),
with zero for players without MVP votes. This is a ranking proxy because this
CSV contains voting ranks rather than vote shares. Scores can fall outside
0–1 and are not probabilities. Highest score is the predicted MVP. These are
same-season retrospective predictions, not preseason forecasts. No official MVP
games/minutes eligibility rule is imposed; partial-season rankings are provisional.

Both MVP models predict an ordered top three for each season. `top3_predictions.csv`
contains the three highest-scoring players, their ranks, teams, and scores.
`predicted_mvps.csv` includes the first, second, and third picks in one row per
season. The console also displays all three picks; `player_predictions.csv`
keeps every candidate with a `Predicted_Top3` flag.

Each model writes `player_predictions.csv`, `top3_predictions.csv`, `predicted_mvps.csv`,
`feature_importance.csv`, `model_details.json`, and `historical_validation.csv`
to `nba_mvp_predictions_linear/` or `nba_mvp_predictions_xgboost/`.
Historical validation defaults to 2015 through the training end year: each season is predicted using only
earlier seasons, starting in 1980. `--validation-start` changes the first
validation season; at least ten earlier training seasons are required.
