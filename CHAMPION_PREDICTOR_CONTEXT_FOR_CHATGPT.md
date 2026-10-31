# NBA championship predictor — context for ChatGPT

This document describes the existing NBA championship prediction project, checked against the local code and saved model metadata on 2026-10-29. Use it as context when discussing improvements. It covers the championship predictors; the project also has separate MVP predictors with different targets and features.

## 1. Goal and models

The goal is to rank NBA teams by their likelihood of winning the championship using that season's regular-season statistics, plus earlier playoff experience. The highest-scoring team is the predicted champion; the first three teams form the predicted top three.

There are two championship scripts:

| Script | Model |
| --- | --- |
| `nba_predict_champions_2010_2026.py` | Logistic regression, a binary classifier |
| `nba_predict_champions_xgboost_2010_2026.py` | XGBoost gradient-boosted decision tree classifier |

Despite the historical filenames, command-line arguments control the actual years. The first model is logistic regression, not ordinary linear regression. No neural network is currently implemented.

Each training row represents one team in one season. Champions receive target 1; all other teams receive target 0. Both champions and non-champions are used for training.

These are same-season predictions/backtests using regular-season information. They are not preseason forecasts: predicting 2025 uses 2025 regular-season statistics.

## 2. Data inputs

- `nba_champions_analysis/team_metrics_all_seasons.csv`: team metrics aggregated from regular-season player totals.
- `nba_updated/NBA_team_seasons_1947_2027.csv`: season/team results, wins and losses, championship labels, and historical playoff qualification.
- A regular-season player totals CSV in `nba_updated/`: selected players' box scores and historical roster membership. The scripts can auto-detect a file or accept `--stats-file`.
- `nba_updated/NBA_advanced_1970_2026.csv`: imported player advanced statistics, including individual team stints.
- `nba_updated/NBA_team_ratings_1970_2026.csv`: actual regular-season team offensive, defensive, and net ratings.
- `nba_champions_analysis/champions_vs_league_all_stats.csv`, when available: an additional source of historical championship labels, checked for conflicts.

Historical advanced data and team ratings are cached from the public NBA/ABA/BAA dataset maintained by Sumitro Datta; the downloader filters NBA records. The player advanced-stat downloader also supports a third-party API fallback for newer seasons. A file's presence or year range does not independently prove every value or outcome is correct; source accuracy still matters.

## 3. Supported input groups

The following groups are enabled by default in both scripts. A particular run may disable groups or select fewer columns. Always check that run's `model_details.json`, especially its `features` list, before claiming a statistic was used.

### A. Regular-season wins and basic team statistics

`Win_Percentage = Wins / (Wins + Losses)` is explicitly prioritized. Raw wins and losses are not separate model inputs.

Other eligible team columns include:

- Field goals made/attempted and field-goal percentage: `FG`, `FGA`, `FG%`.
- Two-point and three-point makes, attempts, and percentages: `2P`, `2PA`, `2P%`, `3P`, `3PA`, `3P%`.
- Free throws made/attempted and percentage: `FT`, `FTA`, `FT%`.
- Effective field-goal percentage: `eFG%`.
- Points, total/offensive/defensive rebounds, assists, steals, blocks, turnovers, personal fouls, minutes, and triple-doubles: `PTS`, `TRB`, `ORB`, `DRB`, `AST`, `STL`, `BLK`, `TOV`, `PF`, `MP`, `Trp-Dbl`.
- Roster age (`Age`, weighted by player minutes), and `GS` if present and usable.

Counting statistics are generally aggregated across player stints and divided by team games. Shooting percentages are rebuilt from makes and attempts, rather than averaging player percentages. Other recognized legacy rate columns can also be eligible if present in the team metrics file.

Eligibility does not guarantee selection: advanced stats and player features have higher priority under the feature cap. For example, three-point percentage can be available in the source without entering a particular fitted model.

### B. Actual team efficiency ratings — three features

| Input | Meaning |
| --- | --- |
| `Team_Offensive_Rating` | Team points scored per 100 possessions |
| `Team_Defensive_Rating` | Team points allowed per 100 possessions |
| `Team_Net_Rating` | Offensive rating minus defensive rating |

These come from a team ratings table. They are distinct from aggregating individual players' ratings.

### C. Advanced roster statistics — ten statistical features

| Statistic | Aggregation |
| --- | --- |
| PER | Player Efficiency Rating, averaged with player minutes as weights |
| BPM | Box Plus/Minus, minute-weighted average |
| OBPM | Offensive BPM, minute-weighted average |
| DBPM | Defensive BPM, minute-weighted average |
| WS/48 | Win Shares per 48 minutes, minute-weighted average |
| WS | Sum of player Win Shares, divided by team games |
| OWS | Sum of offensive Win Shares, divided by team games |
| DWS | Sum of defensive Win Shares, divided by team games |
| VORP | Sum of Value Over Replacement Player, divided by team games |
| Team TS% | Team points / [2 × (field-goal attempts + 0.44 × free-throw attempts)] |

The minute-weighted ratings are roster summaries, not official team PER or BPM measures. For traded players, only the player's stint with that team contributes. Combined-team rows such as `TOT`, `2TM`, and `3TM` are excluded to avoid double counting.

Two additional era indicators can be selected when they vary during training: `Era_BPM_Available` (1974 onward) and `Era_TOV_USG_Available` (1978 onward).

### D. The two best players' own statistics — 17 features each

The model can use the first and second selected player's individual statistics separately, not just their contribution to team aggregates. There is no full third-player statistics group. Optional roster depth adds only the third/fourth players' BPM, as described below.

Selection is recalculated for every team-season:

1. Rank players with a finite regular-season BPM who played at least 10 minutes per team game (820 minutes over 82 team games).
2. Order qualifying players by BPM, then total minutes, then normalized name.
3. Before 1974, use PER instead of BPM because BPM is unavailable.
4. If fewer than two players qualify, fill remaining positions from players with a finite selection metric, ordered by minutes.

This is an operational definition of “best player,” not a manual choice or an MVP-vote ranking. Each player's inputs come from that player's stint with the selected team.

For **each** selected player, include:

- PER, BPM, OBPM, DBPM, WS/48, WS, VORP, TS%, and USG% (usage percentage).
- Points, total rebounds, assists, steals, blocks, and turnovers per player game.
- Games played and minutes per player game.

Columns have `Best_Player_` or `Second_Player_` prefixes. Names, ages used for history matching, and selection diagnostics are recorded for auditing; player identity itself is not a model feature. Availability here means games and minutes, not a medical injury forecast.

### E. Prior playoff experience and first-time contender proxies — six features

| Feature | Meaning |
| --- | --- |
| `Prior_Best_Player_Playoff_Seasons_Proxy` | Earlier seasons in which the leading player logged regular-season minutes for a playoff team |
| `Prior_Second_Player_Playoff_Seasons_Proxy` | Same measure for the second player |
| `Best_Player_First_Playoffs_Proxy` | Leading player has zero observed prior playoff-roster seasons |
| `Both_Leaders_First_Playoffs_Proxy` | Both selected players have zero observed prior playoff-roster seasons |
| `Prior_Team_Playoff_Appearances_Last3` | Team playoff appearances in the previous three seasons |
| `New_Contender_Proxy` | No team playoff appearance in the previous three seasons AND zero observed prior playoff-roster seasons for its leading player |

These are proxies: roster membership on a playoff team does not prove the player participated in playoff games. They do not measure playoff minutes, rounds won, Finals appearances, or championships won.

Player history follows players across team moves using normalized names and approximate birth year (`Season - Age`). A player receives at most one playoff-roster credit per season, even if traded between playoff teams. Known team-code aliases are normalized.

Only seasons strictly before the predicted season contribute. History can begin before the model's training window, subject to available records. Thus an experienced player such as Curry or LeBron can retain experience credit after moving teams. No player receives a manually assigned reputation bonus, and no inexperienced team receives a fixed penalty: the model learns associations from training data.

## 4. Feature selection and preprocessing

The default cap is **59 features**, not 59 trees or 59 training seasons. Selection uses a fixed priority scheme and training-data coverage, not an optimization that searches for the most predictive subset.

Priority is: win percentage; actual team ratings; experience proxies; advanced roster and selected-player statistics; then other eligible team statistics. Within otherwise equal priorities, training coverage and original column order determine selection.

By default, a column needs at least 35% nonmissing training coverage, more than ten observed training values, and nonzero variation. Columns with no usable variation after normalization are removed. Constant era indicators therefore do not enter runs entirely within one era.

Each selected statistic is converted to a within-season z-score: `(team value - that season's league mean) / that season's league standard deviation`. Selected-player statistics and experience measures also undergo this transformation. Era indicators are preserved as raw indicators. This compares teams with contemporaries and reduces differences in statistical scale across eras. Computing a prediction season's league mean uses that season's regular-season inputs, not playoff outcomes.

Missing values are filled using medians learned from training rows only. Logistic regression then applies an additional scaler fitted on training rows. XGBoost uses median imputation without that additional scaler.

## 5. Training and prediction chronology

The ordinary scripts fit one model on the configured training window and reuse it for all requested prediction seasons. They do not automatically retrain annually.

For example, training on 1980–2009 and predicting 2010–2026 means a 2025 prediction uses:

- Model parameters learned from 1980–2009 champions and non-champions.
- Each team's 2025 regular-season statistics and selected players.
- Observed playoff-roster/team history through 2024.

The 2025 prediction does not add 2010–2024 championship outcomes to model training. Historical experience is nevertheless updated through 2024 because those facts would be available by 2025. This is a sequence of same-season predictions with a fixed fitted model, not a single forecast made in 2009 of all future seasons.

Current script defaults are:

| Model | Training seasons | Prediction seasons |
| --- | --- | --- |
| Logistic regression | 1980–2023 | 2024–2026 |
| XGBoost | 1983–2002 | 2003–2026 |

Dates are inclusive and can be overridden. Training must end before prediction starts. At least ten eligible training seasons are required.

## 6. Model settings and scores

Logistic regression uses `C=0.20`, balanced class weights, `max_iter=3000`, and random seed 42.

XGBoost defaults to 240 trees, maximum depth 2, learning rate 0.04, minimum child weight 3, row and column sampling of 0.85, L2 regularization 7, L1 regularization 0.1, and random seed 42. Positive-class weighting is the number of non-champion training rows divided by champion rows. Tree count, depth, and learning rate can be changed by arguments.

Both models score teams independently as binary examples, then rank teams within each season. The top three means the three highest-ranked championship candidates, not a prediction of the exact playoff finishing order. Scores are not calibrated title probabilities and need not sum to one across teams.

Evaluation reports whether the actual champion was ranked first and whether it appeared in the top three. Logistic coefficients and XGBoost feature importances are exported; they show model associations, not proof of causation.

## 7. Data quality and historical availability

- Advanced data extends from 1970 through 2026, but not every metric existed throughout that period.
- BPM, OBPM, DBPM, VORP, steals, and blocks are treated as unavailable before 1974; turnovers and usage percentage before 1978.
- Known unavailable historical values stay missing until training-only imputation; they are not invented or set to zero.
- Advanced player minutes must cover 90–110% of expected team minutes. Each required aggregate metric must cover at least 90% of player minutes in an era where it exists. Required selected-player values are also checked.
- Missing actual team ratings, ambiguous joins, inconsistent net ratings, and unexpected advanced-stat gaps stop the run and produce coverage diagnostics.
- Season eligibility requires at least 75% team coverage by default. Training seasons must contain their known champion. Prediction seasons marked “in progress” are excluded.

## 8. Date-window search and leaderboard

`nba_champion_window_search.py` explores eligible training start years, training cutoffs, and validation end years between 1973 and 2025, with later seasons used for testing. Defaults require ten training seasons and one validation season; testing extends through 2026.

For example: train 1973–2009, validate 2010–2020, then test 2021–2026. Each fitted model stays fixed after its training cutoff. Identical fits are reused across validation end dates.

The default grid contains 1,892 fits across both algorithms and 28,380 date-window combinations. Each fit selects features and learns imputation and model parameters from its own training rows.

Leaderboard order uses validation accuracy's 95% Wilson lower bound, then top-one accuracy, top-three accuracy, mean reciprocal rank, and validation sample count. Later-test results do not determine ranking. In the default search, 2026 is excluded from all validation and selection; earlier later-test years may overlap other candidates' validation periods.

Searching many windows can overfit validation. Different validation periods are not directly equivalent, and one final holdout season is weak evidence of generalization. The older leaderboard directory may describe an earlier feature set. The current default output directory is `nba_champion_window_search_top2_ratings/`; do not assume the entire updated search has been run merely because the script supports it.

## 9. Latest saved runs versus supported defaults

The following describes the metadata currently on disk, which may reflect command-line exclusions. It is a snapshot, not a guarantee about future runs.

| Saved output directory | Actual feature groups recorded |
| --- | --- |
| `nba_champion_predictions/` | Win percentage; three team ratings; six experience proxies; ten roster advanced statistics; both selected players' 17 statistics; five other team statistics |
| `nba_champion_predictions_xgboost/` | Win percentage; three team ratings; ten roster advanced statistics; only the first player's 17 statistics; other basic team statistics. Second-player and experience lists are empty. |

The logistic snapshot contains 59 features. Its five additional team columns are `FG`, `FGA`, `FG%`, `FT`, and `FTA`.

The XGBoost snapshot's additional team columns are `FG`, `FGA`, `FG%`, `FT`, `FTA`, `FT%`, `AST`, `PF`, `PTS`, `Trp-Dbl`, `TRB`, `MP`, `STL`, `BLK`, `TOV`, `ORB`, `DRB`, `3P`, `3PA`, `3P%`, `2P`, `2PA`, `2P%`, `eFG%`, and `Age`. Its saved feature list contains 56 columns. Empty feature groups identify what was excluded from that fitted run, without establishing which command-line arguments produced it.

To inspect a new run, read its `model_details.json`: `features`, `training_min_season`, `training_max_season`, `years_for_predictions`, and the per-group feature lists. Audit files include `advanced_stat_coverage.csv`, `team_rating_coverage.csv`, `top_two_player_features.csv`, `first_time_contender_features.csv`, and `season_data_coverage.csv` when relevant groups are enabled. Old audit files can remain from previous runs, so metadata for the fitted run is the better guide.

## 10. Exclusions and interpretation limits

The championship models do not currently include:

- Third-best-player statistics, roster concentration, top-player scoring shares, or team distribution measures.
- Same-season playoff outcomes, series results, championship labels, or MVP voting as predictive inputs. Championship labels are used for training targets and post-prediction evaluation only; prior playoff qualification is used for experience.
- A manually imposed voter-fatigue penalty, prior MVP-win history, or explicit player championship/deep-playoff experience. The separate MVP predictors have their own award-history logic.
- Explicit future injury availability, lineup fit, on/off or tracking data, trade forecasts, coaching adjustments, opponent-specific playoff matchups, bracket simulation, or betting odds.
- A single manually grouped “accuracy” score combining two-point and three-point shooting. Distinct percentages remain distinct eligible columns; TS% and eFG% are separate efficiency measures.

The dataset contains relatively few independent championship seasons, even though it contains many team-season rows. A model that wins a date-window search or improves a short backtest is not automatically more accurate on future seasons. Evaluate changes on comparable chronological periods, keep whole seasons together, and distinguish selecting a configuration from testing it on untouched later data.

## Suggested instruction to ChatGPT

“Use this document as the description of my existing NBA championship predictor. Distinguish supported/default features from the features actually recorded in each saved model. When suggesting improvements, account for the small number of seasons, correlated teams within each season, historical metric availability, chronological evaluation, and validation overfitting. Do not assume it is a preseason forecast, an annually retrained model, or a calibrated playoff simulator. Explain which proposed changes require additional data.”

## Update: optional roster depth experiment

The original snapshots and exclusions above describe depth-disabled runs, which
remain the default. Both championship scripts now support an optional depth
group with `--include-roster-depth`; `--no-roster-depth` explicitly disables it.
Old HHI/scoring distribution features and the full third-player statistics
group remain removed. The separate new depth group intentionally includes one
minutes-share distribution feature and two individual BPM inputs.

Its five columns are `Third_Player_BPM`, `Fourth_Player_BPM`,
`Supporting_Cast_BPM`, `Rotation_Depth`, and `Top_2_Minutes_Share`. Third/fourth
players follow the same ranking as the leaders. Supporting cast is the
team-stint-minute-weighted BPM of qualifying players at ranks 3–8; fewer than
six can contribute. Rotation depth counts players with at least 12 minutes per
team game. Top-two share uses the same selected leaders' minutes divided by all
team player minutes. BPM stays missing before 1974; the earlier PER ranking
fallback never becomes a substitute value under a BPM name.

Depth adds up to five eligible columns beyond the original base cap (normally
59 + 5 = 64), preserving base selection and training-only imputation. Runs
export depth calculations, names/rankings/minutes, missingness, and actual
selected flags. Model metadata now records configuration and command-line
arguments, so support and actual use can be distinguished.

`nba_compare_roster_depth.py` compares the frozen saved 56-feature baseline A,
A plus second-player/experience groups B (79 columns), and B plus depth C
(84 columns). The expanded caps retain every baseline column for a controlled
additive comparison. All three fit 1983–2002, validate 2003–2009, and use the
same fixed model for the previously explored 2010–2026 historical backtest.
The ordinary scripts' defaults and the date-window search are preserved.

Measured validation top-one results: A 4/7, B 4/7, C 3/7. Measured backtest
results: A 5/17, B 8/17, C 8/17. Individual depth ablations did not improve
validation performance. Depth stays disabled by default; additional inputs did
not establish an accuracy improvement. Detailed metrics, every season's picks,
and recommendations are in
`nba_roster_depth_experiments_1983_2002/REPORT.md` and its companion CSVs.

## Update: actual playoff experience and roster continuity

There is now an optional alternative to the old six experience proxies.
`--include-playoff-experience` replaces them with actual prior player playoff
participation and roster/coach continuity. `--no-playoff-continuity` retains the
17 participation inputs and excludes the five continuity inputs. New experience
remains off by default, preserving the earlier fitted behavior; both models
support these switches and record actual selected columns in metadata.

For each current leading player, five prior career values are tracked: NBA
playoff games, recorded minutes, series wins in which they participated,
Conference Finals appearances, and NBA Finals appearances. The same five
statistics are averaged across the current roster using current regular-season
stint minutes. The coach's prior playoff games and the inexperienced roster
minute share complete the 17 inputs. These roster weights include all regular-season
team stints, including departed players; they are not a verified playoff-opening
roster. Coach games cover head-coaching roles, excluding assistant assignments.
Stable NBA player identities follow
experience across trades. DNP players receive no credit for a series they sat
out. Conference Finals begin in 1971 rather than relabeling old division finals.

Historical roster similarity is the overlap of current regular-season and
previous playoff minute distributions: sum of each shared player's smaller
minute share. A new roster has zero overlap even if its franchise once won a
title. The five model continuity inputs are last-season overlap, unchanged
coach, and last-five-season series wins/Conference Finals/Finals weighted by
overlap, coach continuity, and a three-year half-life. The coach factor is 1
when unchanged and 0.5 after a change. Those definitions were fixed before the
comparison, rather than tuned on backtest outcomes. The audit also compares
distant playoff rosters but marks them outside the model window. Thus a 2005
Spurs roster cannot provide team-success credit to the 2026 Spurs; individual
player/coach experience is evaluated separately.

Actual historical records have been downloaded and checked, not merely
implemented as empty inputs. Compact player, coach, series, team, identity and
coverage files are in `nba_updated/playoff_history/`, covering 1954–2026. Earlier
game-archive years are incomplete, so older player careers are marked censored
instead of treated as zero experience. At least 95% usable current roster
minutes and resolved leaders are required. Series tables audit the few
published season/game-count conflicts; minute totals use recorded season totals
where verified and retain disclosed game-log precision when no cross-check is
available. Experience is NBA-only and uses seasons strictly before prediction.

`nba_compare_playoff_experience.py` holds all 73 statistical inputs, dates, and
per-algorithm settings fixed while comparing old proxies (79 total inputs),
actual participation (90), and participation plus continuity (95). Both
algorithms train 1983–2002, validate 2003–2009, and backtest 2010–2026 with a
fixed fit. No 2010+ championship labels enter fitting or feature selection.
These later years have already been explored, so they are not untouched.

XGBoost validation winner accuracy was 4/7 with proxies, 5/7 with actual
experience, and 4/7 with continuity added. Corresponding backtest counts were
8/17, 8/17, and 7/17. Logistic validation counts were 1/7, 2/7, and 0/7;
backtest counts were 3/17, 5/17, and 4/17. Actual participation alone is the
more promising candidate for winner accuracy on this small validation sample.
Full metrics, every season's predictions, roster comparisons, exact selected
columns, and source hashes are in
`nba_playoff_experience_comparison_verified_1983_2002/REPORT.md` and its CSVs.
