#!/usr/bin/env python3
"""Compare old playoff-roster proxies, actual participation, and continuity."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from nba_predictor.nba_champion_best_player import SECOND_PLAYER_FEATURES
from nba_predictor.nba_champion_first_time import FIRST_TIME_FEATURES
from nba_predictor.nba_champion_playoff_experience import (ACTUAL_EXPERIENCE_FEATURES, CONTINUITY_FEATURES,
                                             PLAYOFF_EXPERIENCE_FEATURES, EXPERIENCE_RULE, add_playoff_experience)
from nba_predictor.nba_champion_window_search import prepare_data
from nba_predictor.nba_compare_roster_depth import fit_configuration, summarize, validate_dates
from nba_predictor.nba_predict_champions_xgboost_2010_2026 import locate


def feature_configurations(baseline):
    base = list(dict.fromkeys([*baseline, *SECOND_PLAYER_FEATURES]))
    if any(c in FIRST_TIME_FEATURES or c in PLAYOFF_EXPERIENCE_FEATURES for c in base):
        raise ValueError('Baseline metadata must have no experience inputs')
    return {'Proxy': [*base, *FIRST_TIME_FEATURES],
            'Actual': [*base, *ACTUAL_EXPERIENCE_FEATURES],
            'Actual_Continuity': [*base, *PLAYOFF_EXPERIENCE_FEATURES]}


def write_report(output, comparison, differences, args):
    lines = ['# Actual playoff experience and roster continuity', '',
             f'Training: **{args.train_start}–{args.train_end}**. Validation: **{args.validation_start}–{args.validation_end}**. '
             f'Historical backtest: **{args.backtest_start}–{args.backtest_end}**. Models remain fixed after training.', '',
             'The later seasons have already been examined in earlier project experiments. They are not an untouched holdout. '
             'No championship labels from 2010 onward enter fitting or feature selection. Predictions use that season’s '
             'regular-season stats, plus history strictly before that season.', '',
             '## Controlled configurations', '',
             '- **Proxy:** the saved 56-column baseline plus the second player (17 columns) and old six playoff-roster proxies: 79 inputs.',
             '- **Actual:** retain the same 73 statistical inputs, replace the old proxies with 17 actual-participation inputs: 90 inputs.',
             '- **Actual_Continuity:** Actual plus five roster/coach continuity inputs: 95 inputs.', '',
             'Roster depth remains disabled. Expanded explicit lists prevent the feature cap from silently dropping statistics. '
             'Logistic regression and XGBoost each use their existing hyperparameters unchanged across the three configurations.', '',
             '| Algorithm | Configuration | Period | Correct / seasons | Top 1 | Top 3 | MRR |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for row in comparison.itertuples(index=False):
        lines.append(f'| {row.Algorithm} | {row.Configuration} | {row.Period} | {row.Correct}/{row.Seasons} | '
                     f'{row.Top1:.1%} | {row.Top3:.1%} | {row.Mean_Reciprocal_Rank:.4f} |')
    lines += ['', '## Differences', '',
              '| Algorithm | Change | Period | Correct delta | Top 1 delta | Top 3 delta | MRR delta |',
              '| --- | --- | --- | --- | --- | --- | --- |']
    for row in differences.itertuples(index=False):
        lines.append(f'| {row.Algorithm} | {row.Comparison} | {row.Period} | {row.Delta_Correct:+.0f} | '
                     f'{row.Delta_Top1*100:+.1f} pp | {row.Delta_Top3*100:+.1f} pp | {row.Delta_Mean_Reciprocal_Rank:+.4f} |')
    lines += ['', '## What the new inputs mean', '',
              'Five career statistics are accumulated for each actual current player: playoff games, recorded playoff minutes, '
              'series wins in which the player appeared, seasons appearing in a Conference Final, and seasons appearing in an NBA Final. '
              'A bench player who sat out the relevant series gets no series/round credit. Conference Finals start in 1971; '
              'earlier division finals are not relabeled.', '',
              'The 17 participation inputs are these five statistics for each of the two selected leaders (ten), '
              'current-minute-weighted averages across the current roster (five), the current coach’s prior playoff games, '
              'and the share of identified current roster minutes belonging to players with no prior playoff games. '
              'Players keep experience after a trade. NBA identity matching and coverage diagnostics distinguish unresolved '
              'history from true inexperience.', '',
              'Here “current roster” means all regular-season team stints in that season, preserving the existing inputs. '
              'It is not a transaction-verified playoff-opening roster: departed in-season players can still contribute '
              'their stint minutes. Injury availability and assistant-coaching experience are not modeled. '
              'Coach experience is head-coach playoff games only.', '',
              'Continuity adds last-season playoff-roster overlap, whether the coach is unchanged from last season, '
              'and three weighted last-five-season results: series wins, Conference Finals, and Finals. '
              'Roster overlap is the sum of the smaller minute share for every player shared by the current regular-season '
              'roster and a prior playoff roster. It is 0 for disjoint rosters and at most 1.', '',
              '**Transfer weight = overlap × coach factor × 0.5^((years ago − 1)/3)**. The coach factor is 1 for the same coach '
              'and 0.5 after a change. These are fixed experimental feature definitions, chosen before evaluation; they are '
              'not a forced penalty on the model’s predictions. The model can learn to use or ignore the inputs.', '',
              'A completely replaced roster receives zero transferred team success even if the franchise won before. '
              'A 2005 title is outside the five-year team-history window for 2026. Career experience still follows any '
              'actual players or coaches who remain. Team codes and historical team names are never model inputs.', '',
              '## Data quality and practical limits', '',
              'Historical game and player records were downloaded from the public Eoin Moore NBA archive. '
              'Coaches come from Basketball Reference and a historical export. The game archive is incomplete before 1954, '
              'so earlier player careers are marked left-censored and excluded from known-history averages. '
              'The feature builder requires at least 95% usable current-roster minutes and identified leaders; '
              'unknown players are not silently assigned rookie histories.', '',
              'The archive’s missing 2022 player team ids are recovered using game ids and explicit home/away flags. '
              'Historical series are grouped by opponents/stage rather than inconsistent old game ids. '
              'The 1954 round robin contributes games/minutes but no fictitious series victories. '
              'Every historical team’s playoff game count is checked against independent coaching records.', '',
              'Recorded season totals correct minute precision and cross-check player games. Exact duplicate API rows are removed. '
              'The provider lacks pre-2001 totals, so older seasons use Basketball Reference totals. '
              'A one-game discrepancy may be reconciled to the season total when every credited series has independently '
              'recorded positive player minutes. Other disputed records are checked against the player’s actual series table. '
              'Some published game/series counts still disagree with season totals; both remain visible in the cross-check CSV. '
              'Unmatched independent totals retain their reported game-log values and are disclosed in the source audit.', '',
              'Seven validation seasons provide limited evidence. The roster-overlap window, decay, and coach factor are hypotheses '
              'rather than proven optimal settings. The backtest is descriptive and does not select a configuration.', '',
              '## Files', '',
              '- `model_comparison.csv`, `performance_differences.csv`: all measured metrics and changes.',
              '- `season_predictions.csv`: each season’s top three and actual champion rank for every model.',
              '- `actual_playoff_experience.csv`, `player_playoff_experience.csv`: team/leader inputs and individual histories.',
              '- `playoff_roster_similarity.csv`: each current team versus all its earlier franchise playoff rosters, coaches, shared players, overlap, and transfer weights. Only rows flagged inside five years enter the model.',
              '- `playoff_identity_coverage.csv`: identity matching and left-censoring diagnostics.',
              '- Model subdirectories: all team scores, feature selection audits, coefficients/importances, and exact selected-feature metadata.',
              '- `data/nba_updated/playoff_history/`: downloaded/derived player, coach, series and team histories, source coverage and totals cross-checks.', '',
              'All original prediction outputs and earlier experiments are preserved. See `experiment_manifest.json` for '
              'source hashes, arguments, and selected configurations.', '']
    (output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('data/nba_updated'))
    parser.add_argument('--analysis-dir', type=Path, default=Path('outputs/nba_champions_analysis'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/nba_playoff_experience_comparison_1983_2002'))
    parser.add_argument('--playoff-history-dir', type=Path, default=Path('data/nba_updated/playoff_history'))
    parser.add_argument('--baseline-details', type=Path, default=Path('outputs/nba_champion_predictions_xgboost/model_details.json'))
    parser.add_argument('--train-start', type=int, default=1983)
    parser.add_argument('--train-end', type=int, default=2002)
    parser.add_argument('--validation-start', type=int, default=2003)
    parser.add_argument('--validation-end', type=int, default=2009)
    parser.add_argument('--backtest-start', type=int, default=2010)
    parser.add_argument('--backtest-end', type=int, default=2026)
    parser.add_argument('--n-estimators', type=int, default=240)
    parser.add_argument('--max-depth', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=.04)
    args = parser.parse_args()
    validate_dates(args)
    if args.validation_start != args.train_end + 1 or args.backtest_start != args.validation_end + 1:
        raise ValueError('Use consecutive training, validation, and backtest periods')
    if args.output_dir.exists():
        raise ValueError('Choose a new output directory; existing results are preserved')
    baseline = json.loads(args.baseline_details.read_text())
    configs = feature_configurations(baseline['features'])
    args.output_dir.mkdir(parents=True)
    args.first_year, args.test_end, args.last_validation_year = 1970, args.backtest_end, args.validation_end
    args.advanced_file = args.stats_file = args.team_ratings_file = None
    args.min_team_coverage, args.min_feature_coverage = .75, .35
    frame, _, labels, sources = prepare_data(args)
    frame = frame.drop(columns=['_label'], errors='ignore')
    stats = locate(args.input_dir, 'NBA_stats_[0-9]*_[0-9]*.csv')
    frame, audit, player_audit, similarities, identities = add_playoff_experience(frame, stats, args.playoff_history_dir)
    audit.to_csv(args.output_dir/'actual_playoff_experience.csv', index=False)
    player_audit.to_csv(args.output_dir/'player_playoff_experience.csv', index=False)
    similarities.to_csv(args.output_dir/'playoff_roster_similarity.csv', index=False)
    identities.to_csv(args.output_dir/'playoff_identity_coverage.csv', index=False)
    manifest = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    configurations=configs, definition=EXPERIENCE_RULE,
                    sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [*sources, *sorted(args.playoff_history_dir.glob('*.csv')), args.playoff_history_dir/'metadata.json']})
    (args.output_dir/'experiment_manifest.json').write_text(json.dumps(manifest, indent=2))
    summaries, season_results = [], []
    for algorithm in ('logistic', 'xgboost'):
        for config, features in configs.items():
            model, predictions, results = fit_configuration(frame, labels, features, args, args.backtest_end,
                                                             model_name=algorithm, include_playoff_experience=True)
            name = f'{algorithm}_{config}'
            directory = args.output_dir/name
            directory.mkdir()
            predictions.to_csv(directory/'team_predictions.csv', index=False)
            results.to_csv(directory/'season_predictions.csv', index=False)
            classifier = model.named_steps['classifier']
            importance = classifier.feature_importances_ if algorithm == 'xgboost' else classifier.coef_.ravel()
            pd.DataFrame({'Feature': features, 'Importance_Or_Coefficient': importance}).to_csv(directory/'feature_importance.csv', index=False)
            training = frame.loc[frame.Season.between(args.train_start, args.train_end)]
            pd.DataFrame([dict(Feature=c, Selected=c in features,
                               Training_Coverage=training[c].notna().mean())
                          for c in dict.fromkeys([*configs['Proxy'], *configs['Actual_Continuity']])]).to_csv(
                              directory/'feature_selection_audit.csv', index=False)
            details = dict(algorithm=algorithm, configuration=config, features=features,
                           training_min_season=args.train_start, training_max_season=args.train_end,
                           validation_seasons=[args.validation_start, args.validation_end],
                           years_for_predictions=[args.validation_start, args.backtest_end],
                           hyperparameters=classifier.get_params(),
                           first_time_features_used=[c for c in features if c in FIRST_TIME_FEATURES],
                           actual_experience_features_used=[c for c in features if c in ACTUAL_EXPERIENCE_FEATURES],
                           continuity_features_used=[c for c in features if c in CONTINUITY_FEATURES],
                           playoff_experience_configuration=dict(
                               enabled=config != 'Proxy', definition=EXPERIENCE_RULE,
                               continuity_enabled=config == 'Actual_Continuity',
                               continuity_parameters=dict(lookback_seasons=5, half_life_years=3, same_coach_factor=1, changed_coach_factor=.5)),
                           roster_depth_features_used=[], same_season_postseason_features_used=False,
                           backtest_is_untouched_holdout=False)
            (directory/'model_details.json').write_text(json.dumps(details, indent=2))
            rows = summarize(results, name)
            for row in rows:
                row.update(Algorithm=algorithm, Configuration=config)
            summaries.extend(rows)
            season_results.append(results.assign(Algorithm=algorithm, Configuration=config))
            print(name, rows, flush=True)
    comparison = pd.DataFrame(summaries)
    comparison.to_csv(args.output_dir/'model_comparison.csv', index=False)
    pd.concat(season_results, ignore_index=True).to_csv(args.output_dir/'season_predictions.csv', index=False)
    deltas = []
    for algorithm in ('logistic', 'xgboost'):
        for period in ('Validation', 'Backtest'):
            group = comparison.loc[comparison.Algorithm.eq(algorithm) & comparison.Period.eq(period)].set_index('Configuration')
            for newer, older in [('Actual', 'Proxy'), ('Actual_Continuity', 'Actual'), ('Actual_Continuity', 'Proxy')]:
                deltas.append(dict(Algorithm=algorithm, Period=period, Comparison=f'{newer} - {older}',
                                   **{f'Delta_{c}': group.loc[newer, c] - group.loc[older, c]
                                      for c in ('Correct', 'Top1', 'Top3', 'Mean_Reciprocal_Rank')}))
    differences = pd.DataFrame(deltas)
    differences.to_csv(args.output_dir/'performance_differences.csv', index=False)
    write_report(args.output_dir, comparison, differences, args)
    print('Saved', args.output_dir.resolve())
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f'ERROR: {exc}')
