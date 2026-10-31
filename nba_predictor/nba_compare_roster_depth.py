#!/usr/bin/env python3
"""Controlled A/B/C depth comparison; all fitted labels precede 2010.

The baseline column list is frozen from saved metadata. B appends second-player
and experience inputs; C appends depth. Nothing displaces a baseline column.
The fixed fit is validated on 2003–2009 and backtested on 2010–2026 by default.
Individual/leave-one-out depth ablations use validation only, never backtest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_predictor.nba_champion_advanced import TEAM_ADVANCED_FEATURES
from nba_predictor.nba_champion_best_player import BEST_PLAYER_FEATURES, SECOND_PLAYER_FEATURES
from nba_predictor.nba_champion_first_time import FIRST_TIME_FEATURES
from nba_predictor.nba_champion_roster_depth import (ROSTER_DEPTH_FEATURES, add_roster_depth,
                                       depth_configuration, feature_selection_audit)
from nba_predictor.nba_champion_window_search import prepare_data
from nba_predictor.nba_predict_champions_xgboost_2010_2026 import (available_features, make_xgb_pipeline,
                                                   to_season_zscores)
from nba_predictor.nba_team_ratings import TEAM_RATING_FEATURES


def validate_dates(args):
    if not (args.train_start <= args.train_end < args.validation_start <= args.validation_end < 2010
            <= args.backtest_start <= args.backtest_end):
        raise ValueError('Require training < validation < 2010 <= backtest, with inclusive valid ranges')
    if args.train_end - args.train_start + 1 < 10:
        raise ValueError('At least ten training seasons required')


def configurations(baseline):
    if any(c in (*SECOND_PLAYER_FEATURES, *FIRST_TIME_FEATURES, *ROSTER_DEPTH_FEATURES) for c in baseline):
        raise ValueError('Model A metadata must exclude second-player, experience, and depth inputs')
    expanded = list(dict.fromkeys([*baseline, *SECOND_PLAYER_FEATURES, *FIRST_TIME_FEATURES]))
    return {'A': list(baseline), 'B': expanded, 'C': [*expanded, *ROSTER_DEPTH_FEATURES]}


def fit_configuration(frame, labels, features, args, end, model_name='xgboost', include_playoff_experience=False):
    """Fit ONLY on explicit train rows; features have no outcome-based selection."""
    validate_dates(args)
    train = frame.Season.between(args.train_start, args.train_end)
    target = frame.Season.between(args.validation_start, end)
    allowed = available_features(frame, train, args.min_feature_coverage, 10000, True, include_playoff_experience)
    invalid = [column for column in features if column not in allowed]
    if invalid:
        raise ValueError(f'Explicit features unavailable in training: {invalid}')
    zscores = to_season_zscores(frame, features)
    empty = [c for c in features if not zscores.loc[train, c].notna().any()]
    if empty:
        raise ValueError(f'Explicit features have no within-season training variation: {empty}')
    # Deliberately construct targets only for training rows. Later labels are
    # consulted below AFTER predict_proba, solely to score predictions.
    y = pd.Series([int(labels[int(row.Season)] == row.Team)
                   for row in frame.loc[train, ['Season', 'Team']].itertuples()],
                  index=frame.index[train], dtype=int)
    if model_name == 'xgboost':
        pipeline = make_xgb_pipeline(y, args)
        pipeline.set_params(classifier__n_jobs=1)
    elif model_name == 'logistic':
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(C=.20, class_weight='balanced', max_iter=3000, random_state=42))])
    else:
        raise ValueError('Unknown model')
    with threadpool_limits(limits=1):
        pipeline.fit(zscores.loc[train, features], y)
        predictions = frame.loc[target, ['Season', 'Team']].copy()
        predictions['Model_Score'] = pipeline.predict_proba(zscores.loc[target, features])[:, 1]
    predictions['Rank'] = predictions.groupby('Season').Model_Score.rank(
        ascending=False, method='first').astype(int)
    predictions['Predicted_Winner'] = predictions.Rank.eq(1)
    rows = []
    for year, sample in predictions.groupby('Season', sort=True):
        ordered = sample.sort_values('Rank')
        actual = labels.get(int(year))
        match = ordered.loc[ordered.Team.eq(actual)]
        actual_rank = int(match.Rank.iloc[0]) if len(match) else None
        rows.append(dict(Season=int(year), Period='Validation' if year <= args.validation_end else 'Backtest',
                         Predicted_Champion=ordered.Team.iloc[0],
                         Predicted_Second=ordered.Team.iloc[1], Predicted_Third=ordered.Team.iloc[2],
                         Actual_Champion=actual, Actual_Rank=actual_rank,
                         Correct=actual_rank == 1 if actual_rank else None,
                         Actual_In_Top3=actual_rank <= 3 if actual_rank else None,
                         Reciprocal_Rank=1 / actual_rank if actual_rank else None,
                         Teams_Scored=len(ordered)))
    results = pd.DataFrame(rows)
    predictions['Actual_Champion_EVALUATION_ONLY'] = predictions.Season.map(labels)
    predictions['Is_Actual_Champion_EVALUATION_ONLY'] = predictions.Team.eq(
        predictions.Actual_Champion_EVALUATION_ONLY)
    return pipeline, predictions.sort_values(['Season', 'Rank']), results


def summarize(results, model):
    rows = []
    for period, group in results.groupby('Period', sort=False):
        scored = group.loc[group.Actual_Rank.notna()]
        rows.append(dict(Model=model, Period=period, Seasons=len(scored),
                         Correct=int(scored.Correct.sum()),
                         Top1=float(scored.Correct.mean()) if len(scored) else None,
                         Top3=float(scored.Actual_In_Top3.mean()) if len(scored) else None,
                         Mean_Reciprocal_Rank=float(scored.Reciprocal_Rank.mean()) if len(scored) else None))
    return rows


def save_model(name, features, pipeline, predictions, results, frame, args, depth_audit):
    directory = args.output_dir / f'model_{name}'
    directory.mkdir()
    predictions.to_csv(directory/'team_predictions_2010_2026.csv', index=False)
    results.to_csv(directory/'predicted_champions_2010_2026.csv', index=False)
    depth_enabled = any(c in ROSTER_DEPTH_FEATURES for c in features)
    train = frame.Season.between(args.train_start, args.train_end)
    eligible = available_features(frame, train, args.min_feature_coverage, 10000, True)
    selection = feature_selection_audit(frame, train, features, eligible, depth_enabled)
    selection.loc[~selection.Selected & ~selection.Roster_Depth, 'Reason'] = 'excluded by explicit experiment configuration'
    selection.to_csv(directory/'feature_selection_audit.csv', index=False)
    audit = depth_audit.copy()
    for column in ROSTER_DEPTH_FEATURES:
        audit[f'{column}_Selected'] = column in features
    audit.to_csv(directory/'roster_depth_features.csv', index=False)
    importance = pipeline.named_steps['classifier'].feature_importances_
    pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values(
        'Importance', ascending=False).to_csv(directory/'feature_importances.csv', index=False)
    depth_config = depth_configuration(depth_enabled, features, len(features) - sum(c in ROSTER_DEPTH_FEATURES for c in features))
    depth_config['cap_policy'] = 'Explicit additive feature lists; cap expanded so no existing inputs are displaced.'
    metadata = dict(
        algorithm='XGBClassifier', experiment=name,
        training_min_season=args.train_start, training_max_season=args.train_end,
        validation_seasons=[args.validation_start, args.validation_end],
        years_for_predictions=[args.validation_start, args.backtest_end],
        historical_backtest_seasons=[args.backtest_start, args.backtest_end],
        number_of_training_seasons=int(frame.loc[train, 'Season'].nunique()),
        features=features, feature_selection='Explicit frozen baseline plus named groups; no label-based selection',
        hyperparameters=pipeline.named_steps['classifier'].get_params(),
        first_time_features_used=[c for c in features if c in FIRST_TIME_FEATURES],
        advanced_features_used=[c for c in features if c in TEAM_ADVANCED_FEATURES],
        best_player_features_used=[c for c in features if c in BEST_PLAYER_FEATURES],
        second_player_features_used=[c for c in features if c in SECOND_PLAYER_FEATURES],
        third_player_features_used=[c for c in features if c == 'Third_Player_BPM'],
        team_rating_features_used=[c for c in features if c in TEAM_RATING_FEATURES],
        roster_depth_features_used=[c for c in features if c in ROSTER_DEPTH_FEATURES],
        roster_depth_configuration=depth_config,
        same_season_postseason_features_used=False,
        uses_same_season_regular_season_stats_for_predictions=True,
        prediction_season_championship_labels_used_for_training=False,
        team_distribution_features_used='Top_2_Minutes_Share' in features,
        note='2010–2026 is a previously explored historical backtest, not an untouched holdout. Scores are not calibrated probabilities.')
    (directory/'model_details.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


def write_report(directory, comparison, ablations, args):
    """Keep the human-readable result next to its machine-readable evidence."""
    validation = comparison.loc[comparison.Period.eq('Validation')].set_index('Model')
    counts = {name: len(json.loads((directory/f'model_{name}'/'model_details.json').read_text())['features'])
              for name in ('A', 'B', 'C')}
    report = [
        '# Roster depth experiment results', '',
        f'All models train on **{args.train_start}–{args.train_end}**, validate on '
        f'**{args.validation_start}–{args.validation_end}**, and use the same fixed fit for '
        f'the **{args.backtest_start}–{args.backtest_end}** historical backtest.', '',
        '**The backtest has been examined in earlier project experiments. It is not an untouched holdout.** '
        'No championship labels from 2010 onward are used for fitting, feature selection, or the depth ablations. '
        'Prior-season playoff qualification remains available for the experience proxies.', '',
        '## Configurations', '',
        f'- **A:** exact {counts["A"]}-column saved XGBoost baseline, without second-player statistics, experience proxies, or depth.',
        f'- **B:** A plus 17 second-player statistics and six existing experience proxies ({counts["B"]} columns).',
        f'- **C:** B plus all five depth features ({counts["C"]} columns).', '',
        'The baseline list is frozen from `baseline_details_snapshot.json`. The saved baseline lacked second-player '
        'and experience columns even though current script defaults support them. These priority groups fit under '
        'the old cap; the cap does not explain their absence. `--no-second-player` reproduces that group exclusion '
        'and automatically disables experience. The original invocation was not saved, so its exact flags are unknown.', '',
        'All baseline columns remain in B and C. The cap is expanded explicitly for this controlled comparison; '
        'the ordinary scripts retain their 59-column base cap and append up to five depth inputs when enabled. '
        'Therefore ordinary default-cap runs are not identical to this frozen-column experiment.', '',
        f'XGBoost settings are identical: {args.n_estimators} trees, depth {args.max_depth}, '
        f'learning rate {args.learning_rate}, minimum child weight 3, '
        'row/column sampling 0.85, L2 7, L1 0.1, seed 42, and training-only class weighting. '
        'Exact settings and data hashes are in `experiment_manifest.json` and each model directory. '
        'Within-season normalization is unchanged; medians are fitted only on training rows.', '',
        '## Measured results', '',
        '| Model | Period | Correct / seasons | Top 1 | Top 3 | Mean reciprocal rank |',
        '| --- | --- | --- | --- | --- | --- |',
    ]
    for row in comparison.itertuples(index=False):
        report.append(f'| {row.Model} | {row.Period} | {row.Correct}/{row.Seasons} | '
                      f'{row.Top1:.1%} | {row.Top3:.1%} | {row.Mean_Reciprocal_Rank:.4f} |')
    report += ['', '## Differences', '',
               '| Comparison | Period | Correct delta | Top 1 delta | Top 3 delta | MRR delta |',
               '| --- | --- | --- | --- | --- | --- |']
    differences = pd.read_csv(directory/'performance_differences.csv')
    for row in differences.itertuples(index=False):
        report.append(f'| {row.Comparison} | {row.Period} | {row.Delta_Correct:+.0f} | '
                      f'{100*row.Delta_Top1:+.1f} pp | {100*row.Delta_Top3:+.1f} pp | '
                      f'{row.Delta_Mean_Reciprocal_Rank:+.4f} |')
    report += ['', '## Which depth inputs helped validation?', '',
               'These experiments use only the pre-2010 validation period. “B + feature” tests one depth input; '
               '“C minus feature” removes it from the full depth model. All fits retain the same training years '
               'and hyperparameters. Trying several subsets itself increases validation-selection risk.', '',
               '| Feature | B + feature: correct | B + feature: MRR | C minus feature: correct | C minus feature: MRR |',
               '| --- | --- | --- | --- | --- |']
    for feature in ROSTER_DEPTH_FEATURES:
        rows = ablations.loc[ablations.Feature.eq(feature)].set_index('Experiment')
        add, remove = rows.loc['Add_One'], rows.loc['Leave_One_Out']
        report.append(f'| {feature} | {int(add.Correct)} | {add.Mean_Reciprocal_Rank:.4f} | '
                      f'{int(remove.Correct)} | {remove.Mean_Reciprocal_Rank:.4f} |')
    singles = ablations.loc[ablations.Experiment.eq('Add_One')]
    improvements = singles.loc[(singles.Top1 > validation.loc['B', 'Top1']) |
                               (singles.Top3 > validation.loc['B', 'Top3']) |
                               (singles.Mean_Reciprocal_Rank > validation.loc['B', 'Mean_Reciprocal_Rank']), 'Feature'].tolist()
    report += ['', 'Individual additions improving at least one validation metric over B: '
               + (', '.join(improvements) if improvements else '**none**') + '.']
    if validation.loc['C', 'Top1'] <= validation.loc['B', 'Top1']:
        recommendation = ('Keep roster depth **disabled by default**. Adding all five inputs did not improve '
                          'validation winner accuracy. No subset should be promoted solely because of the backtest.')
    else:
        recommendation = ('Depth improved winner accuracy on this validation period, but should remain experimental '
                          'until checked on additional chronological folds. This is not evidence of guaranteed future gains.')
    report += ['', recommendation, '',
               'The single-feature results distinguish measured usefulness from mere inclusion. Feature importance '
               'does not establish an accuracy gain: extra columns also change XGBoost column sampling. '
               'The validation period is short, and historical training windows were previously explored. '
               'These results support cautious model comparison, not a claim of a reliable accuracy improvement.', '',
               '## Five new calculations', '',
               '1. **Third_Player_BPM:** BPM of the player ranked third by the existing leader-selection order.',
               '2. **Fourth_Player_BPM:** BPM of the player ranked fourth by the same order.',
               '3. **Supporting_Cast_BPM:** sum(BPM × team-stint minutes) / sum(team-stint minutes) for qualifying '
               'players at ranks 3–8; use the available subset when fewer qualify. No support players means missing.',
               '4. **Rotation_Depth:** number of players with team-stint minutes / team games ≥ 12.',
               '5. **Top_2_Minutes_Share:** the existing selected leaders’ combined team-stint minutes divided by '
               'all positive team-stint player minutes.', '',
               'Ranking uses the existing threshold of 10 minutes per team game, BPM descending, minutes, then '
               'normalized name. Remaining players with finite ratings can fill third/fourth positions by the '
               'existing minutes fallback; fallback players are excluded from the supporting-cast average. '
               'Before 1974, PER determines ranks but all three BPM features remain missing. '
               'Missing BPM is never treated as zero. Combined-team records are excluded, exact duplicates '
               'are removed, conflicting stint duplicates are rejected, and traded players contribute only '
               'their minutes with the relevant team.', '',
               '## Outputs and audit', '',
               '- `model_comparison.csv`: top-one/top-three/MRR and correct counts for every period.',
               '- `performance_differences.csv`: B−A, C−B, and C−A changes for each period.',
               '- `season_predictions.csv`: first/second/third picks and actual champion rank for every evaluated season.',
               '- `validation_depth_ablations.csv` and `validation_ablation_predictions.csv`: subset tests and every validation prediction.',
               '- `roster_depth_features.csv`: all five inputs and third/fourth names, plus coverage and selection diagnostics.',
               '- `roster_depth_player_rankings.csv`: player ranks, stint minutes, qualification, BPM availability, and support inclusion.',
               '- `model_A/`, `model_B/`, `model_C/`: team scores, selected feature audits, importances, depth selection flags, and model metadata.',
               '- `baseline_details_snapshot.json` and `experiment_manifest.json`: baseline snapshot, exact configurations, sources, and arguments.', '',
               'Original predictor output directories are preserved. No network downloads were needed.', '',
               '## Source files', '',
               '- Modified both championship scripts to add optional depth inputs, additive cap handling, audits, and configuration metadata.',
               '- Extracted shared ranking into `nba_champion_best_player.py` without changing the existing top-two order.',
               '- Added `nba_champion_roster_depth.py` for calculations and diagnostics.',
               '- Added `nba_compare_roster_depth.py` for reproducible fixed-window comparison and validation-only ablations.',
               '- Added `test_nba_champion_roster_depth.py` for formulas, chronology, missingness, stints, and compatibility.',
               '- Updated `README.md` and `CHAMPION_PREDICTOR_CONTEXT_FOR_CHATGPT.md` for the optional feature group.', '']
    (directory/'REPORT.md').write_text('\n'.join(report), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('data/nba_updated'))
    parser.add_argument('--analysis-dir', type=Path, default=Path('outputs/nba_champions_analysis'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/nba_roster_depth_experiments'))
    parser.add_argument('--baseline-details', type=Path, default=Path('outputs/nba_champion_predictions_xgboost/model_details.json'))
    parser.add_argument('--advanced-file', type=Path)
    parser.add_argument('--stats-file', type=Path)
    parser.add_argument('--team-ratings-file', type=Path)
    parser.add_argument('--train-start', type=int, default=1983)
    parser.add_argument('--train-end', type=int, default=2002)
    parser.add_argument('--validation-start', type=int, default=2003)
    parser.add_argument('--validation-end', type=int, default=2009)
    parser.add_argument('--backtest-start', type=int, default=2010)
    parser.add_argument('--backtest-end', type=int, default=2026)
    parser.add_argument('--n-estimators', type=int, default=240)
    parser.add_argument('--max-depth', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=.04)
    parser.add_argument('--min-feature-coverage', type=float, default=.35)
    parser.add_argument('--min-team-coverage', type=float, default=.75)
    args = parser.parse_args()
    validate_dates(args)
    if args.n_estimators < 1 or not 1 <= args.max_depth <= 10 or not 0 < args.learning_rate <= 1:
        raise ValueError('Invalid XGBoost hyperparameters')
    if args.backtest_start != args.validation_end + 1 or args.validation_start != args.train_end + 1:
        raise ValueError('Use consecutive training, validation, and backtest periods')
    baseline = json.loads(args.baseline_details.read_text())
    configs = configurations(baseline['features'])
    # Never overwrite existing models or a previous experiment.
    if args.output_dir.exists():
        raise ValueError('Experiment directory exists; choose a new --output-dir to preserve outputs')
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'baseline_details_snapshot.json').write_text(json.dumps(baseline, indent=2), encoding='utf-8')
    args.first_year = min(1970, args.train_start)
    args.test_end = args.backtest_end
    args.last_validation_year = args.validation_end
    frame, _, labels, sources = prepare_data(args)
    advanced_path = next(p for p in sources if p.name.startswith('NBA_advanced_')) if args.advanced_file is None else args.advanced_file
    frame, depth_audit, rankings = add_roster_depth(frame, advanced_path)
    # prepare_data also serves window search; remove its outcome column here.
    frame = frame.drop(columns=['_label'], errors='ignore')
    depth_audit.to_csv(args.output_dir/'roster_depth_features.csv', index=False)
    rankings.to_csv(args.output_dir/'roster_depth_player_rankings.csv', index=False)
    manifest = dict(settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    configurations=configs,
                    sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    baseline_exclusion_finding=(
                        'Saved metadata records no second-player or experience inputs, although current defaults include them. '
                        'These inputs are prioritized and fit under the old cap, so the cap does not explain the omission. '
                        'The old metadata lacks command-line arguments; exact invocation cannot be recovered. '
                        '--no-second-player reproduces the feature groups and disables experience automatically. '
                        'This experiment uses the exact saved baseline column list instead of assuming defaults.'),
                    evaluation='Fixed fit; validation before 2010; previously explored historical backtest 2010–2026; no annual refitting.',
                    ablation_selection='Validation only; backtest outcomes never select depth features.')
    (args.output_dir/'experiment_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    summary, predictions = [], []
    for name, features in configs.items():
        model, teams, results = fit_configuration(frame, labels, features, args, args.backtest_end)
        save_model(name, features, model, teams, results, frame, args, depth_audit)
        summary.extend(summarize(results, name))
        predictions.append(results.assign(Model=name))
        print(f'Model {name}: {len(features)} explicit features', flush=True)
    comparison = pd.DataFrame(summary)
    comparison.to_csv(args.output_dir/'model_comparison.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(args.output_dir/'season_predictions.csv', index=False)
    deltas = []
    for period in ('Validation', 'Backtest'):
        rows = comparison.loc[comparison.Period.eq(period)].set_index('Model')
        for newer, older in (('B', 'A'), ('C', 'B'), ('C', 'A')):
            deltas.append(dict(Comparison=f'{newer} - {older}', Period=period,
                               **{f'Delta_{metric}': rows.loc[newer, metric] - rows.loc[older, metric]
                                  for metric in ('Correct', 'Top1', 'Top3', 'Mean_Reciprocal_Rank')}))
    pd.DataFrame(deltas).to_csv(args.output_dir/'performance_differences.csv', index=False)
    ablations, ablation_predictions = [], []
    for column in ROSTER_DEPTH_FEATURES:
        for kind, columns in (('Add_One', [*configs['B'], column]),
                              ('Leave_One_Out', [c for c in configs['C'] if c != column])):
            name = f'{kind}:{column}'
            _, _, results = fit_configuration(frame, labels, columns, args, args.validation_end)
            row = summarize(results, name)[0]
            row.update(Feature=column, Experiment=kind, Features_Used=len(columns))
            ablations.append(row)
            ablation_predictions.append(results.assign(Model=name))
    ablations = pd.DataFrame(ablations)
    ablations.to_csv(args.output_dir/'validation_depth_ablations.csv', index=False)
    pd.concat(ablation_predictions, ignore_index=True).to_csv(args.output_dir/'validation_ablation_predictions.csv', index=False)
    write_report(args.output_dir, comparison, ablations, args)
    print(comparison.to_string(index=False))
    print(f'Saved controlled comparison and pre-2010 depth ablations to {args.output_dir.resolve()}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f'ERROR: {exc}')
