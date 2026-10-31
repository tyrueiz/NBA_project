#!/usr/bin/env python3
"""Search chronological champion-model windows and rank by validation results.

Default: every eligible train start/cutoff/validation end from 1973 through
2025, at least ten training seasons. 2026 is never used to select a window.
Models stay fixed after their training cutoff; this is not annual retraining.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from nba_champion_advanced import add_team_advanced
from nba_team_ratings import TEAM_RATING_FEATURES, add_team_ratings
from nba_champion_first_time import add_first_time_features
from nba_predict_champions_2010_2026 import (
    available_features, load_champion_labels, locate, normalize_frame,
    season_alias_map, to_season_zscores,
)
from nba_predict_champions_xgboost_2010_2026 import make_xgb_pipeline


def training_windows(first_year: int, last_validation_year: int,
                     min_train: int, min_validation: int):
    for cutoff in range(first_year + min_train - 1, last_validation_year - min_validation + 1):
        for start in range(first_year, cutoff - min_train + 2):
            yield start, cutoff


def wilson_lower(wins: int, count: int) -> float:
    """95% lower bound: one lucky season is weaker evidence than many seasons."""
    if count == 0:
        return 0.0
    z = 1.96
    p = wins / count
    return (p + z*z/(2*count) - z*math.sqrt(p*(1-p)/count + z*z/(4*count*count))) / (1 + z*z/count)


def period_metrics(rows: list[dict], start: int, end: int, prefix: str) -> dict:
    selected = [row for row in rows if start <= row['Season'] <= end and row['Actual_Champion'] is not None]
    n = len(selected)
    wins = sum(row['Correct'] for row in selected)
    return {
        f'{prefix}_Seasons': n, f'{prefix}_Correct': wins,
        f'{prefix}_Top1': wins/n if n else None,
        f'{prefix}_Top3': sum(row['Actual_In_Top3'] for row in selected)/n if n else None,
        f'{prefix}_Mean_Reciprocal_Rank': sum(1/row['Actual_Rank'] for row in selected)/n if n else None,
        f'{prefix}_Wilson_Lower': wilson_lower(wins, n),
    }


def leaderboard(fits: list[dict], last_validation_year: int,
                test_end: int, min_validation: int) -> pd.DataFrame:
    rows = []
    for fit in fits:
        validation_start = fit['Train_End'] + 1
        for end in range(validation_start + min_validation - 1, last_validation_year + 1):
            row = {key: fit[key] for key in ('Model', 'Train_Start', 'Train_End', 'Features_Used')}
            row.update(Validation_Start=validation_start, Validation_End=end,
                       Later_Test_Start=end + 1, Later_Test_End=test_end)
            row.update(period_metrics(fit['Predictions'], validation_start, end, 'Validation'))
            row.update(period_metrics(fit['Predictions'], end + 1, test_end, 'Later_Test'))
            rows.append(row)
    return rank_leaderboard(pd.DataFrame(rows))


def rank_leaderboard(table: pd.DataFrame) -> pd.DataFrame:
    # No test column participates in ordering or tie-breaking.
    result = table.drop(columns='Rank', errors='ignore').sort_values(
        ['Validation_Wilson_Lower', 'Validation_Top1', 'Validation_Top3',
         'Validation_Mean_Reciprocal_Rank', 'Validation_Seasons',
         'Train_End', 'Train_Start', 'Model', 'Validation_End'],
        ascending=[False, False, False, False, False, False, True, True, True], kind='stable')
    result.insert(0, 'Rank', range(1, len(result) + 1))
    return result.reset_index(drop=True)


def prepare_data(args):
    metric_path = args.analysis_dir/'team_metrics_all_seasons.csv'
    season_path = locate(args.input_dir, 'NBA_team_seasons_*.csv')
    advanced_path = args.advanced_file or locate(args.input_dir, 'NBA_advanced_*.csv')
    stats_path = args.stats_file or locate(args.input_dir, 'NBA_stats_[0-9]*_[0-9]*.csv')
    comparison_path = args.analysis_dir/'champions_vs_league_all_stats.csv'
    ratings_path = args.team_ratings_file or locate(args.input_dir, 'NBA_team_ratings_*.csv')
    sources = [metric_path, season_path, advanced_path, stats_path, ratings_path]
    comparisons = None
    if comparison_path.exists() and comparison_path.stat().st_size:
        sources.append(comparison_path)
        try:
            comparisons = pd.read_csv(comparison_path, low_memory=False)
        except pd.errors.EmptyDataError:
            pass
    metrics = normalize_frame(pd.read_csv(metric_path, low_memory=False), 'team metrics')
    seasons = normalize_frame(pd.read_csv(season_path, low_memory=False), 'team results')
    for frame in (metrics, seasons):
        if frame.duplicated(['Season', 'Team']).any():
            raise ValueError('Duplicate team-season keys')
    labels = load_champion_labels(seasons, comparisons)
    translate = season_alias_map(metrics)
    labels = {year: translate(year, team) or team for year, team in labels.items()}
    metrics = metrics.loc[metrics.Season.between(args.first_year, args.test_end)].copy()
    metrics, coverage = add_team_advanced(metrics, advanced_path, stats_path, include_second_player=True)
    player_columns = [column for column in metrics if column.startswith(('Best_Player_', 'Second_Player_'))]
    metrics[['Season', 'Team', *player_columns]].to_csv(args.output_dir/'top_two_player_features.csv', index=False)
    coverage.to_csv(args.output_dir/'advanced_stat_coverage.csv', index=False)
    if not coverage.Eligible.all():
        raise ValueError('Incomplete advanced stats; see advanced_stat_coverage.csv')
    metrics, ratings_audit = add_team_ratings(metrics, ratings_path)
    ratings_audit.to_csv(args.output_dir/'team_rating_coverage.csv', index=False)
    if not ratings_audit.Eligible.all():
        raise ValueError('Incomplete actual team ratings; see team_rating_coverage.csv')
    metrics, experience = add_first_time_features(metrics, seasons, stats_path)
    experience.to_csv(args.output_dir/'first_time_contender_features.csv', index=False)
    audits = []
    for year in range(args.first_year, args.test_end + 1):
        group = metrics.loc[metrics.Season.eq(year)]
        expected = seasons.loc[seasons.Season.eq(year), 'Team'].nunique()
        reason = 'OK'
        if not expected or len(group)/expected < args.min_team_coverage:
            reason = 'insufficient team coverage'
        if 'Season_Status' in seasons and seasons.loc[seasons.Season.eq(year), 'Season_Status'].astype(str).str.lower().eq('in progress').any():
            reason = 'season in progress'
        # Require complete labels for every season searched. Final holdout may
        # be unlabeled; it still receives a prediction without a scored result.
        if year <= args.last_validation_year and labels.get(year) not in set(group.Team):
            reason = 'missing historical champion label/team'
        if year > args.last_validation_year and labels.get(year) is not None and labels[year] not in set(group.Team):
            reason = 'holdout champion absent from input'
        audits.append(dict(Season=year, Teams=len(group), Expected_Teams=expected, Reason=reason))
    audit = pd.DataFrame(audits)
    audit.to_csv(args.output_dir/'season_data_coverage.csv', index=False)
    if not audit.Reason.eq('OK').all():
        raise ValueError('Some seasons are unavailable; see season_data_coverage.csv. '
                         'The search requires consecutive seasons rather than silently skipping years.')
    metrics = metrics.reset_index(drop=True)
    metrics['_label'] = [int(labels.get(year) == team) for year, team in zip(metrics.Season, metrics.Team)]
    # A label-free superset for preprocessing; actual feature selection is
    # repeated independently using each candidate's training mask only.
    pool = available_features(metrics, pd.Series(True, index=metrics.index), 0, 10000)
    zscores = to_season_zscores(metrics, pool)
    return metrics, zscores, labels, sources


def fit_window(model_name, start, cutoff, frame, zscores, labels, args):
    cache = args.output_dir/'fit_cache'/f'{model_name}_{start}_{cutoff}.json'
    if cache.exists():
        return json.loads(cache.read_text())
    train = frame.Season.between(start, cutoff)
    target = frame.Season.gt(cutoff)
    candidates = available_features(frame, train, args.min_feature_coverage, args.max_features)
    candidates = [column for column in candidates if zscores.loc[train, column].notna().any()]
    if not candidates:
        raise ValueError(f'No features for {start}–{cutoff}')
    y = frame.loc[train, '_label'].astype(int)
    if model_name == 'logistic':
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(C=.20, class_weight='balanced', max_iter=3000, random_state=42)),
        ])
    else:
        model = make_xgb_pipeline(y, args)
        model.set_params(classifier__n_jobs=1)
    model.fit(zscores.loc[train, candidates], y)
    predictions = frame.loc[target, ['Season', 'Team']].copy()
    predictions['Score'] = model.predict_proba(zscores.loc[target, candidates])[:, 1]
    results = []
    for year, group in predictions.groupby('Season', sort=True):
        ranked = group.sort_values('Score', ascending=False, kind='stable')
        teams = ranked.Team.tolist()
        actual = labels.get(int(year))
        actual_rank = teams.index(actual) + 1 if actual in teams else None
        results.append(dict(Season=int(year), Predicted_Champion=teams[0],
                            Predicted_2nd=teams[1], Predicted_3rd=teams[2],
                            Actual_Champion=actual, Actual_Rank=actual_rank,
                            Correct=actual_rank == 1, Actual_In_Top3=actual_rank is not None and actual_rank <= 3))
    fit = dict(Model=model_name, Train_Start=start, Train_End=cutoff,
               Features_Used=len(candidates), Features=candidates, Predictions=results)
    temporary = cache.with_suffix('.tmp')
    temporary.write_text(json.dumps(fit, indent=2), encoding='utf-8')
    temporary.replace(cache)
    return fit


def fingerprint(args, sources):
    settings = {key: value for key, value in vars(args).items()
                if key not in ('output_dir', 'workers', 'models')}
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode())
    code = [Path(__file__), *Path(__file__).parent.glob('nba_champion_*.py'),
            Path(__file__).with_name('nba_advanced_stats.py'),
            Path(__file__).with_name('nba_team_ratings.py'),
            Path(__file__).with_name('nba_download_team_ratings.py'),
            Path(__file__).with_name('nba_predict_champions_2010_2026.py'),
            Path(__file__).with_name('nba_predict_champions_xgboost_2010_2026.py')]
    for path in sorted(set([*sources, *code])):
        digest.update(str(path.resolve()).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('nba_updated'))
    parser.add_argument('--analysis-dir', type=Path, default=Path('nba_champions_analysis'))
    parser.add_argument('--advanced-file', type=Path)
    parser.add_argument('--stats-file', type=Path)
    parser.add_argument('--team-ratings-file', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('nba_champion_window_search_top2_ratings'))
    parser.add_argument('--first-year', type=int, default=1973)
    parser.add_argument('--last-validation-year', type=int, default=2025)
    parser.add_argument('--test-end', type=int, default=2026)
    parser.add_argument('--min-train-seasons', type=int, default=10)
    parser.add_argument('--min-validation-seasons', type=int, default=1)
    parser.add_argument('--max-features', type=int, default=59)
    parser.add_argument('--min-feature-coverage', type=float, default=.35)
    parser.add_argument('--min-team-coverage', type=float, default=.75)
    parser.add_argument('--n-estimators', type=int, default=240)
    parser.add_argument('--max-depth', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=.04)
    parser.add_argument('--models', nargs='+', choices=['logistic', 'xgboost'], default=['logistic', 'xgboost'])
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args(argv)
    if not 1973 <= args.first_year < args.last_validation_year <= 2025 < args.test_end <= 2026:
        parser.error('Require 1973 <= first-year < last-validation-year <= 2025 < test-end <= 2026')
    if args.min_train_seasons < 10 or args.min_validation_seasons < 1 or args.workers < 1 or args.max_features < 1:
        parser.error('Require at least ten training seasons, one validation season, and positive workers/features')
    if args.n_estimators < 1 or not 1 <= args.max_depth <= 10 or not 0 < args.learning_rate <= 1:
        parser.error('Invalid XGBoost parameters')
    windows = list(training_windows(args.first_year, args.last_validation_year,
                                    args.min_train_seasons, args.min_validation_seasons))
    if not windows:
        parser.error('Date range cannot provide the required training and validation seasons')
    args.models = list(dict.fromkeys(args.models))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print('Preparing team ratings, top-two-player, and prior-history features once ...', flush=True)
    frame, zscores, labels, sources = prepare_data(args)
    signature = fingerprint(args, sources)
    manifest_path = args.output_dir/'search_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())['fingerprint'] != signature:
        raise ValueError('Inputs or settings changed; choose a new --output-dir to keep cached fits separate.')
    manifest = dict(fingerprint=signature, Settings=vars(args).copy(),
                    Test_Selection_Policy='Rank only on validation through 2025. 2026 is never used for selection.',
                    Training_Mode='Fixed window; no refit after validation.',
                    Ranking='Validation Wilson 95% lower bound, top1, top3, reciprocal rank, count; never test scores.')
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding='utf-8')
    (args.output_dir/'fit_cache').mkdir(exist_ok=True)
    tasks = [(model, start, cutoff) for model in args.models for start, cutoff in windows]
    total_windows = len(args.models) * sum(args.last_validation_year - cutoff - args.min_validation_seasons + 1
                                          for _, cutoff in windows)
    print(f'{len(tasks):,} fitted models cover {total_windows:,} date-window combinations. '
          'Completed fits are cached for resuming.', flush=True)
    fits, started = [], time.monotonic()
    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = {executor.submit(fit_window, model, start, cutoff, frame, zscores, labels, args):
                   (model, start, cutoff) for model, start, cutoff in tasks}
        for future in as_completed(pending):
            fits.append(future.result())
            if len(fits) % 25 == 0 or len(fits) == len(tasks):
                print(f'Completed {len(fits):,}/{len(tasks):,} fits in {time.monotonic()-started:.0f}s', flush=True)
    table = leaderboard(fits, args.last_validation_year, args.test_end, args.min_validation_seasons)
    table.to_csv(args.output_dir/'all_windows_leaderboard.csv', index=False)
    # Every contender in this final comparison uses the SAME last validation
    # year and the SAME final holdout; test performance cannot alter the ranks.
    final = rank_leaderboard(table.loc[table.Validation_End.eq(args.last_validation_year)])
    final.to_csv(args.output_dir/'final_holdout_leaderboard.csv', index=False)
    yearly = []
    for end, group in table.groupby('Validation_End', sort=True):
        winner = rank_leaderboard(group).iloc[0].to_dict()
        yearly.append(winner)
    pd.DataFrame(yearly).to_csv(args.output_dir/'winners_by_validation_end.csv', index=False)
    predictions = []
    for fit in fits:
        for row in fit['Predictions']:
            predictions.append({key: fit[key] for key in ('Model', 'Train_Start', 'Train_End')} | row)
    pd.DataFrame(predictions).sort_values(['Model', 'Train_Start', 'Train_End', 'Season']).to_csv(
        args.output_dir/'all_season_predictions.csv', index=False)
    manifest['Completed_Fits'] = len(fits)
    manifest['Date_Window_Combinations'] = len(table)
    manifest['Elapsed_Fit_Seconds'] = time.monotonic() - started
    manifest['Selected_Final_Window'] = final.iloc[0].to_dict()
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding='utf-8')
    print('\nTop windows using the final validation cutoff (holdout does not affect rank):')
    print(final.head(10)[['Rank', 'Model', 'Train_Start', 'Train_End', 'Validation_End',
                          'Validation_Seasons', 'Validation_Top1', 'Validation_Top3',
                          'Later_Test_Top1']].to_string(index=False))
    print(f'Saved leaderboards to {args.output_dir.resolve()}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f'ERROR: {exc}')
