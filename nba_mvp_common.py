"""Shared offline MVP regression, using regular-season player totals only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_advanced_stats import ADVANCED_FEATURES, join_advanced


TOTAL_STATS = ('MP', 'PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'ORB', 'DRB',
               'FG', 'FGA', '3P', '3PA', 'FT', 'FTA', 'PF', 'Trp-Dbl')
RATE_STATS = ('FG%', '3P%', '2P%', 'FT%', 'eFG%')
CONTEXT_FEATURES = ('Team_Wins', 'Team_Win_Percentage',
                    'Previous_MVP_Winner', 'Career_MVP_Wins', 'Consecutive_MVP_Wins',
                    'MVP_Wins_Last3', 'Years_Since_MVP_Win')


def add_voting_history(players: pd.DataFrame) -> pd.DataFrame:
    """Use only past MVP wins for fatigue context, never previous voting finishes."""
    players = players.copy()
    # Approximate birth year separates namesakes across seasons without player IDs.
    identity = [players.Player, players.Season - players.Age]
    history_columns = list(CONTEXT_FEATURES[2:])
    history_values = np.zeros((len(players), len(history_columns)))
    for _, group in players.groupby(identity, dropna=False, sort=False):
        ranks = group.groupby('Season').MVP_Rank_LABEL_ONLY.min().to_dict()
        wins = sorted(year for year, rank in ranks.items() if rank == 1)
        for year, rows in group.groupby('Season'):
            prior_wins = [season for season in wins if season < year]
            streak = 0
            while year - streak - 1 in prior_wins:
                streak += 1
            values = [float(year - 1 in prior_wins), len(prior_wins), streak,
                      sum(year - 3 <= season < year for season in prior_wins),
                      year - prior_wins[-1] if prior_wins else 0]
            history_values[players.index.get_indexer(rows.index)] = values
    players[history_columns] = history_values
    return players


def add_team_context(players: pd.DataFrame, stats_path: Path, team_path: Path) -> pd.DataFrame:
    """Attach regular-season wins; weight traded players' teams by stint games."""
    teams = pd.read_csv(team_path, low_memory=False)
    required = {'Season', 'Team', 'Wins', 'Losses'}
    if not required.issubset(teams):
        raise ValueError(f'Team CSV requires {sorted(required)}')
    teams = teams[['Season', 'Team', 'Wins', 'Losses']].copy()
    for column in ('Season', 'Wins', 'Losses'):
        teams[column] = pd.to_numeric(teams[column], errors='coerce')
    teams['Team'] = teams.Team.astype(str).str.strip().str.upper()
    if teams.duplicated(['Season', 'Team']).any():
        raise ValueError('Duplicate team-season keys in team CSV')
    teams['Team_Wins'] = teams.Wins
    total_games = teams.Wins + teams.Losses
    teams['Team_Win_Percentage'] = teams.Wins / total_games.where(total_games > 0)
    stints = pd.read_csv(stats_path, usecols=['Season', 'Player', 'Age', 'Team', 'G']).drop_duplicates()
    for column in ('Season', 'Age', 'G'):
        stints[column] = pd.to_numeric(stints[column], errors='coerce')
    stints['Player'] = stints.Player.astype(str).str.strip()
    stints['Team'] = stints.Team.astype(str).str.strip().str.upper()
    aliases = {'BKN': 'BRK', 'BRK': 'BKN', 'CHA': 'CHO', 'CHO': 'CHA',
               'PHX': 'PHO', 'PHO': 'PHX', 'NOK': 'NOH', 'NOH': 'NOK'}
    known = set(zip(teams.Season, teams.Team))
    stints['Team'] = [team if (year, team) in known else aliases.get(team, team)
                      for year, team in zip(stints.Season, stints.Team)]
    stints = stints.loc[~stints.Team.str.fullmatch(r'TOT|[2-9]TM', na=False)]
    stints = stints.merge(teams[['Season', 'Team', *CONTEXT_FEATURES[:2]]],
                          on=['Season', 'Team'], how='left', validate='many_to_one')
    keys = ['Season', 'Player', 'Age']
    weighted = stints[keys].copy()
    valid = stints.G.gt(0) & stints.Team_Win_Percentage.notna() & stints.Team_Wins.notna()
    weighted['_games'] = stints.G.where(valid, 0)
    for column in CONTEXT_FEATURES[:2]:
        weighted[column] = stints[column].fillna(0) * weighted._games
    combined = weighted.groupby(keys, dropna=False).sum(numeric_only=True)
    for column in CONTEXT_FEATURES[:2]:
        combined[column] /= combined._games.where(combined._games > 0)
    # Single-team rows match on team as well, preserving same-name players.
    result = players.merge(teams[['Season', 'Team', *CONTEXT_FEATURES[:2]]],
                           on=['Season', 'Team'], how='left', validate='many_to_one')
    combined = combined[list(CONTEXT_FEATURES[:2])].add_suffix('_Combined').reset_index()
    result = result.merge(combined, on=keys, how='left', validate='many_to_one')
    aggregate = result.Team.str.fullmatch(r'TOT|[2-9]TM', na=False)
    for column in CONTEXT_FEATURES[:2]:
        result.loc[aggregate, column] = result.loc[aggregate, f'{column}_Combined']
        result = result.drop(columns=f'{column}_Combined')
    # Alias translation for non-aggregate player rows too.
    missing = result.Team_Win_Percentage.isna() & ~aggregate
    lookup = teams.set_index(['Season', 'Team'])
    for idx in result.index[missing]:
        key = (result.at[idx, 'Season'], aliases.get(result.at[idx, 'Team'], result.at[idx, 'Team']))
        if key in lookup.index:
            result.loc[idx, list(CONTEXT_FEATURES[:2])] = lookup.loc[key, list(CONTEXT_FEATURES[:2])].to_numpy()
    return result


def load_players(path: Path) -> pd.DataFrame:
    players = pd.read_csv(path, low_memory=False).drop_duplicates()
    players.columns = players.columns.str.strip()
    required = {'Season', 'Player', 'Age', 'Team', 'G', 'PTS', 'FGA', 'FTA', 'Awards'}
    missing = required - set(players)
    if missing:
        raise ValueError(f'Player CSV is missing columns: {sorted(missing)}')
    players['Season'] = pd.to_numeric(players.Season, errors='coerce')
    players = players.dropna(subset=['Season', 'Player', 'Team']).copy()
    players['Season'] = players.Season.astype(int)
    players['Team'] = players.Team.astype(str).str.strip().str.upper()
    players['Player'] = players.Player.astype(str).str.strip()
    for column in ('Age', 'G', *TOTAL_STATS, *RATE_STATS, *ADVANCED_FEATURES):
        if column in players:
            players[column] = pd.to_numeric(players[column], errors='coerce')

    # A traded player's aggregate row already contains all stints. Age helps
    # distinguish different people with the same name in this source without IDs.
    players['_aggregate'] = players.Team.str.fullmatch(r'TOT|[2-9]TM', na=False)
    keys = ['Season', 'Player', 'Age']
    has_aggregate = players.groupby(keys, dropna=False)['_aggregate'].transform('any')
    players = players.loc[players._aggregate | ~has_aggregate].copy()
    aggregate = players.loc[players._aggregate]
    if aggregate.duplicated(keys).any():
        raise ValueError('Ambiguous aggregate player rows; supply a CSV with unique player identities.')
    # Preserve namesakes without aggregate rows instead of summing their stats.
    players = players.drop(columns='_aggregate').reset_index(drop=True)
    players['Candidate_ID'] = players.index
    players['MVP_Rank_LABEL_ONLY'] = pd.to_numeric(
        players.Awards.fillna('').str.extract(r'(?:^|,)\s*MVP-(\d+)\s*(?=,|$)', expand=False),
        errors='coerce',
    )
    if (players.MVP_Rank_LABEL_ONLY.dropna() < 1).any():
        raise ValueError('MVP ranks must be positive.')
    players['MVP_Target_LABEL_ONLY'] = 1 / players.MVP_Rank_LABEL_ONLY
    players['MVP_Target_LABEL_ONLY'] = players.MVP_Target_LABEL_ONLY.fillna(0)
    return players


def make_features(players: pd.DataFrame) -> pd.DataFrame:
    """Allowlist stats and precomputed history; exclude same-season awards."""
    features = players[['Age', 'G']].copy()
    for column in CONTEXT_FEATURES:
        if column in players:
            features[column] = players[column]
    games = players.G.where(players.G > 0)
    for stat in TOTAL_STATS:
        if stat in players:
            features[f'{stat}_Per_Game'] = players[stat] / games
    for stat in RATE_STATS:
        if stat in players:
            features[stat] = players[stat]
    for stat in ADVANCED_FEATURES:
        if stat in players:
            features[stat] = players[stat]
    shooting_possessions = 2 * (players.FGA + .44 * players.FTA)
    if 'TS%' not in features:
        features['TS%'] = players.PTS / shooting_possessions.where(shooting_possessions > 0)
    features['3PA_Rate'] = players.get('3PA', np.nan) / players.FGA.where(players.FGA > 0)
    features = features.replace([np.inf, -np.inf], np.nan)
    # Relative standing within a season accounts for changes in scoring eras.
    # This uses only contemporaneous stats, never that season's award labels.
    grouped = features.groupby(players.Season)
    return ((features - grouped.transform('mean')) /
            grouped.transform('std').replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def make_model(kind: str) -> Pipeline:
    if kind == 'linear':
        estimator = LinearRegression()
    else:
        from xgboost import XGBRegressor
        estimator = XGBRegressor(
            objective='reg:squarederror', n_estimators=240, max_depth=2,
            learning_rate=.04, min_child_weight=3, subsample=.85,
            colsample_bytree=.85, reg_lambda=7, random_state=42,
            tree_method='hist', n_jobs=2,
        )
    return Pipeline([('imputer', SimpleImputer(strategy='median')),
                     ('scaler', StandardScaler()), ('regressor', estimator)])


def summarize(predictions: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in predictions.groupby('Season', sort=True):
        ranked = group.sort_values(['Rank', 'Candidate_ID'])
        winners = players.loc[players.Season.eq(year) & players.MVP_Rank_LABEL_ONLY.eq(1)]
        if len(winners) > 1:
            raise ValueError(f'Multiple MVP winners in {year}; inspect player identities and labels.')
        actual = winners.iloc[0] if len(winners) else None
        present = actual is not None and actual.Candidate_ID in set(group.Candidate_ID)
        rows.append({
            'Season': int(year), 'Predicted_MVP': ranked.iloc[0].Player,
            'Predicted_Team': ranked.iloc[0].Team,
            'Predicted_2nd': ranked.iloc[1].Player if len(ranked) >= 2 else None,
            'Predicted_2nd_Team': ranked.iloc[1].Team if len(ranked) >= 2 else None,
            'Predicted_3rd': ranked.iloc[2].Player if len(ranked) >= 3 else None,
            'Predicted_3rd_Team': ranked.iloc[2].Team if len(ranked) >= 3 else None,
            'Actual_MVP_EVALUATION_ONLY': actual.Player if actual is not None else None,
            'Actual_In_Input': present if actual is not None else None,
            'Correct': bool(ranked.iloc[0].Candidate_ID == actual.Candidate_ID) if present else None,
            'Actual_In_Top3': bool(actual.Candidate_ID in set(ranked.head(3).Candidate_ID)) if present else None,
            'Players_Scored': len(group),
        })
    return pd.DataFrame(rows)


def score(model: Pipeline, features: pd.DataFrame, players: pd.DataFrame,
          mask: pd.Series) -> pd.DataFrame:
    result = players.loc[mask, ['Candidate_ID', 'Season', 'Player', 'Age', 'Team', 'G']].copy()
    result['Model_Score'] = model.predict(features.loc[mask])
    result['Rank'] = result.groupby('Season').Model_Score.rank(ascending=False, method='first').astype(int)
    result['Predicted_MVP'] = result.Rank.eq(1)
    result['Predicted_Top3'] = result.Rank.le(3)
    return result.sort_values(['Season', 'Rank'])


def main(kind: str) -> int:
    parser = argparse.ArgumentParser(description=f'{kind} regression NBA MVP predictor from regular-season stats.')
    parser.add_argument('--input-dir', type=Path, default=Path('nba_updated'))
    parser.add_argument('--stats-file', type=Path, help='Override the regular-season player totals CSV')
    parser.add_argument('--team-file', type=Path, help='Override regular-season team results CSV')
    parser.add_argument('--advanced-file', type=Path, help='Advanced regular-season CSV; auto-detected by default')
    parser.add_argument('--no-advanced', action='store_true', help='Explicitly run the box-score baseline')
    parser.add_argument('--output-dir', type=Path, default=Path(f'nba_mvp_predictions_{kind}'))
    parser.add_argument('--train-start', type=int, default=1980)
    parser.add_argument('--train-end', type=int, default=2010)
    parser.add_argument('--predict-start', type=int, default=2011)
    parser.add_argument('--predict-end', type=int, default=2026)
    parser.add_argument('--validation-start', type=int, default=2015,
                        help='First year of expanding-window historical validation')
    args = parser.parse_args()
    if not args.train_start <= args.train_end < args.predict_start <= args.predict_end:
        parser.error('Require train-start <= train-end < predict-start <= predict-end')
    if args.no_advanced and args.advanced_file:
        parser.error('--advanced-file and --no-advanced cannot be combined')
    stats_path = args.stats_file
    if stats_path is None:
        candidates = sorted(args.input_dir.glob('NBA_stats_[0-9]*_[0-9]*.csv'),
                            key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
        if not candidates:
            parser.error('No regular-season NBA_stats CSV found; use --stats-file')
        stats_path = candidates[0]
    team_path = args.team_file
    if team_path is None:
        candidates = sorted(args.input_dir.glob('NBA_team_seasons_*.csv'),
                            key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
        if not candidates:
            parser.error('No team results CSV found; use --team-file')
        team_path = candidates[0]
    players = add_voting_history(load_players(stats_path))
    players = add_team_context(players, stats_path, team_path)
    players = players.loc[players.Season.between(args.train_start, args.predict_end) & players.G.gt(0)].copy()
    advanced_path = None
    advanced_coverage = None
    if args.no_advanced:
        players = players.drop(columns=list(ADVANCED_FEATURES), errors='ignore')
    if not args.no_advanced:
        advanced_path = args.advanced_file
        if advanced_path is None:
            candidates = sorted(args.input_dir.glob('NBA_advanced_*.csv'),
                                key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
            if not candidates:
                parser.error('Advanced stats are required. Run python3 nba_download_advanced.py '
                             'first, provide --advanced-file, or use --no-advanced for the baseline.')
            advanced_path = candidates[0]
        players, advanced_coverage = join_advanced(players, advanced_path)
        needed = advanced_coverage.Season.between(args.train_start, args.train_end) | advanced_coverage.Season.between(args.predict_start, args.predict_end)
        core = ['PER_Coverage', 'WS_Coverage', 'BPM_Coverage', 'VORP_Coverage']
        adequate = advanced_coverage[core].ge(.9).all(axis=1)
        if (needed & ~adequate).any():
            bad = advanced_coverage.loc[needed & ~adequate, 'Season'].tolist()
            raise ValueError(f'Advanced-stat coverage below 90% in seasons {bad}. '
                             'Complete the download or explicitly use --no-advanced for the baseline.')
    counts = players.loc[players.MVP_Rank_LABEL_ONLY.eq(1)].groupby('Season').size()
    if counts.gt(1).any():
        raise ValueError(f'Multiple MVP winners: {counts[counts.gt(1)].to_dict()}')
    training_years = list(range(args.train_start, args.train_end + 1))
    missing = [year for year in training_years if counts.get(year, 0) != 1]
    if missing:
        raise ValueError(f'Missing MVP winner labels for training seasons: {missing}')
    if len(training_years) < 10:
        raise ValueError('At least ten labeled training seasons are required.')
    missing_prediction = set(range(args.predict_start, args.predict_end + 1)) - set(players.Season)
    if missing_prediction:
        raise ValueError(f'Missing player stats for prediction seasons: {sorted(missing_prediction)}')
    train = players.Season.between(args.train_start, args.train_end)
    target = players.Season.between(args.predict_start, args.predict_end)
    features = make_features(players)
    usable = features.loc[train].notna().mean().ge(.5) & features.loc[train].std().gt(1e-10)
    features = features.loc[:, usable]
    if features.empty:
        raise ValueError('No usable regular-season features.')
    model = make_model(kind)
    model.fit(features.loc[train], players.loc[train, 'MVP_Target_LABEL_ONLY'])
    predictions = score(model, features, players, target)
    results = summarize(predictions, players)

    # Each validation season is predicted by a fresh model fitted only on prior
    # years. Feature coverage is also selected independently inside each fold.
    validation = []
    for year in range(max(args.validation_start, args.train_start + 10), args.train_end + 1):
        before = players.Season.between(args.train_start, year - 1)
        held_out = players.Season.eq(year)
        fold_features = make_features(players)
        keep = fold_features.loc[before].notna().mean().ge(.5) & fold_features.loc[before].std().gt(1e-10)
        fold_features = fold_features.loc[:, keep]
        validator = make_model(kind)
        validator.fit(fold_features.loc[before], players.loc[before, 'MVP_Target_LABEL_ONLY'])
        validation.append(summarize(score(validator, fold_features, players, held_out), players))
    historical = pd.concat(validation, ignore_index=True) if validation else results.iloc[:0].copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / 'player_predictions.csv', index=False)
    top3 = predictions.loc[predictions.Predicted_Top3].copy()
    top3.to_csv(args.output_dir / 'top3_predictions.csv', index=False)
    results.to_csv(args.output_dir / 'predicted_mvps.csv', index=False)
    historical.to_csv(args.output_dir / 'historical_validation.csv', index=False)
    if advanced_coverage is not None:
        advanced_coverage.to_csv(args.output_dir / 'advanced_stat_coverage.csv', index=False)
    regressor = model.named_steps['regressor']
    importance = regressor.coef_ if kind == 'linear' else regressor.feature_importances_
    pd.DataFrame({'Feature': features.columns,
                  'Coefficient' if kind == 'linear' else 'Importance': importance}).to_csv(
                      args.output_dir / 'feature_importance.csv', index=False)
    details = {
        'model': type(regressor).__name__, 'stats_file': str(stats_path),
        'training_min_season': args.train_start, 'training_max_season': args.train_end,
        'number_of_training_seasons': len(training_years), 'training_rows': int(train.sum()),
        'prediction_years': [args.predict_start, args.predict_end],
        'features': list(features.columns), 'target': '1 / MVP voting rank; zero for unranked players',
        'same_season_award_features_used': False, 'historical_award_features_used': True,
        'history_cutoff': 'Strictly before each row season, including pre-training history. '
                          'Earlier prediction-season awards may inform later seasons.',
        'history_features': 'Past MVP wins only; non-winning voting ranks are excluded. '
                            'Fatigue effects are learned, with no fixed penalty.',
        'team_file': str(team_path), 'postseason_features_used': False,
        'advanced_file': str(advanced_path) if advanced_path else None,
        'advanced_features_used': [column for column in features if column in ADVANCED_FEATURES],
        'team_win_percentage_coverage': float(players.Team_Win_Percentage.notna().mean()),
        'team_distribution_features_used': False,
        'validation_seasons': len(historical),
        'validation_top1_accuracy': float(historical.Correct.mean()) if len(historical) else None,
        'validation_top3_accuracy': float(historical.Actual_In_Top3.mean()) if len(historical) else None,
        'note': 'Scores are regression outputs, not probabilities or actual vote shares. '
                'Predictions use same-season stats; partial seasons give provisional rankings. '
                'No official award eligibility rule is imposed. Player identities use names and age '
                'because the source lacks player IDs; namesakes can be ambiguous.',
    }
    (args.output_dir / 'model_details.json').write_text(json.dumps(details, indent=2), encoding='utf-8')
    print(f'{details["model"]}: trained on {args.train_start}–{args.train_end}, '
          f'{int(train.sum())} player-seasons, {len(features.columns)} features.')
    if len(historical):
        print(f'Historical validation ({len(historical)} seasons): '
              f'top-1 {details["validation_top1_accuracy"]:.1%}, '
              f'top-3 {details["validation_top3_accuracy"]:.1%}')
    print('Predicted top three MVP candidates by season:')
    print(top3[['Season', 'Rank', 'Player', 'Team', 'Model_Score']].to_string(index=False))
    print(f'Saved to {args.output_dir.resolve()}')
    return 0
