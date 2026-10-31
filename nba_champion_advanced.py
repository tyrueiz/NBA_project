"""Regular-season team features from player advanced stats, without concentration."""
from pathlib import Path

import numpy as np
import pandas as pd

from nba_advanced_stats import NBA_STAT_START_SEASON, normalize_advanced, player_key, team_key
from nba_champion_best_player import BEST_PLAYER_FEATURES, SECOND_PLAYER_FEATURES, PLAYER_FEATURE_START, select_best_players


WEIGHTED_STATS = ('PER', 'BPM', 'OBPM', 'DBPM', 'WS/48')
ADDITIVE_STATS = ('WS', 'OWS', 'DWS', 'VORP')
ERA_INDICATOR_FEATURES = ('Era_BPM_Available', 'Era_TOV_USG_Available')
TEAM_ADVANCED_FEATURES = tuple(f'Team_{stat}_MinuteWeighted' for stat in WEIGHTED_STATS) + tuple(
    f'Team_{stat}_Per_Game' for stat in ADDITIVE_STATS) + ('Team_TS%', *ERA_INDICATOR_FEATURES)


def add_team_advanced(metrics: pd.DataFrame, path: Path, stats_path: Path | None = None,
                      include_best_player: bool = True, include_second_player: bool = False):
    """Sum stint contributions; minute-weight player ratings, never combined rows.

    These are roster aggregate proxies, not official team PER/BPM ratings.
    Coverage is measured against team minutes in the existing totals dataset.
    """
    source = normalize_advanced(pd.read_csv(path, low_memory=False)).drop_duplicates()
    required = {'MP', *WEIGHTED_STATS, *ADDITIVE_STATS}
    if not required.issubset(source):
        raise ValueError(f'Advanced team features require columns: {sorted(required)}')
    source = source.loc[source.Season.isin(metrics.Season)].copy()
    for stat, first_year in NBA_STAT_START_SEASON.items():
        if stat in source:
            source[stat] = source[stat].where(source.Season.ge(first_year))
    source = source.loc[~source.Team.str.fullmatch(r'TOT|[2-9]TM', na=False) & source.MP.gt(0)].copy()
    source['_team_key'] = source.Team.map(team_key)
    source['_player_key'] = source.Player.map(player_key)
    if source.duplicated(['Season', '_team_key', '_player_key', 'Age']).any():
        raise ValueError('Duplicate advanced player stint identities; cannot aggregate safely')
    rows = []
    for (season, team), group in source.groupby(['Season', '_team_key']):
        minutes = group.MP.sum()
        row = {'Season': season, '_team_key': team, 'Advanced_Minutes': minutes}
        for stat in (*WEIGHTED_STATS, *ADDITIVE_STATS):
            valid = group[stat].notna() & np.isfinite(group[stat])
            covered_minutes = group.loc[valid, 'MP'].sum()
            row[f'{stat}_Minute_Coverage'] = covered_minutes / minutes
            if stat in WEIGHTED_STATS:
                row[f'Team_{stat}_MinuteWeighted'] = (
                    (group.loc[valid, stat] * group.loc[valid, 'MP']).sum() / covered_minutes
                    if covered_minutes > 0 else np.nan)
            else:
                row[f'Team_{stat}_Total'] = group.loc[valid, stat].sum(min_count=1)
        rows.append(row)
    result = metrics.copy()
    for stat, first_year in NBA_STAT_START_SEASON.items():
        if stat in result:
            result[stat] = result[stat].where(result.Season.ge(first_year))
    result['_team_key'] = result.Team.map(team_key)
    if result.duplicated(['Season', '_team_key']).any():
        raise ValueError('Ambiguous team aliases in team metrics')
    expected_minutes = pd.to_numeric(result.MP, errors='coerce') * pd.to_numeric(result.Team_Games, errors='coerce')
    if expected_minutes.le(0).any() or expected_minutes.isna().any():
        raise ValueError('Team MP and Team_Games must provide positive expected minutes')
    result['_expected_minutes'] = expected_minutes
    # Ensure an empty source still produces an actionable coverage failure.
    derived = pd.DataFrame(rows, columns=['Season', '_team_key', 'Advanced_Minutes',
                                          *[f'{s}_Minute_Coverage' for s in (*WEIGHTED_STATS, *ADDITIVE_STATS)],
                                          *[f'Team_{s}_MinuteWeighted' for s in WEIGHTED_STATS],
                                          *[f'Team_{s}_Total' for s in ADDITIVE_STATS]])
    result = result.merge(derived, on=['Season', '_team_key'], how='left', validate='one_to_one')
    audit = result[['Season', 'Team', 'Advanced_Minutes', '_expected_minutes']].copy()
    audit = audit.rename(columns={'_expected_minutes': 'Expected_Minutes'})
    audit['Minutes_Coverage'] = audit.Advanced_Minutes / audit.Expected_Minutes
    eligible = audit.Minutes_Coverage.between(.9, 1.1)
    reasons = pd.Series('', index=audit.index)
    reasons.loc[~eligible] = 'missing or inconsistent player minutes; '
    for stat in (*WEIGHTED_STATS, *ADDITIVE_STATS):
        coverage_col = f'{stat}_Minute_Coverage'
        audit[coverage_col] = result.pop(coverage_col)
        required_here = audit.Season.ge(NBA_STAT_START_SEASON.get(stat, 0))
        audit[f'{stat}_Required_In_Era'] = required_here
        valid_here = audit[coverage_col].ge(.9) | ~required_here
        eligible &= valid_here
        reasons.loc[~valid_here] += f'insufficient {stat} coverage; '
    audit['Eligible'] = eligible
    games = pd.to_numeric(result.Team_Games, errors='coerce').where(lambda values: values > 0)
    for stat in ADDITIVE_STATS:
        result[f'Team_{stat}_Per_Game'] = result.pop(f'Team_{stat}_Total') / games
    # PTS, FGA, FTA are all already team per-game totals, so units cancel.
    denominator = 2 * (result.FGA + .44 * result.FTA)
    result['Team_TS%'] = result.PTS / denominator.where(denominator > 0)
    result['Era_BPM_Available'] = result.Season.ge(1974).astype(int)
    result['Era_TOV_USG_Available'] = result.Season.ge(1978).astype(int)
    audit['Unavailable_By_Era'] = audit.Season.map(lambda year: ', '.join(
        stat for stat, first_year in NBA_STAT_START_SEASON.items() if year < first_year))
    if include_best_player:
        best = select_best_players(source, metrics, stats_path)
        result = result.merge(best, on=['Season', '_team_key'], how='left', validate='one_to_one')
        audit['Best_Player_Name'] = result.Best_Player_Name
        audit['Best_Player_Selection_Fallback'] = result.Best_Player_Selection_Fallback
        audit['Best_Player_Selection_Metric'] = result.Best_Player_Selection_Metric
        audit['Eligible'] &= result.Best_Player_Name.notna()
        for column in BEST_PLAYER_FEATURES:
            audit[f'{column}_Available'] = result[column].notna()
        if stats_path is not None:
            for column in BEST_PLAYER_FEATURES:
                required_here = result.Season.ge(PLAYER_FEATURE_START.get(column, 0))
                valid_here = np.isfinite(result[column]) | ~required_here
                audit['Eligible'] &= valid_here
                reasons.loc[~valid_here] += f'missing {column}; '
    if include_best_player and include_second_player:
        second = select_best_players(source, metrics, stats_path, rank=2)
        result = result.merge(second, on=['Season', '_team_key'], how='left', validate='one_to_one')
        audit['Second_Player_Name'] = result.Second_Player_Name
        audit['Second_Player_Selection_Metric'] = result.Second_Player_Selection_Metric
        audit['Eligible'] &= result.Second_Player_Name.notna()
        if stats_path is not None:
            for column in SECOND_PLAYER_FEATURES:
                required_here = result.Season.ge(PLAYER_FEATURE_START.get(column, 0))
                valid_here = np.isfinite(result[column]) | ~required_here
                audit['Eligible'] &= valid_here
                reasons.loc[~valid_here] += f'missing {column}; '
    audit['Failure_Reasons'] = reasons.str.rstrip('; ')
    result = result.drop(columns=['_team_key', '_expected_minutes', 'Advanced_Minutes'])
    return result, audit
