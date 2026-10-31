"""Select each team's leading regular-season player and expose their stats."""
from pathlib import Path

import numpy as np
import pandas as pd

from nba_predictor.nba_advanced_stats import NBA_STAT_START_SEASON, player_key, team_key


BEST_PLAYER_ADVANCED = ('PER', 'BPM', 'OBPM', 'DBPM', 'WS/48', 'WS', 'VORP', 'TS%', 'USG%')
BEST_PLAYER_BASIC = ('PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV')
BEST_PLAYER_FEATURES = tuple(f'Best_Player_{stat}' for stat in BEST_PLAYER_ADVANCED) + tuple(
    f'Best_Player_{stat}_Per_Game' for stat in BEST_PLAYER_BASIC) + ('Best_Player_G', 'Best_Player_MP_Per_Game')
SECOND_PLAYER_FEATURES = tuple(column.replace('Best_Player_', 'Second_Player_') for column in BEST_PLAYER_FEATURES)
TOP_TWO_PLAYER_FEATURES = (*BEST_PLAYER_FEATURES, *SECOND_PLAYER_FEATURES)
TOP_PLAYER_FEATURES = TOP_TWO_PLAYER_FEATURES
PLAYER_FEATURE_START = {
    **{f'{prefix}_Player_{stat}': NBA_STAT_START_SEASON[stat]
       for prefix in ('Best', 'Second') for stat in BEST_PLAYER_ADVANCED if stat in NBA_STAT_START_SEASON},
    **{f'{prefix}_Player_{stat}_Per_Game': NBA_STAT_START_SEASON[stat]
       for prefix in ('Best', 'Second') for stat in BEST_PLAYER_BASIC if stat in NBA_STAT_START_SEASON},
}
BEST_PLAYER_RULE = ('Highest regular-season stint BPM among players with at least 10 minutes '
                    'per team game; ties use minutes then name. If none qualify, use the '
                    'most-used player with a finite rating. Before 1974, use PER instead '
                    'because BPM is unavailable. Combined-team rows are excluded.')


def rank_team_players(source: pd.DataFrame, metrics: pd.DataFrame):
    """Shared ordering for leaders and depth; source contains cleaned team stints.

    Only finite selection ratings can be ranked. Qualifiers precede fallback
    players; fallback players use minutes, exactly as in the original selector.
    """
    teams = metrics[['Season', 'Team', 'Team_Games']].copy()
    teams['_team_key'] = teams.Team.map(team_key)
    candidates = source.merge(teams[['Season', '_team_key', 'Team_Games']],
                              on=['Season', '_team_key'], how='inner', validate='many_to_one')
    candidates['_rating'] = pd.to_numeric(
        candidates.BPM.where(candidates.Season.ge(1974), candidates.PER), errors='coerce')
    candidates['_rating_name'] = np.where(candidates.Season.ge(1974), 'BPM', 'PER')
    candidates = candidates.loc[np.isfinite(candidates._rating)].copy()
    candidates['_qualifies'] = candidates.MP.ge(10 * candidates.Team_Games)
    # If there is no qualified player, minutes determine the fallback selection.
    candidates['_selection_bpm'] = candidates._rating.where(candidates._qualifies, -np.inf)
    candidates = candidates.sort_values(
        ['Season', '_team_key', '_qualifies', '_selection_bpm', 'MP', '_player_key'],
        ascending=[True, True, False, False, False, True], kind='stable')
    candidates['_player_rank'] = candidates.groupby(['Season', '_team_key']).cumcount() + 1
    return candidates


def select_best_players(source: pd.DataFrame, metrics: pd.DataFrame, stats_path: Path | None, rank: int = 1):
    if rank not in (1, 2):
        raise ValueError('Player rank must be 1 or 2')
    candidates = rank_team_players(source, metrics)
    best = candidates.loc[candidates._player_rank.eq(rank)].copy()
    result = best[['Season', '_team_key']].copy()
    result['Best_Player_Name'] = best.Player
    result['Best_Player_Age'] = best.Age
    result['Best_Player_Selection_Fallback'] = ~best._qualifies
    result['Best_Player_Selection_Metric'] = best._rating_name
    for stat in BEST_PLAYER_ADVANCED:
        result[f'Best_Player_{stat}'] = best[stat] if stat in best else np.nan
    result['Best_Player_G'] = best.G if 'G' in best else np.nan
    result['Best_Player_MP_Per_Game'] = best.MP / result.Best_Player_G.where(result.Best_Player_G > 0)
    for stat in BEST_PLAYER_BASIC:
        result[f'Best_Player_{stat}_Per_Game'] = np.nan
    if stats_path is not None:
        raw = pd.read_csv(stats_path, low_memory=False).drop_duplicates()
        required = {'Season', 'Team', 'Player', 'Age', 'G', *BEST_PLAYER_BASIC}
        if not required.issubset(raw):
            raise ValueError(f'Best-player box-score stats require columns: {sorted(required)}')
        for column in ('Season', 'Age', 'G', *BEST_PLAYER_BASIC):
            raw[column] = pd.to_numeric(raw[column], errors='coerce')
        raw['_team_key'] = raw.Team.map(team_key)
        raw['_player_key'] = raw.Player.map(player_key)
        raw = raw.loc[raw.Season.isin(metrics.Season) & raw._team_key.ne('TOT')].copy()
        keys = ['Season', '_team_key', '_player_key', 'Age']
        if raw.duplicated(keys).any():
            raise ValueError('Ambiguous player stint identities in basic stats')
        selected = best[keys].merge(raw[keys + ['G', *BEST_PLAYER_BASIC]], on=keys,
                                   how='left', validate='one_to_one')
        # Merge per-game values by team; do not rely on row ordering after joins.
        for stat in BEST_PLAYER_BASIC:
            selected[f'Best_Player_{stat}_Per_Game'] = selected[stat] / selected.G.where(selected.G > 0)
        result = result.drop(columns=[f'Best_Player_{stat}_Per_Game' for stat in BEST_PLAYER_BASIC])
        result = result.merge(selected[['Season', '_team_key', *[f'Best_Player_{stat}_Per_Game' for stat in BEST_PLAYER_BASIC]]],
                              on=['Season', '_team_key'], how='left', validate='one_to_one')
    if rank == 2:
        result = result.rename(columns={column: column.replace('Best_Player_', 'Second_Player_') for column in result})
    for column, first_year in PLAYER_FEATURE_START.items():
        if column in result:
            result[column] = result[column].where(result.Season.ge(first_year))
    return result
