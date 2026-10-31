"""Optional regular-season depth inputs shared by both championship models."""
from pathlib import Path

import numpy as np
import pandas as pd

from nba_predictor.nba_advanced_stats import normalize_advanced, player_key, team_key
from nba_predictor.nba_champion_best_player import rank_team_players


ROSTER_DEPTH_FEATURES = ('Third_Player_BPM', 'Fourth_Player_BPM',
                         'Supporting_Cast_BPM', 'Rotation_Depth', 'Top_2_Minutes_Share')
ROSTER_DEPTH_RULE = (
    'Same team-stint BPM/PER ranking as leaders, 10 minutes per team game threshold. '
    'Third/fourth can use the existing minutes fallback. Supporting cast uses only '
    'qualifying players at ranks 3–8, weighted by their team-stint minutes. '
    'Rotation counts all players with at least 12 minutes per team game. '
    'Top-two share uses the same selected leaders and all positive team-stint minutes. '
    'BPM features are missing before 1974; PER is used only for ranking.')


def add_roster_depth(metrics: pd.DataFrame, path: Path):
    """Return joined inputs, team diagnostics, and per-player ranking diagnostics.

    This function reads no team results, awards, or playoff history. Exact
    duplicate source rows are removed; conflicting duplicate stint identities
    are rejected. Missing BPM never enters a weighted average as zero.
    """
    source = normalize_advanced(pd.read_csv(path, low_memory=False)).drop_duplicates()
    if not {'MP', 'PER', 'BPM'}.issubset(source):
        raise ValueError('Roster depth requires MP, PER, and BPM')
    source = source.loc[source.Season.isin(metrics.Season) &
                        ~source.Team.map(team_key).eq('TOT') & source.MP.gt(0)].copy()
    source['_team_key'] = source.Team.map(team_key)
    source['_player_key'] = source.Player.map(player_key)
    identity = ['Season', '_team_key', '_player_key', 'Age']
    if source.duplicated(identity).any():
        raise ValueError('Duplicate advanced player stint identities in roster depth')
    if not np.isfinite(source.MP).all():
        raise ValueError('Roster depth requires finite player minutes')
    source['BPM'] = source.BPM.where(source.Season.ge(1974))
    teams = metrics[['Season', 'Team', 'Team_Games']].copy()
    teams['_team_key'] = teams.Team.map(team_key)
    if teams.duplicated(['Season', '_team_key']).any():
        raise ValueError('Duplicate roster depth team-season keys')
    if not (np.isfinite(teams.Team_Games) & teams.Team_Games.gt(0)).all():
        raise ValueError('Roster depth requires positive finite team games')
    ranked = rank_team_players(source, teams)
    diagnostics = source.merge(teams, on=['Season', '_team_key'],
                               how='inner', suffixes=('_Source', ''), validate='many_to_one')
    diagnostics = diagnostics.merge(
        ranked[identity + ['_player_rank', '_rating', '_rating_name', '_qualifies']],
        on=identity, how='left', validate='one_to_one')
    diagnostics['Minutes_Per_Team_Game'] = diagnostics.MP / diagnostics.Team_Games
    diagnostics['BPM_Available'] = np.isfinite(diagnostics.BPM)
    diagnostics['Qualifies_10_Minutes'] = diagnostics.Minutes_Per_Team_Game.ge(10)
    diagnostics['Rotation_12_Minutes'] = diagnostics.Minutes_Per_Team_Game.ge(12)
    diagnostics['Supporting_Cast_Included'] = (diagnostics._player_rank.between(3, 8) &
                                               diagnostics.Qualifies_10_Minutes & diagnostics.BPM_Available)
    diagnostics['Selection_Fallback'] = diagnostics._player_rank.notna() & ~diagnostics.Qualifies_10_Minutes
    groups = {(int(y), t): g for (y, t), g in diagnostics.groupby(['Season', '_team_key'])}
    rows = []
    for team in teams.itertuples(index=False):
        key = (team.Season, team_key(team.Team))
        group = groups.get(key, diagnostics.iloc[:0])
        ordered = group.sort_values('_player_rank').loc[lambda g: g._player_rank.notna()]
        total_minutes = group.MP.sum()
        bpm_minutes = group.loc[group.BPM_Available, 'MP'].sum()
        row = dict(Season=team.Season, Team=team.Team, Team_Games=team.Team_Games,
                   Selection_Metric='BPM' if team.Season >= 1974 else 'PER',
                   BPM_Available_In_Era=team.Season >= 1974,
                   Team_Player_Minutes=total_minutes, Players_With_Minutes=len(group),
                   Qualifying_Players=int(group.Qualifies_10_Minutes.sum()),
                   Missing_BPM_Players=int((~group.BPM_Available).sum()),
                   BPM_Minutes_Coverage=bpm_minutes / total_minutes if total_minutes else np.nan)
        for rank, prefix in ((3, 'Third'), (4, 'Fourth')):
            selected = ordered.loc[ordered._player_rank.eq(rank)]
            player = selected.iloc[0] if len(selected) else None
            row[f'{prefix}_Player_Name'] = player.Player if player is not None else None
            row[f'{prefix}_Player_BPM'] = player.BPM if player is not None else np.nan
            row[f'{prefix}_Player_Minutes'] = player.MP if player is not None else np.nan
            row[f'{prefix}_Player_Selection_Fallback'] = bool(player.Selection_Fallback) if player is not None else None
        support = group.loc[group.Supporting_Cast_Included]
        support_minutes = support.MP.sum()
        row['Supporting_Cast_Players'] = len(support)
        row['Supporting_Cast_Minutes'] = support_minutes
        row['Supporting_Cast_BPM'] = ((support.BPM * support.MP).sum() / support_minutes
                                      if support_minutes > 0 else np.nan)
        row['Rotation_Depth'] = int(group.Rotation_12_Minutes.sum()) if len(group) else np.nan
        leaders = ordered.loc[ordered._player_rank.le(2)]
        row['Top_2_Minutes_Share'] = (leaders.MP.sum() / total_minutes
                                      if len(leaders) == 2 and total_minutes > 0 else np.nan)
        row['Best_Player_Name_For_Share'] = leaders.Player.iloc[0] if len(leaders) else None
        row['Second_Player_Name_For_Share'] = leaders.Player.iloc[1] if len(leaders) > 1 else None
        rows.append(row)
    audit = pd.DataFrame(rows)
    result = metrics.drop(columns=list(ROSTER_DEPTH_FEATURES), errors='ignore').merge(
        audit[['Season', 'Team', *ROSTER_DEPTH_FEATURES]], on=['Season', 'Team'],
        how='left', validate='one_to_one')
    diagnostics = diagnostics.rename(columns={'_player_rank': 'Player_Rank', '_rating': 'Selection_Rating',
                                              '_rating_name': 'Selection_Metric'})
    columns = ['Season', 'Team', 'Player', 'Age', 'MP', 'Team_Games', 'Minutes_Per_Team_Game',
               'BPM', 'BPM_Available', 'Player_Rank', 'Selection_Metric', 'Selection_Rating',
               'Qualifies_10_Minutes', 'Rotation_12_Minutes', 'Selection_Fallback', 'Supporting_Cast_Included']
    return result, audit, diagnostics[columns].sort_values(['Season', 'Team', 'Player_Rank', 'Player'])


def depth_candidates(frame, train, minimum_coverage):
    result = []
    for name in ROSTER_DEPTH_FEATURES:
        if name not in frame:
            continue
        values = pd.to_numeric(frame.loc[train, name], errors='coerce').replace([np.inf, -np.inf], np.nan)
        valid = values.dropna()
        if values.notna().mean() >= minimum_coverage and len(valid) > 10 and valid.std() > 1e-10:
            result.append(name)
    return result


def feature_selection_audit(frame, train, selected, eligible, include_depth, additional_features=()):
    """Record selection separately from mere presence in a source file."""
    rows = []
    for name in dict.fromkeys([*eligible, *selected, *ROSTER_DEPTH_FEATURES, *additional_features]):
        values = (pd.to_numeric(frame.loc[train, name], errors='coerce') if name in frame
                  else pd.Series(np.nan, index=frame.index[train]))
        chosen = name in selected
        reason = ('selected' if chosen else 'roster depth disabled' if name in ROSTER_DEPTH_FEATURES and not include_depth
                  else 'not present in inputs' if name not in frame else 'feature cap or no within-season variation'
                  if name in eligible else 'insufficient training coverage or variation')
        rows.append(dict(Feature=name, Roster_Depth=name in ROSTER_DEPTH_FEATURES, Selected=chosen,
                         Training_Nonmissing=int(values.notna().sum()),
                         Training_Coverage=float(values.notna().mean()), Reason=reason))
    return pd.DataFrame(rows)


def depth_configuration(enabled, selected, base_cap):
    return dict(enabled=enabled, requested_features=list(ROSTER_DEPTH_FEATURES) if enabled else [],
                selected_features=[c for c in selected if c in ROSTER_DEPTH_FEATURES],
                base_feature_cap=base_cap,
                effective_feature_cap=base_cap + (len(ROSTER_DEPTH_FEATURES) if enabled else 0),
                cap_policy='Preserve the existing base selection, then append up to five eligible depth features.',
                definition=ROSTER_DEPTH_RULE)
