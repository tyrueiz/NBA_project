"""Prior-history proxies for a new contender and inexperienced leading players."""
from pathlib import Path

import pandas as pd

from nba_predictor.nba_advanced_stats import player_key, team_key


FIRST_TIME_FEATURES = ('Prior_Best_Player_Playoff_Seasons_Proxy',
                       'Prior_Second_Player_Playoff_Seasons_Proxy',
                       'Best_Player_First_Playoffs_Proxy', 'Both_Leaders_First_Playoffs_Proxy',
                       'Prior_Team_Playoff_Appearances_Last3', 'New_Contender_Proxy')
FIRST_TIME_RULE = ('Only prior seasons. Player experience counts regular-season roster '
                   'membership on teams that made the playoffs, once per year, not actual '
                   'playoff games. New contender means no team playoff appearance in the '
                   'prior three seasons and a leading player with no observed prior playoff '
                   'roster experience. No current-season playoff result or hand-set penalty is used.')


def add_first_time_features(metrics: pd.DataFrame, seasons: pd.DataFrame, stats_path: Path):
    required = {'Season', 'Team', 'Made_Playoffs'}
    if not required.issubset(seasons):
        raise ValueError(f'First-time contender features require {sorted(required)}')
    outcomes = seasons[['Season', 'Team', 'Made_Playoffs']].copy()
    outcomes['_team'] = outcomes.Team.map(team_key)
    flags = pd.to_numeric(outcomes.Made_Playoffs, errors='coerce')
    textual = outcomes.Made_Playoffs.astype('string').str.lower().map({'true': 1, 'false': 0})
    outcomes['_made'] = flags.where(flags.isin([0, 1]), textual)
    if outcomes.duplicated(['Season', '_team']).any():
        raise ValueError('Ambiguous team histories after code normalization')
    roster = pd.read_csv(stats_path, low_memory=False).drop_duplicates()
    for column in ('Season', 'Age', 'MP'):
        roster[column] = pd.to_numeric(roster[column], errors='coerce')
    roster['_team'] = roster.Team.map(team_key)
    roster['_name'] = roster.Player.map(player_key)
    roster['_birth'] = roster.Season - roster.Age
    roster = roster.loc[roster._team.ne('TOT') & roster.MP.gt(0)].copy()
    keys = ['Season', '_name', '_birth']
    selected_identities = []
    for prefix in ('Best', 'Second'):
        if f'{prefix}_Player_Name' not in metrics:
            raise ValueError('First-time contender features require both leading players; use --no-first-time to exclude them')
        selected_identities.append(pd.DataFrame({
            '_name': metrics[f'{prefix}_Player_Name'].map(player_key),
            '_birth': metrics.Season - metrics[f'{prefix}_Player_Age']}))
    wanted = pd.concat(selected_identities).drop_duplicates()
    # Only history relevant to the selected leaders is required. Defunct teams
    # elsewhere in the archive must not invalidate unrelated modern players.
    roster = roster.merge(wanted, on=['_name', '_birth'], how='inner', validate='many_to_one')
    if roster[keys].isna().any().any():
        raise ValueError('Missing player identity for experience history')
    membership = roster.merge(outcomes[['Season', '_team', '_made']], on=['Season', '_team'],
                              how='left', validate='many_to_one')
    # A player on two playoff teams in one year receives one season of credit.
    history = membership.groupby(keys)['_made'].max().reset_index().sort_values('Season')
    # Do not silently treat absent historical results as confirmed inexperience.
    unknown = history._made.isna() & history.Season.lt(metrics.Season.max())
    if unknown.any():
        raise ValueError('Missing prior team playoff results for player history')
    history['_made'] = history._made.fillna(0)
    history['_prior'] = history.groupby(['_name', '_birth'])._made.cumsum() - history._made
    result = metrics.copy()
    result['_team'] = result.Team.map(team_key)
    lookup = outcomes.set_index(['Season', '_team'])._made
    counts = []
    for year, team in zip(result.Season, result['_team']):
        # Teams with no record in an earlier season are treated as newly formed
        # under this code; common same-franchise code aliases are normalized.
        previous = [lookup.get((year - lag, team), 0) for lag in (1, 2, 3)]
        if any(pd.isna(value) for value in previous):
            raise ValueError(f'Missing prior team playoff result for {team} in {year}')
        counts.append(sum(previous))
    result['Prior_Team_Playoff_Appearances_Last3'] = counts
    for prefix, feature in [('Best', FIRST_TIME_FEATURES[0]), ('Second', FIRST_TIME_FEATURES[1])]:
        if f'{prefix}_Player_Name' not in result:
            raise ValueError('First-time contender features require both leading players; use --no-first-time to exclude them')
        selected = result[['Season', f'{prefix}_Player_Name', f'{prefix}_Player_Age']].copy()
        selected['_name'] = selected[f'{prefix}_Player_Name'].map(player_key)
        selected['_birth'] = selected.Season - selected[f'{prefix}_Player_Age']
        selected = selected.merge(history[keys + ['_prior']], on=keys, how='left', validate='many_to_one')
        if selected._prior.isna().any():
            raise ValueError(f'Cannot match {prefix.lower()} player to experience history')
        result[feature] = selected._prior.to_numpy()
    result['Best_Player_First_Playoffs_Proxy'] = result[FIRST_TIME_FEATURES[0]].eq(0).astype(int)
    result['Both_Leaders_First_Playoffs_Proxy'] = (result[FIRST_TIME_FEATURES[0]].eq(0) &
                                                 result[FIRST_TIME_FEATURES[1]].eq(0)).astype(int)
    result['New_Contender_Proxy'] = (result.Prior_Team_Playoff_Appearances_Last3.eq(0) &
                                    result.Best_Player_First_Playoffs_Proxy.eq(1)).astype(int)
    audit = result[['Season', 'Team', 'Best_Player_Name', 'Second_Player_Name', *FIRST_TIME_FEATURES]].copy()
    return result.drop(columns='_team'), audit
