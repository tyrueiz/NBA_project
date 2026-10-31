"""Advanced-stat schema and strict joins shared by downloader and MVP models."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd


API_URL = 'https://api.server.nbaapi.com/api/playeradvancedstats'
API_COLUMNS = {
    'season': 'Season', 'playerName': 'Player', 'playerId': 'Player_ID',
    'age': 'Age', 'team': 'Team', 'games': 'G', 'minutesPlayed': 'MP',
    'per': 'PER', 'tsPercent': 'TS%', 'threePAR': '3PAr', 'ftr': 'FTr',
    'offensiveRBPercent': 'ORB%', 'defensiveRBPercent': 'DRB%',
    'totalRBPercent': 'TRB%', 'assistPercent': 'AST%', 'stealPercent': 'STL%',
    'blockPercent': 'BLK%', 'turnoverPercent': 'TOV%', 'usagePercent': 'USG%',
    'offensiveWS': 'OWS', 'defensiveWS': 'DWS', 'winShares': 'WS',
    'winSharesPer': 'WS/48', 'offensiveBox': 'OBPM', 'defensiveBox': 'DBPM',
    'box': 'BPM', 'vorp': 'VORP',
}
ADVANCED_FEATURES = tuple(value for key, value in API_COLUMNS.items()
                          if key not in ('season', 'playerName', 'playerId', 'age',
                                         'team', 'games', 'minutesPlayed'))

# First seasons of consistent NBA coverage in the downloaded source. Older
# isolated values in the totals archive must not stand in for NBA-wide data.
NBA_STAT_START_SEASON = {'BPM': 1974, 'OBPM': 1974, 'DBPM': 1974, 'VORP': 1974,
                         'STL': 1974, 'BLK': 1974, 'TOV': 1978,
                         'USG%': 1978, 'TOV%': 1978}


def player_key(value: str) -> str:
    value = str(value).strip().rstrip('*')
    # The existing scraper CSV contains UTF-8 names decoded as Latin-1.
    if any(character in value for character in ('Ã', 'Ä', 'Å', 'Â', 'È', 'É', 'Ð', 'Ñ')):
        for encoding in ('latin1', 'cp1252'):
            try:
                value = value.encode(encoding).decode('utf-8')
                break
            except (UnicodeError, LookupError):
                pass
    value = unicodedata.normalize('NFKD', value).casefold()
    return ''.join(character for character in value
                   if character.isalnum() and not unicodedata.combining(character))


def team_key(value: str) -> str:
    value = str(value).strip().upper()
    if re.fullmatch(r'TOT|[2-9]TM', value):
        return 'TOT'
    return {'BKN': 'BRK', 'CHA': 'CHO', 'PHX': 'PHO', 'NOK': 'NOH', 'GS': 'GSW'}.get(value, value)


def normalize_advanced(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.rename(columns=API_COLUMNS).copy()
    required = {'Season', 'Player', 'Age', 'Team', 'PER', 'WS', 'BPM', 'VORP'}
    if not required.issubset(frame):
        raise ValueError(f'Advanced stats require columns: {sorted(required)}')
    if 'isPlayoff' in frame:
        flags = frame.isPlayoff.astype(str).str.lower()
        if not flags.isin(['false', '0']).all():
            raise ValueError('Advanced stats must contain only verified regular-season rows')
    for column in ('Season', 'Age', 'G', 'MP', *ADVANCED_FEATURES):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors='coerce')
    if frame[['Season', 'Player', 'Age', 'Team']].isna().any().any():
        raise ValueError('Advanced stats contain missing player identity fields')
    frame['Season'] = frame.Season.astype(int)
    frame['Player'] = frame.Player.astype(str).str.strip().str.rstrip('*')
    frame['Team'] = frame.Team.astype(str).str.strip().str.upper()
    return frame


def join_advanced(players: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    advanced = normalize_advanced(pd.read_csv(path, low_memory=False)).drop_duplicates()
    players = players.copy()
    for frame in (players, advanced):
        frame['_advanced_name'] = frame.Player.map(player_key)
        frame['_advanced_team'] = frame.Team.map(team_key)
    keys = ['Season', '_advanced_name', 'Age', '_advanced_team']
    if advanced.duplicated(keys).any():
        raise ValueError('Ambiguous advanced player-season-team rows; inspect source identities')
    columns = [column for column in ADVANCED_FEATURES if column in advanced]
    advanced['_advanced_matched'] = True
    # Allowlist excludes awards, ranks, identifiers, and postseason fields.
    result = players.merge(advanced[keys + columns + ['_advanced_matched']], on=keys,
                           how='left', validate='many_to_one', suffixes=('', '_Advanced'))
    # Use the imported metric where the totals CSV already has a derived stat.
    for column in columns:
        if f'{column}_Advanced' in result:
            result[column] = result.pop(f'{column}_Advanced')
    result['_advanced_matched'] = result._advanced_matched.eq(True)
    coverage = result.groupby('Season').agg(
        Players=('Player', 'size'), Matched=('_advanced_matched', 'sum'))
    coverage['Match_Coverage'] = coverage.Matched / coverage.Players
    for column in columns:
        coverage[f'{column}_Coverage'] = result.groupby('Season')[column].apply(lambda values: values.notna().mean())
    result = result.drop(columns=['_advanced_name', '_advanced_team', '_advanced_matched'])
    return result, coverage.reset_index()
