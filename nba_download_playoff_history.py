#!/usr/bin/env python3
"""Build actual postseason participation caches from public game box scores.

Game/player data: eoinamoore's historical NBA game archive on Kaggle.
Coaching records: Basketball Reference, with a cached 1960–2019 historical
export from spoonertaylor/NBA_Coaches. Later/earlier years use season pages.
No predictor performs network requests. Completed downloads are reusable.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from urllib.parse import quote

from bs4 import BeautifulSoup, Comment
import numpy as np
import pandas as pd
import requests

from nba_advanced_stats import player_key, team_key
from nba_champion_playoff_experience import match_roster_identities

ARCHIVE = 'https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores'
DOWNLOAD = 'https://www.kaggle.com/api/v1/datasets/download/eoinamoore/historical-nba-data-and-player-box-scores/'
COACH_SOURCE = 'https://raw.githubusercontent.com/spoonertaylor/NBA_Coaches/master/coaches_long.csv'


def season_from_game_id(values):
    code = values.astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)
    start = code.str[3:5].astype(int)
    return start + np.where(start.ge(46), 1901, 2001)


def cached_download(session, url, destination):
    if destination.exists():
        return destination
    partial = destination.with_suffix(destination.suffix + '.partial')
    with session.get(url, timeout=(15, 120), stream=True) as response:
        response.raise_for_status()
        with partial.open('wb') as stream:
            for block in response.iter_content(1024 * 1024):
                stream.write(block)
    partial.replace(destination)
    print('Downloaded', destination.name, flush=True)
    return destination


def archive_csv(path, **kwargs):
    with path.open('rb') as handle:
        zipped = handle.read(2) == b'PK'
    return pd.read_csv(path, compression='zip' if zipped else None, **kwargs)


def parse_coaches(html, year):
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table', id='NBA_coaches')
    if table is None:
        for comment in soup.find_all(string=lambda x: isinstance(x, Comment)):
            if 'NBA_coaches' in comment:
                table = BeautifulSoup(comment, 'html.parser').find('table', id='NBA_coaches')
                if table is not None:
                    break
    if table is None:
        raise ValueError(f'{year}: missing coaches table')
    rows, order = [], {}
    for row in table.select('tbody tr'):
        cells = {cell.get('data-stat'): cell for cell in row.find_all(['th', 'td'])}
        if 'coach' not in cells or 'team' not in cells or not cells['coach'].find('a'):
            continue
        team = team_key(cells['team'].get_text(strip=True))
        order[team] = order.get(team, 0) + 1
        def number(column):
            value = cells.get(column)
            text = value.get_text(strip=True) if value is not None else ''
            return int(text) if text else 0
        rows.append(dict(Season=year, Team=team, Coach=cells['coach'].get_text(strip=True),
                         Coach_Order=order[team], Regular_Games=number('cur_g'),
                         Playoff_Games=number('cur_g_p'),
                         Source=f'https://www.basketball-reference.com/leagues/NBA_{year}_coaches.html'))
    if not rows:
        raise ValueError(f'{year}: empty coaches table')
    return pd.DataFrame(rows)


def load_coaches(session, cache, start, end, delay):
    legacy_path = cached_download(session, COACH_SOURCE, cache/'coaches_long_source.csv')
    legacy = pd.read_csv(legacy_path).rename(columns={
        'season': 'Season', 'team_abbrv': 'Team', 'coach_name': 'Coach',
        'games_coached': 'Regular_Games', 'playoff_games': 'Playoff_Games', 'team_coach_number': 'Coach_Order'})
    # The source's 2020 postseason is unfinished; never use that incomplete year.
    legacy = legacy.loc[legacy.Season.between(max(start, 1960), min(end, 2019)),
                        ['Season', 'Team', 'Coach', 'Coach_Order', 'Regular_Games', 'Playoff_Games']].copy()
    legacy['Source'] = COACH_SOURCE
    frames = [legacy]
    for year in range(start, end + 1):
        if 1960 <= year <= 2019:
            continue
        path = cache/f'coaches_{year}.html'
        if year == 2026 and not path.exists() and (cache/'coaches_2026_sample.html').exists():
            path.write_bytes((cache/'coaches_2026_sample.html').read_bytes())
        was_cached = path.exists()
        cached_download(session, f'https://www.basketball-reference.com/leagues/NBA_{year}_coaches.html', path)
        frames.append(parse_coaches(path.read_text(), year))
        if not was_cached:
            time.sleep(delay)
    result = pd.concat(frames, ignore_index=True)
    result['Team'] = result.Team.map(team_key)
    result['Coach_Key'] = result.Coach.map(player_key)
    if result.duplicated(['Season', 'Team', 'Coach_Order']).any():
        raise ValueError('Ambiguous coaching chronology')
    return result


def map_team_codes(games, seasons):
    rows = []
    for side in ('home', 'away'):
        block = games[['Season', f'{side}teamId', f'{side}teamCity', f'{side}teamName']].copy()
        block.columns = ['Season', 'Team_ID', 'City', 'Name']
        block['Name_Key'] = (block.City.fillna('') + ' ' + block.Name.fillna('')).map(player_key)
        rows.append(block[['Season', 'Team_ID', 'Name_Key']])
    lookup = pd.concat(rows).drop_duplicates()
    # Official historical spellings differ for these names.
    aliases = {'laclippers': 'losangelesclippers',
               'ftwaynezollnerpistons': 'fortwaynepistons', 'ftwaynepistons': 'fortwaynepistons',
               'stlouishawks': 'stlouishawks', 'oklahomacityhornets': 'neworleansoklahomacityhornets',
               'neworleanshornets': 'neworleansoklahomacityhornets'}
    lookup['Name_Key'] = lookup.Name_Key.replace(aliases)
    lookup = lookup.drop_duplicates()
    teams = seasons[['Season', 'Team', 'team_name']].copy()
    teams['Name_Key'] = teams.team_name.map(player_key).replace(aliases)
    teams['Team'] = teams.Team.map(team_key)
    result = teams.merge(lookup, on=['Season', 'Name_Key'], how='left', validate='one_to_one')
    return result[['Season', 'Team', 'Team_ID']]


def build_history(games, player_games, coaches, team_codes, start, end):
    """A series credit requires a real played game in that particular series."""
    games = games.loc[games.gameType.eq('Playoffs') & games.Season.between(start, end)].copy()
    # Some old NBA game ids switch series numbers during a series (e.g. the
    # 1962 Finals). Group by opponents and stage rather than trusting that id.
    smaller = games[['hometeamId', 'awayteamId']].min(axis=1).astype(str)
    larger = games[['hometeamId', 'awayteamId']].max(axis=1).astype(str)
    games['Round_Robin'] = (games.gameLabel.fillna('').str.contains('round robin', case=False) |
                            (games.Season.eq(1954) & games.gameLabel.fillna('').str.fullmatch(r'(East|West)\s*-\s*')))
    games['Series_ID'] = games.Season.astype(str) + '_' + smaller + '_' + larger + np.where(games.Round_Robin, '_round_robin', '_series')
    games['Conference_Finals'] = games.gameLabel.fillna('').str.contains(
        r'conf\.?\s*finals|conference finals', case=False, regex=True) & games.Season.ge(1971)
    games['Finals'] = games.gameLabel.fillna('').str.fullmatch(r'NBA Finals', case=False)
    series_rows, team_rows = [], []
    for (year, series_id), group in games.groupby(['Season', 'Series_ID']):
        wins = group.winner.value_counts()
        last = group.sort_values('gameDateTimeEst').iloc[-1]
        winner = int(last.winner)
        teams = set(group.hometeamId) | set(group.awayteamId)
        round_robin = bool(group.Round_Robin.any())
        if len(teams) != 2 or len(wins) == 0 or (not round_robin and wins.get(winner, 0) <= max(
                [wins.get(team, 0) for team in teams if team != winner])):
            raise ValueError(f'Incomplete or ambiguous series {year}/{series_id}')
        series_rows.append(dict(Season=year, Series_ID=series_id, Winner_ID=winner if not round_robin else -1,
                                Games=len(group), Conference_Finals=bool(group.Conference_Finals.any()),
                                Finals=bool(group.Finals.any())))
        for team in teams:
            team_rows.append(dict(Season=year, Series_ID=series_id, Team_ID=team,
                                  Games=len(group), Series_Wins=int(team == winner and not round_robin),
                                  Conference_Finals=int(group.Conference_Finals.any()), Finals=int(group.Finals.any())))
    series = pd.DataFrame(series_rows)
    per_team = pd.DataFrame(team_rows)
    annual_teams = per_team.groupby(['Season', 'Team_ID']).agg(
        Playoff_Games=('Games', 'sum'), Series_Wins=('Series_Wins', 'sum'),
        Conference_Finals=('Conference_Finals', 'max'), Finals=('Finals', 'max')).reset_index()
    known_codes = team_codes.dropna(subset=['Team_ID'])
    annual_teams = annual_teams.merge(known_codes, on=['Season', 'Team_ID'], how='left', validate='one_to_one')
    if annual_teams.Team.isna().any():
        raise ValueError('Unmapped playoff team codes: ' + str(annual_teams.loc[annual_teams.Team.isna()].to_dict('records')[:10]))
    players = player_games.loc[player_games.gameId.isin(games.gameId)].copy()
    if players.duplicated(['gameId', 'personId']).any():
        raise ValueError('Duplicate player-game rows')
    # Blank minutes are inactive/DNP records. Actual participation requires
    # reported minutes; zero-second appearances remain valid played games.
    players = players.loc[players.numMinutes.notna() & players.numMinutes.ge(0)].copy()
    if players.comment.fillna('').str.match(r'DNP|DND|NWT', case=False).any():
        raise ValueError('A DNP row has nonmissing played minutes; inspect source')
    players = players.merge(games[['gameId', 'Season', 'Series_ID', 'Conference_Finals', 'Finals', 'hometeamId', 'awayteamId']],
                            on='gameId', validate='many_to_one')
    # The archive omits team ids throughout 2022. Its explicit home flag and
    # game schedule provide an unambiguous join; never infer team from a name.
    if players.playerteamId.isna().any():
        missing = players.playerteamId.isna()
        if not players.loc[missing, 'home'].isin([0, 1]).all():
            raise ValueError('Missing player-team ids without a verified home/away flag')
        players.loc[missing, 'playerteamId'] = np.where(players.loc[missing, 'home'].eq(1),
                                                       players.loc[missing, 'hometeamId'], players.loc[missing, 'awayteamId'])
    players = players.merge(series[['Season', 'Series_ID', 'Winner_ID']], on=['Season', 'Series_ID'], validate='many_to_one')
    players['Won_Series'] = players.playerteamId.eq(players.Winner_ID)
    per_player_series = players.groupby(['Season', 'personId', 'playerteamId', 'Series_ID']).agg(
        G=('gameId', 'nunique'), MP=('numMinutes', 'sum'), Series_Wins=('Won_Series', 'max'),
        Conference_Finals=('Conference_Finals', 'max'), Finals=('Finals', 'max')).reset_index()
    annual_players = per_player_series.groupby(['Season', 'personId', 'playerteamId']).agg(
        G=('G', 'sum'), MP=('MP', 'sum'), Series_Wins=('Series_Wins', 'sum'),
        Conference_Finals=('Conference_Finals', 'max'), Finals=('Finals', 'max'),
        Minimum_Series_MP=('MP', 'min')).reset_index().rename(
            columns={'personId': 'Player_ID', 'playerteamId': 'Team_ID'})
    annual_players = annual_players.merge(known_codes, on=['Season', 'Team_ID'], how='left', validate='many_to_one')
    if annual_players.Team.isna().any():
        raise ValueError('Missing player playoff team mapping')
    # Independent coach game counts detect missing games in the archive.
    coach_games = coaches.groupby(['Season', 'Team']).Playoff_Games.sum().rename('Coach_Playoff_Games')
    coverage = annual_teams.merge(coach_games, on=['Season', 'Team'], how='left', validate='one_to_one')
    coverage['Player_Games_With_Box_Scores'] = coverage.apply(lambda row: players.loc[
        players.Season.eq(row.Season) & players.playerteamId.eq(row.Team_ID), 'gameId'].nunique(), axis=1)
    coverage['Eligible'] = coverage.Playoff_Games.eq(coverage.Coach_Playoff_Games) & coverage.Playoff_Games.eq(coverage.Player_Games_With_Box_Scores)
    return annual_players, annual_teams, series, coverage


def verified_identities(cache):
    path = cache/'verified_nba_player_ids.csv'
    if path.exists():
        ids = set(pd.read_csv(path).Player_ID)
    else:
        ids = set()
        for chunk in archive_csv(cache/'PlayerStatistics.csv.download', usecols=['personId', 'gameType'], chunksize=150000):
            ids.update(chunk.loc[chunk.gameType.isin(['Regular Season', 'Playoffs']), 'personId'].dropna().astype(int))
        pd.DataFrame({'Player_ID': sorted(ids)}).to_csv(path, index=False)
    identities = archive_csv(cache/'Players.csv.download')
    identities['Verified_NBA_Game_ID'] = identities.personId.isin(ids)
    return identities


def fetch_playoff_totals(cache, year):
    path = cache/f'playoff_totals_{year}.csv'
    if path.exists():
        return pd.read_csv(path)
    rows, page = [], 1
    with requests.Session() as session:
        while True:
            response = session.get('https://api.server.nbaapi.com/api/playertotals', params={
                'season': year, 'isPlayoff': 'true', 'pageSize': 50, 'page': page}, timeout=45)
            response.raise_for_status()
            payload = response.json()
            rows.extend(payload['data'])
            if page >= int(payload['pagination']['pages']):
                if len(rows) != int(payload['pagination']['total']):
                    raise ValueError(f'{year}: incomplete API pagination')
                break
            page += 1
            time.sleep(.3)
    result = pd.DataFrame(rows)
    if result.empty:
        result = pd.DataFrame(columns=['playerName', 'age', 'season', 'team', 'games', 'minutesPg', 'isPlayoff'])
        result.to_csv(path, index=False)
        print(f'{year}: no independent season-totals records; retain verified game logs', flush=True)
        return result
    if not result.season.eq(year).all() or not result.isPlayoff.eq(True).all():
        raise ValueError(f'{year}: invalid playoff totals response')
    result.to_csv(path, index=False)
    print(f'Checked playoff season totals {year}: {len(result)} players', flush=True)
    return result


def parse_season_totals(html, year):
    soup = BeautifulSoup(html, 'html.parser')
    for comment in list(soup.find_all(string=lambda x: isinstance(x, Comment))):
        if '<table' in comment:
            comment.replace_with(BeautifulSoup(comment, 'html.parser'))
    rows = []
    for row in soup.select('table tbody tr'):
        cells = {cell.get('data-stat'): cell for cell in row.find_all(['th', 'td'])}
        name = cells.get('player', cells.get('name_display'))
        if name is None or 'g' not in cells or 'mp' not in cells:
            continue
        team = cells.get('team_id', cells.get('team_name_abbr', cells.get('team_name')))
        age = cells.get('age')
        if team is None or age is None:
            continue
        rows.append(dict(playerName=name.get_text(strip=True).rstrip('*'),
                         playerId=Path(name.find('a')['href']).stem if name.find('a') else None,
                         age=pd.to_numeric(age.get_text(strip=True), errors='coerce'),
                         season=year, team=team.get_text(strip=True),
                         games=pd.to_numeric(cells['g'].get_text(strip=True), errors='coerce'),
                         minutesPg=pd.to_numeric(cells['mp'].get_text(strip=True), errors='coerce'),
                         isPlayoff=True))
    result = pd.DataFrame(rows).drop_duplicates()
    if result.empty:
        raise ValueError(f'{year}: missing playoff player totals table')
    return result


def fetch_legacy_totals(session, cache, year, delay):
    path = cache/f'playoff_totals_bref_{year}.csv'
    if path.exists() and 'playerId' in pd.read_csv(path, nrows=0).columns:
        return pd.read_csv(path)
    html = cache/f'playoff_totals_bref_{year}.html'
    new = not html.exists()
    cached_download(session, f'https://www.basketball-reference.com/playoffs/NBA_{year}_totals.html', html)
    frame = parse_season_totals(html.read_text(), year)
    frame.to_csv(path, index=False)
    print(f'Parsed historical playoff totals {year}', flush=True)
    if new:
        time.sleep(delay)
    return frame


def verify_player_totals(players, identities, totals):
    """Game logs determine actual series; season totals give recorded MP precision."""
    totals = totals.drop_duplicates().rename(columns={'playerName': 'Player', 'age': 'Age', 'season': 'Season',
                                     'team': 'Team', 'games': 'API_G', 'minutesPg': 'API_MP'})
    for column in ('API_G', 'API_MP', 'Age', 'Season'):
        totals[column] = pd.to_numeric(totals[column], errors='coerce')
    totals = match_roster_identities(totals, identities)
    totals['Team'] = totals.Team.map(team_key)
    known = totals.loc[totals.Player_ID.notna()]
    if known.duplicated(['Season', 'Player_ID', 'Team']).any():
        duplicates = known.loc[known.duplicated(['Season', 'Player_ID', 'Team'], False),
                               ['Season', 'Team', 'Player', 'Age', 'Player_ID', 'Identity_Match']]
        raise ValueError('Duplicate verified postseason totals: ' + str(duplicates.head(12).to_dict('records')))
    result = players.merge(known[['Season', 'Player_ID', 'Team', 'API_G', 'API_MP']],
                           on=['Season', 'Player_ID', 'Team'], how='left', validate='one_to_one')
    result['GP_Matches'] = result.G.eq(result.API_G)
    result['Game_Log_G'] = result.G
    # Two sources occasionally disagree by one zero-minute appearance. Use
    # the reported season G only when every credited series has positive
    # minutes, so series/round participation remains independently established.
    safe = result.API_G.notna() & ~result.GP_Matches & result.G.sub(result.API_G).abs().eq(1) & result.Minimum_Series_MP.gt(0)
    result.loc[safe, 'G'] = result.loc[safe, 'API_G']
    result['GP_Count_Corrected'] = safe
    result['GP_Matches'] = result.G.eq(result.API_G)
    result['Game_Log_MP'] = result.MP
    # Replace precision only when participation agrees independently. A
    # mismatch is an actionable source discrepancy, not an imputed rookie.
    valid = result.GP_Matches & result.API_MP.notna()
    result.loc[valid, 'MP'] = result.loc[valid, 'API_MP']
    return result, totals


def reconcile_series_discrepancies(players, totals, cache, delay):
    """Resolve uncertain zero-minute records against actual player-series rows."""
    result = players.copy()
    result['Series_Reconciled_From_Player_Page'] = False
    with requests.Session() as session:
        for index, row in result.loc[result.API_G.notna() & ~result.GP_Matches].iterrows():
            match = totals.loc[totals.Season.eq(row.Season) & totals.Player_ID.eq(row.Player_ID) & totals.Team.eq(row.Team)]
            if len(match) != 1 or 'playerId' not in match or pd.isna(match.playerId.iloc[0]):
                raise ValueError('Cannot identify discrepant player for series cross-check')
            player_id = match.playerId.iloc[0]
            path = cache/f'player_series_{player_id}.html'
            fresh = not path.exists()
            url = f'https://www.basketball-reference.com/players/{player_id[0]}/{player_id}.html'
            cached_download(session, url, path)
            soup = BeautifulSoup(path.read_text(), 'html.parser')
            for comment in list(soup.find_all(string=lambda x: isinstance(x, Comment))):
                if 'playoffs_series' in comment:
                    comment.replace_with(BeautifulSoup(comment, 'html.parser'))
            table = soup.find('table', id='playoffs_series')
            if table is None:
                raise ValueError(f'Missing player series table: {player_id}')
            series = []
            for tr in table.select('tbody tr'):
                cells = {c.get('data-stat'): c for c in tr.find_all(['th', 'td'])}
                year_cell = cells.get('year_id')
                team_cell = cells.get('team_name_abbr', cells.get('team_id'))
                if year_cell is None or team_cell is None:
                    continue
                link = year_cell.find('a')
                text = link.get('href', '') if link else year_cell.get_text()
                year_match = re.search(r'NBA_(\d{4})', text)
                if year_match:
                    year = int(year_match.group(1))
                else:
                    year_text = year_cell.get_text(strip=True)
                    if not re.match(r'\d{4}', year_text):
                        continue
                    year = int(year_text[:4]) + 1
                if year != row.Season or team_key(team_cell.get_text(strip=True)) != row.Team:
                    continue
                g_cell = cells.get('games', cells.get('g'))
                if g_cell is None:
                    continue
                g = pd.to_numeric(g_cell.get_text(strip=True), errors='coerce')
                if not g > 0:
                    continue
                series.append(dict(G=g, Round=cells['ps_round'].get_text(strip=True),
                                   Won=cells['series_result'].get_text(strip=True).startswith('W')))
            gp = sum(s['G'] for s in series)
            if gp not in (row.API_G, row.Game_Log_G):
                raise ValueError(f'Series games do not reconcile for {player_id}/{row.Season}: {gp} vs {row.API_G}/{row.Game_Log_G}')
            result.loc[index, 'G'] = gp
            result.loc[index, 'MP'] = row.API_MP
            result.loc[index, 'Series_Wins'] = sum(s['Won'] for s in series)
            result.loc[index, 'Conference_Finals'] = any(s['Round'] in ('ECF', 'WCF') for s in series) and row.Season >= 1971
            result.loc[index, 'Finals'] = any(s['Round'] in ('FIN', 'F', 'Finals') for s in series)
            result.loc[index, 'GP_Matches'] = gp == row.API_G
            result.loc[index, 'GP_Count_Corrected'] = True
            result.loc[index, 'Series_Reconciled_From_Player_Page'] = True
            print(f'Reconciled actual series {player_id}/{row.Season}', flush=True)
            if fresh:
                time.sleep(delay)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('nba_updated'))
    parser.add_argument('--start', type=int, default=1954)
    parser.add_argument('--end', type=int, default=2026)
    parser.add_argument('--delay', type=float, default=6, help='Seconds between Basketball Reference requests (minimum 6)')
    parser.add_argument('--no-verify-season-totals', action='store_true', help='Skip the independent playoff G/MP cross-check')
    args = parser.parse_args()
    if args.start < 1954 or args.start > args.end or args.delay < 6:
        parser.error('Use a valid range from 1954 onward and --delay >= 6; earlier archive seasons are incomplete')
    cache, output = args.input_dir/'playoff_experience_sources', args.input_dir/'playoff_history'
    cache.mkdir(exist_ok=True, parents=True)
    output.mkdir(exist_ok=True, parents=True)
    with requests.Session() as session:
        for name in ('Games.csv', 'Players.csv', 'PlayerStatistics.csv'):
            cached_download(session, DOWNLOAD + quote(name), cache/(name+'.download'))
        coaches = load_coaches(session, cache, args.start, args.end, args.delay)
    coaches.to_csv(output/'coaches.csv', index=False)
    games = archive_csv(cache/'Games.csv.download', low_memory=False)
    games['Season'] = season_from_game_id(games.gameId)
    partial = cache/'playoff_player_games.csv'
    if partial.exists() and 'home' in pd.read_csv(partial, nrows=0).columns:
        player_games = pd.read_csv(partial, low_memory=False)
    else:
        columns = ['personId', 'gameId', 'gameType', 'numMinutes', 'playerteamId', 'comment', 'home']
        chunks = [chunk.loc[chunk.gameType.eq('Playoffs')] for chunk in archive_csv(
            cache/'PlayerStatistics.csv.download', usecols=columns, chunksize=100000)]
        player_games = pd.concat(chunks, ignore_index=True)
        player_games.to_csv(partial, index=False)
    seasons = pd.read_csv(args.input_dir/'NBA_team_seasons_1947_2027.csv', low_memory=False)
    codes = map_team_codes(games, seasons)
    codes.to_csv(output/'team_identities.csv', index=False)
    players, teams, series, coverage = build_history(games, player_games, coaches, codes, args.start, args.end)
    coverage.to_csv(output/'source_coverage.csv', index=False)
    if not coverage.Eligible.all():
        raise ValueError('Incomplete playoff game archive; see source_coverage.csv')
    if set(range(args.start, args.end + 1)) - set(teams.Season):
        raise ValueError('Missing playoff seasons')
    identities = verified_identities(cache)
    if not args.no_verify_season_totals:
        with ThreadPoolExecutor(max_workers=2) as pool:
            totals = pd.concat(list(pool.map(lambda year: fetch_playoff_totals(cache, year),
                                            range(args.start, args.end + 1))), ignore_index=True)
        # The API has no pre-2001 playoff totals. Retrieve recorded season MP
        # rather than treating the archive's old missing/rounded minutes as 0.
        legacy = []
        with requests.Session() as session:
            for year in range(args.start, min(args.end, 2000) + 1):
                legacy.append(fetch_legacy_totals(session, cache, year, args.delay))
        if legacy:
            totals = pd.concat([totals, *legacy], ignore_index=True)
        players, identity_audit = verify_player_totals(players, identities, totals)
        players = reconcile_series_discrepancies(players, identity_audit, cache, args.delay)
        players.to_csv(output/'player_totals_crosscheck.csv', index=False)
        identity_audit.to_csv(output/'totals_identity_audit.csv', index=False)
        available_years = set(totals.season.dropna().astype(int))
        compared = players.API_G.notna()
        accepted = players.GP_Matches | players.Series_Reconciled_From_Player_Page
        if not accepted.loc[compared].all():
            raise ValueError('Player playoff games disagree; see player_totals_crosscheck.csv')
    players.to_csv(output/'players.csv', index=False)
    teams.to_csv(output/'teams.csv', index=False)
    series.to_csv(output/'series.csv', index=False)
    identities.to_csv(output/'player_identities.csv', index=False)
    (output/'metadata.json').write_text(json.dumps(dict(
        history_start=args.start, history_end=args.end, source=ARCHIVE,
        season_totals_checked=not args.no_verify_season_totals,
        minute_precision='Reported game-log minutes; independent season totals replace them where available and games agree.',
        built_utc=datetime.now(timezone.utc).isoformat(),
        player_participation='A reported-minute game in the actual series; inactive/DNP rows excluded.',
        conference_finals='Actual conference-final game participation from 1971; earlier division finals are not relabeled.',
        source_coverage_passed=True), indent=2))
    print(f'Saved {len(players)} player-season stints and {len(teams)} team playoff seasons to {output}', flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, requests.RequestException) as exc:
        raise SystemExit(f'ERROR: {exc}')
