#!/usr/bin/env python3
"""Cache actual regular-season team ratings from the public historical dataset."""
import argparse
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.parse import quote
from zipfile import ZipFile, is_zipfile

import pandas as pd
import requests

from nba_advanced_stats import team_key

SOURCE = 'https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats'
URL = 'https://www.kaggle.com/api/v1/datasets/download/sumitrodatta/nba-aba-baa-stats/' + quote('Team Summaries.csv')
TEAM_RATING_FEATURES = ('Team_Offensive_Rating', 'Team_Defensive_Rating', 'Team_Net_Rating')


def normalize_ratings(raw):
    raw = raw.copy()
    if 'isPlayoff' in raw and not raw.isPlayoff.astype(str).str.lower().isin(['false', '0']).all():
        raise ValueError('Team summaries must contain regular-season data only')
    aliases = {'season': 'Season', 'lg': 'League', 'league': 'League',
               'abbreviation': 'Team', 'tm': 'Team', 'ortg': TEAM_RATING_FEATURES[0],
               'drtg': TEAM_RATING_FEATURES[1], 'nrtg': TEAM_RATING_FEATURES[2],
               'o_rtg': TEAM_RATING_FEATURES[0], 'd_rtg': TEAM_RATING_FEATURES[1],
               'n_rtg': TEAM_RATING_FEATURES[2]}
    raw = raw.rename(columns=aliases)
    required = {'Season', 'Team', 'League', *TEAM_RATING_FEATURES[:2]}
    if not required.issubset(raw):
        raise ValueError(f'Team summaries require {sorted(required)}; found {list(raw.columns)}')
    raw = raw.loc[raw.League.eq('NBA')].copy()
    raw['Season'] = pd.to_numeric(raw.Season, errors='coerce')
    raw = raw.dropna(subset=['Season', 'Team'])
    raw['Season'] = raw.Season.astype(int)
    raw['Team'] = raw.Team.map(team_key)
    for column in TEAM_RATING_FEATURES[:2]:
        raw[column] = pd.to_numeric(raw[column], errors='coerce')
    # Define net rating consistently from the imported team efficiencies.
    raw[TEAM_RATING_FEATURES[2]] = raw[TEAM_RATING_FEATURES[0]] - raw[TEAM_RATING_FEATURES[1]]
    result = raw[['Season', 'Team', *TEAM_RATING_FEATURES]].copy()
    if result.duplicated(['Season', 'Team']).any():
        raise ValueError('Duplicate team-season ratings after alias normalization')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=int, default=1970)
    parser.add_argument('--end', type=int, default=2026)
    parser.add_argument('--outdir', type=Path, default=Path('nba_updated'))
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args(argv)
    if args.start > args.end:
        parser.error('Invalid season range')
    args.outdir.mkdir(parents=True, exist_ok=True)
    cache = args.outdir/'team_summaries_source.csv'
    if cache.exists() and not args.refresh:
        raw = pd.read_csv(cache, low_memory=False)
    else:
        response = requests.get(URL, timeout=90)
        if response.status_code in (401, 403, 429):
            raise PermissionError(f'Team summaries HTTP {response.status_code}; download stopped')
        response.raise_for_status()
        content = BytesIO(response.content)
        if is_zipfile(content):
            with ZipFile(content) as archive:
                names = [name for name in archive.namelist() if Path(name).name == 'Team Summaries.csv']
                if len(names) != 1:
                    raise ValueError('No unique Team Summaries.csv in source archive')
                with archive.open(names[0]) as handle:
                    raw = pd.read_csv(handle, low_memory=False)
        else:
            content.seek(0)
            raw = pd.read_csv(content, low_memory=False)
        raw.to_csv(cache, index=False)
    result = normalize_ratings(raw)
    result = result.loc[result.Season.between(args.start, args.end)].copy()
    missing = set(range(args.start, args.end + 1)) - set(result.Season)
    if missing:
        raise ValueError(f'Missing team summary seasons: {sorted(missing)}')
    result['Source'] = SOURCE
    result['Data_Collected_UTC'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    path = args.outdir/f'NBA_team_ratings_{args.start}_{args.end}.csv'
    result.to_csv(path, index=False)
    print(f'Saved {len(result)} team-seasons to {path}')
    print(result.groupby('Season')[list(TEAM_RATING_FEATURES)].count().head(10).to_string())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
