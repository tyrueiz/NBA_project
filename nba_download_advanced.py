#!/usr/bin/env python3
"""Download regular-season advanced stats from a public historical dataset.

Dataset: https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats
Newer seasons can fall back to https://github.com/nprasad2077/nbaStats.
Neither provider is an official NBA endpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from io import BytesIO
from zipfile import ZipFile, is_zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from nba_advanced_stats import ADVANCED_FEATURES, API_URL, normalize_advanced

HISTORICAL_URL = 'https://www.kaggle.com/api/v1/datasets/download/sumitrodatta/nba-aba-baa-stats/Advanced.csv'
HISTORICAL_COLUMNS = {
    'season': 'Season', 'player': 'Player', 'player_id': 'Player_ID', 'age': 'Age',
    'tm': 'Team', 'team': 'Team', 'g': 'G', 'mp': 'MP', 'per': 'PER',
    'ts_percent': 'TS%', 'x3p_ar': '3PAr', 'f_tr': 'FTr',
    'orb_percent': 'ORB%', 'drb_percent': 'DRB%', 'trb_percent': 'TRB%',
    'ast_percent': 'AST%', 'stl_percent': 'STL%', 'blk_percent': 'BLK%',
    'tov_percent': 'TOV%', 'usg_percent': 'USG%', 'ows': 'OWS', 'dws': 'DWS',
    'ws': 'WS', 'ws_48': 'WS/48', 'obpm': 'OBPM', 'dbpm': 'DBPM',
    'bpm': 'BPM', 'vorp': 'VORP',
}


def load_historical(session, outdir: Path) -> pd.DataFrame:
    """Public regular-season historical dataset; kept as a separate source cache."""
    path = outdir / 'advanced_historical_source.csv'
    if path.exists():
        return normalize_advanced(pd.read_csv(path, low_memory=False))
    response = session.get(HISTORICAL_URL, timeout=90)
    if response.status_code in (401, 403, 429):
        raise PermissionError(f'Historical dataset HTTP {response.status_code}; no bypass attempted')
    response.raise_for_status()
    content = BytesIO(response.content)
    if is_zipfile(content):
        with ZipFile(content) as archive:
            names = [name for name in archive.namelist() if Path(name).name == 'Advanced.csv']
            if len(names) != 1:
                raise ValueError('Historical archive has no unique Advanced.csv')
            with archive.open(names[0]) as source:
                frame = pd.read_csv(source, low_memory=False)
    else:
        content.seek(0)
        frame = pd.read_csv(content, low_memory=False)
    if 'lg' not in frame:
        raise ValueError('Historical source lacks league identifier')
    frame = frame.loc[frame.lg.eq('NBA')].rename(columns=HISTORICAL_COLUMNS)
    frame = normalize_advanced(frame)
    frame['Source'] = 'https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats'
    frame['Data_Collected_UTC'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    frame.to_csv(path, index=False)
    return frame


def fetch_season(session, year: int, delay: float) -> pd.DataFrame:
    rows = []
    page = 1
    while True:
        response = session.get(API_URL, params={'season': year, 'isPlayoff': 'false',
                                               'pageSize': 1000, 'page': page}, timeout=45)
        if response.status_code in (403, 429):
            raise PermissionError(f'HTTP {response.status_code}; stopping downloads without retries')
        response.raise_for_status()
        payload = response.json()
        batch = payload['data']
        if not batch:
            if page == 1:
                raise ValueError(f'No advanced stats available for {year}')
            raise ValueError(f'Unexpected empty page {page} for {year}')
        rows.extend(batch)
        pages = int(payload['pagination']['pages'])
        if page >= pages:
            expected = int(payload['pagination']['total'])
            if len(rows) != expected:
                raise ValueError(f'{year}: fetched {len(rows)} rows, expected {expected}')
            break
        page += 1
        time.sleep(delay)
    frame = normalize_advanced(pd.DataFrame(rows))
    if not frame.Season.eq(year).all():
        raise ValueError(f'API returned seasons other than {year}')
    frame['Source'] = API_URL
    frame['Data_Collected_UTC'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=int, default=1970)
    parser.add_argument('--end', type=int, default=2026)
    parser.add_argument('--outdir', type=Path, default=Path('nba_updated'))
    parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--delay', type=float, default=1, help='Seconds between API requests')
    args = parser.parse_args()
    if args.start > args.end or args.delay < 0:
        parser.error('Invalid season range or delay')
    args.outdir.mkdir(parents=True, exist_ok=True)
    collected, report = [], []
    historical = None
    with requests.Session() as session:
        session.headers['User-Agent'] = 'NBA-project-advanced-stats/1.0'
        for year in range(args.start, args.end + 1):
            cache = args.outdir / f'advanced_{year}.csv'
            try:
                if cache.exists() and not args.refresh:
                    frame = normalize_advanced(pd.read_csv(cache, low_memory=False))
                    if frame.empty or not frame.Season.eq(year).all():
                        raise ValueError(f'Invalid advanced cache: {cache}')
                    status = 'cached'
                else:
                    if historical is None:
                        print('Loading public historical advanced-stat dataset ...', flush=True)
                        historical = load_historical(session, args.outdir)
                    frame = historical.loc[historical.Season.eq(year)].copy()
                    if frame.empty:
                        frame = fetch_season(session, year, args.delay)
                        time.sleep(args.delay)
                    frame.to_csv(cache, index=False)
                    status = 'downloaded'
                collected.append(frame)
                report.append({'Season': year, 'Status': status, 'Rows': len(frame)})
                print(f'{year}: {status}, {len(frame)} rows', flush=True)
            except (requests.RequestException, ValueError, KeyError, PermissionError) as exc:
                report.append({'Season': year, 'Status': 'failed', 'Error': str(exc)})
                print(f'{year}: {exc}', file=sys.stderr, flush=True)
                # Retain completed caches; resume with the same command.
                break
    complete = len(collected) == args.end - args.start + 1
    if collected:
        filename = f'NBA_advanced_{args.start}_{args.end}.csv' if complete else 'advanced_partial.csv'
        combined = pd.concat(collected, ignore_index=True)
        combined.to_csv(args.outdir / filename, index=False)
        coverage = combined.groupby('Season').size().to_frame('Rows')
        for column in ADVANCED_FEATURES:
            if column in combined:
                coverage[f'{column}_Coverage'] = combined.groupby('Season')[column].apply(
                    lambda values: values.notna().mean())
        coverage.to_csv(args.outdir / f'advanced_metric_coverage_{args.start}_{args.end}.csv')
    (args.outdir / 'advanced_download_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return 0 if complete else 1


if __name__ == '__main__':
    sys.exit(main())
