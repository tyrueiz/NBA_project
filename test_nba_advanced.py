"""Checks for advanced-stat identity joins, postseason exclusion, and pagination."""
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import pandas as pd

from nba_advanced_stats import join_advanced, normalize_advanced, player_key
from nba_download_advanced import fetch_season, main as download_main
from nba_mvp_common import make_features


def advanced_row(**overrides):
    return dict(season=2024, playerName='Nikola Jokić', playerId='jokicni01',
                age=28, team='DEN', games=79, per=31, winShares=17,
                box=13.2, vorp=10.6, isPlayoff=False) | overrides


class AdvancedTests(unittest.TestCase):
    def test_mojibake_and_unicode_names_match(self):
        self.assertEqual(player_key('Nikola Jokić'), player_key('Nikola JokiÄ\x87'))

    def test_postseason_rows_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'regular-season'):
            normalize_advanced(pd.DataFrame([advanced_row(isPlayoff=True)]))

    def test_aggregate_and_namesake_joins_do_not_duplicate_players(self):
        advanced = pd.DataFrame([
            advanced_row(team='TOT'), advanced_row(team='AAA', per=10),
            advanced_row(age=22, team='TOT', per=20)])
        players = pd.DataFrame([
            dict(Season=2024, Player='Nikola JokiÄ\x87', Age=28, Team='2TM'),
            dict(Season=2024, Player='Nikola Jokić', Age=22, Team='3TM'),
            dict(Season=2025, Player='Nikola Jokić', Age=29, Team='DEN')])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'advanced.csv'
            advanced.to_csv(path, index=False)
            result, coverage = join_advanced(players, path)
        self.assertEqual(len(result), 3)
        self.assertEqual(result.PER.iloc[:2].tolist(), [31, 20])
        self.assertTrue(pd.isna(result.PER.iloc[2]))
        self.assertEqual(coverage.Match_Coverage.tolist(), [1, 0])

    def test_conflicting_source_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'advanced.csv'
            pd.DataFrame([advanced_row(), advanced_row(per=20)]).to_csv(path, index=False)
            players = pd.DataFrame([dict(Season=2024, Player='Nikola Jokić', Age=28, Team='DEN')])
            with self.assertRaisesRegex(ValueError, 'Ambiguous'):
                join_advanced(players, path)

    def test_advanced_metrics_are_features_without_awards(self):
        players = pd.DataFrame([
            dict(Season=2024, Age=28, G=79, PTS=2000, FGA=1200, FTA=300,
                 PER=31, BPM=13.2, VORP=10.6, WS=17, Awards='MVP-1'),
            dict(Season=2024, Age=25, G=75, PTS=1800, FGA=1100, FTA=250,
                 PER=20, BPM=5, VORP=4, WS=10, Awards='')])
        features = make_features(players)
        self.assertTrue({'PER', 'BPM', 'VORP', 'WS'}.issubset(features))
        self.assertNotIn('Awards', features)
        self.assertGreater(features.PER.iloc[0], features.PER.iloc[1])

    def test_api_pages_are_combined_and_regular_season_requested(self):
        class Response:
            status_code = 200

            def __init__(self, page):
                self.page = page

            def raise_for_status(self):
                pass

            def json(self):
                return dict(data=[advanced_row(playerId=str(self.page))],
                            pagination=dict(pages=2, total=2))

        class Session:
            def get(self, url, params, timeout):
                if params['isPlayoff'] != 'false':
                    raise AssertionError('Postseason query')
                return Response(params['page'])

        self.assertEqual(len(fetch_season(Session(), 2024, 0)), 2)

    def test_older_years_are_combined_without_inventing_unavailable_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for year in (1970, 1971):
                frame = normalize_advanced(pd.DataFrame([
                    advanced_row(season=year, box=None, vorp=None)]))
                frame.to_csv(root / f'advanced_{year}.csv', index=False)
            args = ['nba_download_advanced.py', '--start', '1970', '--end', '1971',
                    '--outdir', directory]
            with patch('sys.argv', args), patch('requests.Session.get', side_effect=AssertionError('Unexpected network call')):
                self.assertEqual(download_main(), 0)
            result = pd.read_csv(root / 'NBA_advanced_1970_1971.csv')
            self.assertEqual(result.Season.tolist(), [1970, 1971])
            self.assertTrue(result.BPM.isna().all())
            self.assertTrue(result.VORP.isna().all())
            coverage = pd.read_csv(root / 'advanced_metric_coverage_1970_1971.csv')
            self.assertTrue(coverage.PER_Coverage.eq(1).all())
            self.assertTrue(coverage.BPM_Coverage.eq(0).all())


if __name__ == '__main__':
    unittest.main()
