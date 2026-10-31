"""Checks for team aggregation of advanced player statistics."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from nba_champion_advanced import TEAM_ADVANCED_FEATURES, add_team_advanced
from nba_champion_best_player import BEST_PLAYER_FEATURES, SECOND_PLAYER_FEATURES, select_best_players
from nba_advanced_stats import player_key, team_key
from nba_predict_champions_2010_2026 import available_features as logistic_features
from nba_predict_champions_xgboost_2010_2026 import available_features as xgb_features
from nba_predict_champions_2010_2026 import to_season_zscores as logistic_zscores
from nba_predict_champions_xgboost_2010_2026 import to_season_zscores as xgb_zscores


class TeamAdvancedTests(unittest.TestCase):
    def metrics(self):
        return pd.DataFrame([dict(Season=2024, Team=team, Team_Games=10, MP=240,
                                  PTS=110, FGA=90, FTA=20) for team in ('AAA', 'BBB')])

    def source(self):
        def row(player, team, minutes, per, ws):
            return dict(Season=2024, Player=player, Age=25, Team=team, MP=minutes, G=10,
                        PER=per, BPM=per/10, OBPM=per/20, DBPM=per/20, **{'WS/48': .1},
                        **{'TS%': .6, 'USG%': 20},
                        WS=ws, OWS=.6*ws, DWS=.4*ws, VORP=ws/2, Awards='MVP-1')
        return pd.DataFrame([row('Traded', 'AAA', 600, 30, 2),
                             row('Other', 'AAA', 1800, 10, 3),
                             row('Traded', 'BBB', 2400, 20, 4),
                             row('Traded', '2TM', 3000, 22, 6)])

    def aggregate(self, source):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'advanced.csv'
            source.to_csv(path, index=False)
            return add_team_advanced(self.metrics(), path)

    def test_stints_and_weights_and_per_game_totals(self):
        result, audit = self.aggregate(self.source())
        self.assertTrue(audit.Eligible.all())
        self.assertEqual(result.Team_PER_MinuteWeighted.tolist(), [15, 20])
        self.assertEqual(result.Team_WS_Per_Game.tolist(), [.5, .4])
        self.assertEqual(result.Team_VORP_Per_Game.tolist(), [.25, .2])
        self.assertTrue(audit.Minutes_Coverage.eq(1).all())
        self.assertAlmostEqual(result['Team_TS%'].iloc[0], 110/(2*(90+.44*20)))

    def test_missing_team_and_metric_fail_coverage(self):
        source = self.source().loc[lambda frame: frame.Team.eq('AAA')].copy()
        source.loc[source.Player.eq('Other'), 'BPM'] = None
        _, audit = self.aggregate(source)
        self.assertFalse(audit.Eligible.any())

    def test_awards_and_outcomes_do_not_affect_aggregates(self):
        source = self.source()
        before, _ = self.aggregate(source)
        source['Awards'] = 'MVP-9'
        source['Won_Title'] = True
        after, _ = self.aggregate(source)
        pd.testing.assert_frame_equal(before, after)
        self.assertFalse(any('Roster_' in column or 'HHI' in column for column in after))

    def test_advanced_features_survive_both_feature_selectors(self):
        priority = ('Win_Percentage', *TEAM_ADVANCED_FEATURES, *BEST_PLAYER_FEATURES)
        frame = pd.DataFrame({column: range(50) for column in (*priority, 'PTS', 'AST', 'TRB', 'FGA', 'Age', 'TOV')})
        for selector in (logistic_features, xgb_features):
            selected = selector(frame, pd.Series(True, index=frame.index), .35, 32)
            self.assertTrue(set(priority).issubset(selected))
            self.assertEqual(len(selected), 32)

    def test_best_player_stats_belong_to_one_selected_player(self):
        result, _ = self.aggregate(self.source())
        self.assertEqual(result.Best_Player_Name.tolist(), ['Traded', 'Traded'])
        self.assertEqual(result.Best_Player_PER.tolist(), [30, 20])
        # AAA's other player has more WS, but all selected stats must come
        # from the BPM leader rather than taking each stat's maximum.
        self.assertEqual(result.Best_Player_WS.tolist(), [2, 4])

    def test_tiny_sample_does_not_replace_established_best_player(self):
        source = self.source()
        tiny = source.iloc[0].copy()
        tiny['Player'], tiny['MP'], tiny['BPM'] = 'Tiny sample', 10, 100
        result, _ = self.aggregate(pd.concat([source, tiny.to_frame().T], ignore_index=True))
        self.assertEqual(result.loc[result.Team.eq('AAA'), 'Best_Player_Name'].iloc[0], 'Traded')

    def test_no_qualifiers_falls_back_to_most_used_player(self):
        source = self.source()
        source.loc[source.Player.eq('Traded') & source.Team.eq('AAA'), 'MP'] = 10
        source.loc[source.Player.eq('Other'), 'MP'] = 20
        result, audit = self.aggregate(source)
        self.assertEqual(result.loc[result.Team.eq('AAA'), 'Best_Player_Name'].iloc[0], 'Other')
        self.assertTrue(audit.loc[audit.Team.eq('AAA'), 'Best_Player_Selection_Fallback'].iloc[0])

    def test_selected_box_scores_use_the_same_team_stint(self):
        with tempfile.TemporaryDirectory() as directory:
            advanced_path = Path(directory)/'advanced.csv'
            raw_path = Path(directory)/'totals.csv'
            self.source().to_csv(advanced_path, index=False)
            rows = [dict(Season=2024, Player='Traded', Team=team, Age=25, G=10,
                         PTS=pts, TRB=100, AST=50, STL=10, BLK=5, TOV=20)
                    for team, pts in [('AAA', 200), ('BBB', 300), ('2TM', 500)]]
            pd.DataFrame(rows).to_csv(raw_path, index=False)
            result, audit = add_team_advanced(self.metrics(), advanced_path, raw_path)
            self.assertTrue(audit.Eligible.all())
            pd.DataFrame(rows[1:]).to_csv(raw_path, index=False)
            _, missing_audit = add_team_advanced(self.metrics(), advanced_path, raw_path)
            self.assertFalse(missing_audit.loc[missing_audit.Team.eq('AAA'), 'Eligible'].iloc[0])
        self.assertEqual(result.Best_Player_PTS_Per_Game.tolist(), [20, 30])

    def test_second_player_is_distinct_and_has_own_statistics(self):
        source = self.source()
        source = source.loc[source.Team.ne('2TM')].copy()
        source['_team_key'] = source.Team.map(team_key)
        source['_player_key'] = source.Player.map(player_key)
        second = select_best_players(source, self.metrics(), None, rank=2)
        self.assertEqual(second.Second_Player_Name.tolist(), ['Other'])
        self.assertEqual(second.Second_Player_PER.tolist(), [10])
        self.assertEqual(second.Second_Player_WS.tolist(), [3])
        priority = ('Win_Percentage', *TEAM_ADVANCED_FEATURES, *BEST_PLAYER_FEATURES, *SECOND_PLAYER_FEATURES)
        frame = pd.DataFrame({column: range(50) for column in priority})
        for selector in (logistic_features, xgb_features):
            self.assertEqual(set(selector(frame, pd.Series(True, index=frame.index), .35, 56)), set(priority))

    def test_pre_1974_uses_per_and_preserves_missing_bpm(self):
        source = self.source()
        source['Season'] = 1973
        metrics = self.metrics()
        metrics['Season'] = 1973
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'advanced.csv'
            source.to_csv(path, index=False)
            result, audit = add_team_advanced(metrics, path)
        self.assertTrue(audit.Eligible.all())
        self.assertTrue(result.Team_BPM_MinuteWeighted.isna().all())
        self.assertTrue(result.Team_VORP_Per_Game.isna().all())
        self.assertTrue(result.Best_Player_BPM.isna().all())
        self.assertTrue(audit.Best_Player_Selection_Metric.eq('PER').all())
        self.assertFalse(audit.BPM_Required_In_Era.any())

    def test_1974_missing_usage_and_turnovers_are_expected(self):
        source = self.source()
        source['Season'] = 1974
        source['USG%'] = None
        metrics = self.metrics()
        metrics['Season'] = 1974
        raw = source[['Season', 'Player', 'Team', 'Age', 'G']].copy()
        for column in ('PTS','TRB','AST','STL','BLK'):
            raw[column] = 100
        raw['TOV'] = None
        with tempfile.TemporaryDirectory() as directory:
            path, raw_path = Path(directory)/'advanced.csv', Path(directory)/'raw.csv'
            source.to_csv(path, index=False)
            raw.to_csv(raw_path, index=False)
            result, audit = add_team_advanced(metrics, path, raw_path)
        self.assertTrue(audit.Eligible.all())
        self.assertTrue(result.Best_Player_TOV_Per_Game.isna().all())
        self.assertTrue(result['Best_Player_USG%'].isna().all())
        self.assertTrue(audit.Best_Player_Selection_Metric.eq('BPM').all())

    def test_modern_missing_usage_is_still_a_failure(self):
        source = self.source()
        source['USG%'] = None
        raw = source[['Season', 'Player', 'Team', 'Age', 'G']].copy()
        for column in ('PTS','TRB','AST','STL','BLK','TOV'):
            raw[column] = 100
        with tempfile.TemporaryDirectory() as directory:
            path, raw_path = Path(directory)/'advanced.csv', Path(directory)/'raw.csv'
            source.to_csv(path, index=False)
            raw.to_csv(raw_path, index=False)
            _, audit = add_team_advanced(self.metrics(), path, raw_path)
        self.assertFalse(audit.Eligible.any())
        self.assertTrue(audit.Failure_Reasons.str.contains('USG%', regex=False).all())

    def test_era_indicators_survive_within_season_normalization(self):
        frame = pd.DataFrame({'Season': [1973, 1973, 1974, 1974],
                              'Era_BPM_Available': [0, 0, 1, 1],
                              'PTS': [100, 110, 105, 115]})
        for transform in (logistic_zscores, xgb_zscores):
            result = transform(frame, ['Era_BPM_Available', 'PTS'])
            self.assertEqual(result.Era_BPM_Available.tolist(), [0, 0, 1, 1])
            self.assertFalse(result.isna().any().any())


if __name__ == '__main__':
    unittest.main()
