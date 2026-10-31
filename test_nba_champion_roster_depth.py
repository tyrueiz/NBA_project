import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from nba_advanced_stats import player_key, team_key
from nba_champion_best_player import select_best_players
from nba_champion_roster_depth import ROSTER_DEPTH_FEATURES, add_roster_depth
from nba_compare_roster_depth import configurations, fit_configuration, validate_dates
from nba_predict_champions_2010_2026 import available_features as logistic_features
from nba_predict_champions_xgboost_2010_2026 import available_features as xgb_features
from nba_predict_champions_2010_2026 import to_season_zscores as logistic_zscores
from nba_predict_champions_xgboost_2010_2026 import to_season_zscores as xgb_zscores


class RosterDepthTests(unittest.TestCase):
    def source(self, year=2024):
        minutes = [400, 350, 300, 250, 200, 150, 120, 100, 90, 40]
        bpms = [10, 9, 8, 7, 6, 5, 4, 3, 100, 200]
        return pd.DataFrame([dict(Season=year, Team='AAA', Player=f'P{i+1}', Age=25,
                                  MP=mp, G=10, BPM=bpm, PER=20-i, WS=1, VORP=1)
                             for i, (mp, bpm) in enumerate(zip(minutes, bpms))])

    def calculate(self, source, teams=None):
        if teams is None:
            teams = source.loc[~source.Team.isin(['TOT', '2TM', '3TM']), ['Season', 'Team']].drop_duplicates()
            teams['Team_Games'] = 10
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'advanced.csv'
            source.to_csv(path, index=False)
            return add_roster_depth(teams, path)

    def test_exact_calculations_and_short_stint_threshold(self):
        frame, audit, players = self.calculate(self.source())
        row = frame.iloc[0]
        self.assertEqual(row.Third_Player_BPM, 8)
        self.assertEqual(row.Fourth_Player_BPM, 7)
        expected = (8*300 + 7*250 + 6*200 + 5*150 + 4*120 + 3*100) / 1120
        self.assertAlmostEqual(row.Supporting_Cast_BPM, expected)
        self.assertEqual(row.Rotation_Depth, 7)
        self.assertAlmostEqual(row.Top_2_Minutes_Share, 750/2000)
        self.assertEqual(audit.Third_Player_Name.iloc[0], 'P3')
        self.assertEqual(audit.Fourth_Player_Name.iloc[0], 'P4')
        self.assertEqual(audit.Supporting_Cast_Players.iloc[0], 6)
        # P10 has BPM 200 but only four minutes per team game, even if it
        # appeared in just one game: neither ranking nor rotation is inflated.
        self.assertEqual(players.loc[players.Player.eq('P10'), 'Player_Rank'].iloc[0], 10)

    def test_share_uses_the_existing_leader_selection(self):
        source = self.source()
        metrics = pd.DataFrame([dict(Season=2024, Team='AAA', Team_Games=10)])
        source['_team_key'] = source.Team.map(team_key)
        source['_player_key'] = source.Player.map(player_key)
        best = select_best_players(source, metrics, None, 1)
        second = select_best_players(source, metrics, None, 2)
        _, audit, _ = self.calculate(source)
        self.assertEqual(audit.Best_Player_Name_For_Share.iloc[0], best.Best_Player_Name.iloc[0])
        self.assertEqual(audit.Second_Player_Name_For_Share.iloc[0], second.Second_Player_Name.iloc[0])

    def test_traded_stints_combined_rows_and_exact_duplicates(self):
        source = self.source()
        other = source.iloc[:4].copy().assign(Team='BBB', MP=[800, 700, 600, 500], BPM=[4, 3, 2, 1])
        combined = source.iloc[:1].copy().assign(Team='2TM', MP=1200, BPM=999)
        data = pd.concat([source, source.iloc[:1], other, combined,
                          combined.assign(Team='TOT'), combined.assign(Team='3TM')], ignore_index=True)
        frame, audit, players = self.calculate(data)
        aaa = frame.loc[frame.Team.eq('AAA')].iloc[0]
        bbb = frame.loc[frame.Team.eq('BBB')].iloc[0]
        self.assertAlmostEqual(aaa.Top_2_Minutes_Share, 750/2000)
        self.assertAlmostEqual(bbb.Top_2_Minutes_Share, 1500/2600)
        self.assertEqual(bbb.Third_Player_BPM, 2)
        self.assertEqual(len(players), 14)
        self.assertEqual(audit.Team_Player_Minutes.tolist(), [2000, 2600])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.calculate(pd.concat([source, source.iloc[:1].assign(MP=401)], ignore_index=True))

    def test_fewer_than_eight_qualifiers_and_fallback(self):
        source = self.source().iloc[:4].copy()
        source['MP'] = [400, 350, 90, 40]
        frame, audit, _ = self.calculate(source)
        self.assertTrue(np.isnan(frame.Supporting_Cast_BPM.iloc[0]))
        self.assertEqual(audit.Supporting_Cast_Players.iloc[0], 0)
        self.assertTrue(audit.Third_Player_Selection_Fallback.iloc[0])
        self.assertEqual(frame.Third_Player_BPM.iloc[0], 8)
        source = self.source().iloc[:4]
        frame, audit, _ = self.calculate(source)
        self.assertEqual(audit.Supporting_Cast_Players.iloc[0], 2)
        self.assertAlmostEqual(frame.Supporting_Cast_BPM.iloc[0], (8*300+7*250)/550)
        frame, audit, _ = self.calculate(source.iloc[:2])
        self.assertTrue(np.isnan(frame.Third_Player_BPM.iloc[0]))
        self.assertTrue(np.isnan(frame.Fourth_Player_BPM.iloc[0]))

    def test_before_bpm_era_uses_per_only_for_ranking(self):
        source = self.source(1973)
        source['PER'] = list(range(10, 20))
        frame, audit, players = self.calculate(source)
        self.assertTrue(frame[['Third_Player_BPM', 'Fourth_Player_BPM', 'Supporting_Cast_BPM']].isna().all().all())
        self.assertEqual(audit.Third_Player_Name.iloc[0], 'P6')
        self.assertEqual(audit.Selection_Metric.iloc[0], 'PER')
        self.assertTrue(players.BPM.isna().all())
        self.assertEqual(frame.Rotation_Depth.iloc[0], 7)

    def test_missing_bpm_is_never_zero_and_is_diagnosed(self):
        source = self.source().iloc[:4].copy()
        source.loc[source.Player.eq('P3'), 'BPM'] = np.nan
        frame, audit, players = self.calculate(source)
        self.assertEqual(frame.Third_Player_BPM.iloc[0], 7)
        self.assertEqual(frame.Supporting_Cast_BPM.iloc[0], 7)
        self.assertEqual(audit.Missing_BPM_Players.iloc[0], 1)
        self.assertAlmostEqual(audit.BPM_Minutes_Coverage.iloc[0], 1-300/1300)
        self.assertTrue(players.loc[players.Player.eq('P3'), 'Player_Rank'].isna().all())

    def test_awards_and_playoff_results_have_no_effect(self):
        source = self.source()
        first = self.calculate(source)
        second = self.calculate(source.assign(Won_Title=True, Made_Playoffs=True, Awards='MVP-1', Playoff_Result='champion'))
        for before, after in zip(first, second):
            pd.testing.assert_frame_equal(before, after)
        with self.assertRaisesRegex(ValueError, 'regular-season'):
            self.calculate(source.assign(isPlayoff=True))

    def test_depth_is_opt_in_and_additive_in_both_models(self):
        base = ['Win_Percentage', 'Team_Offensive_Rating', 'PTS', 'AST', 'Age']
        frame = pd.DataFrame({column: range(50) for column in [*base, *ROSTER_DEPTH_FEATURES]})
        train = pd.Series(True, index=frame.index)
        for selector in (logistic_features, xgb_features):
            disabled = selector(frame, train, .35, 3)
            self.assertFalse(set(disabled).intersection(ROSTER_DEPTH_FEATURES))
            self.assertEqual(selector(frame, train, .35, 3, True), [*disabled, *ROSTER_DEPTH_FEATURES])
            no_depth_columns = frame.drop(columns=list(ROSTER_DEPTH_FEATURES))
            self.assertEqual(disabled, selector(no_depth_columns, train, .35, 3))
        self.assertEqual(logistic_features(frame, train, .35, 59, True),
                         xgb_features(frame, train, .35, 59, True))
        frame['Season'] = np.repeat([2000, 2001], 25)
        pd.testing.assert_frame_equal(logistic_zscores(frame, list(ROSTER_DEPTH_FEATURES)),
                                      xgb_zscores(frame, list(ROSTER_DEPTH_FEATURES)))

    def test_training_only_eligibility_and_historical_missingness(self):
        frame = pd.DataFrame({column: range(40) for column in ROSTER_DEPTH_FEATURES})
        train = pd.Series([True]*20 + [False]*20)
        frame.loc[train, 'Third_Player_BPM'] = np.nan
        for selector in (logistic_features, xgb_features):
            selected = selector(frame, train, .35, 59, True)
            self.assertNotIn('Third_Player_BPM', selected)
            frame.loc[~train, 'Third_Player_BPM'] = -99999
            self.assertEqual(selected, selector(frame, train, .35, 59, True))


class ExperimentTests(unittest.TestCase):
    def args(self):
        return argparse.Namespace(train_start=1983, train_end=2002, validation_start=2003,
                                  validation_end=2009, backtest_start=2010, backtest_end=2026,
                                  min_feature_coverage=.35, n_estimators=2, max_depth=2, learning_rate=.04)

    def test_experiment_groups_are_strictly_additive(self):
        configs = configurations(['Win_Percentage', 'PTS'])
        self.assertEqual(configs['B'][:2], configs['A'])
        self.assertEqual(configs['C'][:-5], configs['B'])
        self.assertEqual(configs['C'][-5:], list(ROSTER_DEPTH_FEATURES))
        for change in (dict(train_end=2010), dict(validation_end=2010), dict(train_start=2000)):
            args = self.args()
            vars(args).update(change)
            with self.assertRaises(ValueError):
                validate_dates(args)

    def test_fit_ignores_future_labels_and_uses_only_training_rows(self):
        frame = pd.DataFrame([dict(Season=year, Team=team, PTS=value)
                              for year in range(1983, 2012)
                              for team, value in [('AAA', 1.), ('BBB', 2.), ('CCC', 3.)]])
        labels = {year: 'AAA' for year in range(1983, 2012)}
        calls = []

        class Spy:
            def set_params(self, **kwargs):
                return self

            def fit(self, x, y):
                calls.append((x.copy(), y.copy()))
                return self

            def predict_proba(self, x):
                return np.tile([.7, .3], (len(x), 1))

        with patch('nba_compare_roster_depth.make_xgb_pipeline', return_value=Spy()):
            _, first, _ = fit_configuration(frame, labels, ['PTS'], self.args(), 2011)
            changed = {year: 'CCC' if year >= 2003 else team for year, team in labels.items()}
            _, second, _ = fit_configuration(frame, changed, ['PTS'], self.args(), 2011)
        pd.testing.assert_frame_equal(calls[0][0], calls[1][0])
        pd.testing.assert_series_equal(calls[0][1], calls[1][1])
        self.assertEqual(len(calls[0][0]), 20*3)
        self.assertLess(frame.loc[calls[0][0].index, 'Season'].max(), 2003)
        pd.testing.assert_series_equal(first.Model_Score, second.Model_Score)

    def test_real_imputer_is_fitted_on_training_zscores_only(self):
        frame = pd.DataFrame([dict(Season=year, Team=team, PTS=value)
                              for year in range(1983, 2027)
                              for team, value in [('AAA', 1.), ('BBB', 2.), ('CCC', 10.)]])
        # Different distributions after training change the global median;
        # the fitted imputer must nevertheless use the earlier rows only.
        frame.loc[frame.Season.ge(2003) & frame.Team.eq('BBB'), 'PTS'] = 10.
        frame.loc[frame.Season.ge(2003) & frame.Team.eq('CCC'), 'PTS'] = 11.
        train = frame.Season.le(2002)
        zscores = xgb_zscores(frame, ['PTS'])
        labels = {year: 'AAA' for year in range(1983, 2027)}
        model, _, _ = fit_configuration(frame, labels, ['PTS'], self.args(), 2011)
        self.assertAlmostEqual(model.named_steps['imputer'].statistics_[0],
                               zscores.loc[train, 'PTS'].median())
        self.assertNotAlmostEqual(model.named_steps['imputer'].statistics_[0], zscores.PTS.median())


if __name__ == '__main__':
    unittest.main()
