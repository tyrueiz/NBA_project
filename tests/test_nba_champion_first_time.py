"""Verify prior-history proxies, including moves and absence of target leakage."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from nba_predictor.nba_champion_first_time import FIRST_TIME_FEATURES, add_first_time_features
from nba_predictor.nba_predict_champions_2010_2026 import available_features as logistic_features
from nba_predictor.nba_predict_champions_xgboost_2010_2026 import available_features as xgb_features


class FirstTimeTests(unittest.TestCase):
    def fixtures(self):
        metrics = pd.DataFrame([
            dict(Season=2024, Team='AAA', Best_Player_Name='Rookie', Best_Player_Age=20,
                 Second_Player_Name='Other rookie', Second_Player_Age=21),
            dict(Season=2024, Team='BBB', Best_Player_Name='Veteran', Best_Player_Age=26,
                 Second_Player_Name='New teammate', Second_Player_Age=22),
            dict(Season=2024, Team='CCC', Best_Player_Name='New leader', Best_Player_Age=22,
                 Second_Player_Name='New second', Second_Player_Age=21)])
        outcomes = pd.DataFrame([dict(Season=year, Team=team, Made_Playoffs=int(team == 'AAA' or year == 2024))
                                 for year in (2022, 2023, 2024) for team in ('AAA', 'BBB', 'CCC')])
        roster = pd.DataFrame([
            dict(Season=2022, Team='AAA', Player='Veteran', Age=24, MP=2000),
            dict(Season=2023, Team='AAA', Player='Veteran', Age=25, MP=2000),
            *[dict(Season=2024, Team=row.Team, Player=getattr(row, f'{prefix}_Player_Name'),
                   Age=getattr(row, f'{prefix}_Player_Age'), MP=2000)
              for row in metrics.itertuples() for prefix in ('Best', 'Second')]])
        return metrics, outcomes, roster

    def calculate(self, metrics, outcomes, roster):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'stats.csv'
            roster.to_csv(path, index=False)
            return add_first_time_features(metrics, outcomes, path)[0]

    def test_experience_follows_players_not_their_current_franchise(self):
        metrics, outcomes, roster = self.fixtures()
        result = self.calculate(metrics, outcomes, roster).set_index('Team')
        self.assertEqual(result.loc['BBB', FIRST_TIME_FEATURES[0]], 2)
        self.assertEqual(result.loc['AAA', FIRST_TIME_FEATURES[0]], 0)
        self.assertEqual(result.loc['AAA', 'Both_Leaders_First_Playoffs_Proxy'], 1)
        self.assertEqual(result.loc['BBB', 'New_Contender_Proxy'], 0)
        self.assertEqual(result.loc['CCC', 'New_Contender_Proxy'], 1)

    def test_current_and_future_playoff_results_cannot_change_features(self):
        metrics, outcomes, roster = self.fixtures()
        before = self.calculate(metrics, outcomes, roster)
        outcomes.loc[outcomes.Season.eq(2024), 'Made_Playoffs'] = 0
        outcomes = pd.concat([outcomes, pd.DataFrame([dict(Season=2025, Team='AAA', Made_Playoffs=1)])])
        pd.testing.assert_frame_equal(before, self.calculate(metrics, outcomes, roster))

    def test_traded_player_receives_one_credit_per_season(self):
        metrics, outcomes, roster = self.fixtures()
        roster = pd.concat([roster, pd.DataFrame([dict(Season=2023, Team='BBB', Player='Veteran', Age=25, MP=100)])])
        outcomes.loc[outcomes.Season.eq(2023) & outcomes.Team.eq('BBB'), 'Made_Playoffs'] = 1
        result = self.calculate(metrics, outcomes, roster).set_index('Team')
        self.assertEqual(result.loc['BBB', FIRST_TIME_FEATURES[0]], 2)

    def test_only_allowlisted_prior_features_bypass_postseason_filter(self):
        columns = (*FIRST_TIME_FEATURES, 'Playoff_Result', 'Won_Title', 'Playoff_Series_Won')
        frame = pd.DataFrame({column: range(50) for column in columns})
        for selector in (logistic_features, xgb_features):
            result = selector(frame, pd.Series(True, index=frame.index), .35, 56)
            self.assertEqual(set(result), set(FIRST_TIME_FEATURES))


if __name__ == '__main__':
    unittest.main()
