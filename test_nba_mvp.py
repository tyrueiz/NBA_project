"""Regression checks for award leakage and traded-player accounting."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from nba_mvp_common import (add_team_context, add_voting_history, load_players,
                            make_features, score, summarize, CONTEXT_FEATURES)


class MVPDataTests(unittest.TestCase):
    def load(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'stats.csv'
            pd.DataFrame(rows).to_csv(path, index=False)
            return load_players(path)

    def row(self, **overrides):
        row = dict(Season=2024, Player='Player A', Age=28, Team='AAA',
                   G=40, PTS=1000, FGA=700, FTA=200, Awards='MVP-1,AS')
        return row | overrides

    def test_combined_totals_replace_traded_stints(self):
        data = self.load([self.row(), self.row(Team='BBB', G=20, PTS=500),
                          self.row(Team='2TM', G=60, PTS=1500)])
        self.assertEqual(len(data), 1)
        self.assertEqual(data.iloc[0].PTS, 1500)
        self.assertEqual(data.iloc[0].G, 60)

    def test_namesakes_are_preserved(self):
        data = self.load([self.row(), self.row(Age=23, Team='BBB', Awards='')])
        self.assertEqual(len(data), 2)

    def test_awards_and_postseason_cannot_change_features(self):
        data = self.load([self.row(), self.row(Player='Player B', PTS=600, Awards='FMVP-1')])
        self.assertEqual(data.MVP_Target_LABEL_ONLY.tolist(), [1, 0])
        before = make_features(data)
        data['Awards'] = 'MVP-1'
        data['MVP_Target_LABEL_ONLY'] = 123
        data['Playoff_PTS'] = 9999
        pd.testing.assert_frame_equal(before, make_features(data))

    def test_missing_truth_is_not_scored_as_a_loss(self):
        data = self.load([self.row(Awards='')])
        predictions = data[['Candidate_ID', 'Season', 'Player', 'Team']].assign(Rank=1)
        result = summarize(predictions, data)
        self.assertTrue(pd.isna(result.iloc[0].Correct))

    def test_top_three_are_ranked_separately_for_each_season(self):
        class FixedModel:
            def predict(self, features):
                return features['score'].to_numpy()

        data = self.load([self.row(Season=year, Player=f'Player {i}', Awards='')
                          for year in (2024, 2025) for i in range(4)])
        features = pd.DataFrame({'score': [2, 4, 1, 3, 7, 5, 8, 6]}, index=data.index)
        predictions = score(FixedModel(), features, data, data.G.gt(0))
        selected = predictions.loc[predictions.Predicted_Top3]
        self.assertEqual(selected.groupby('Season').size().tolist(), [3, 3])
        self.assertEqual(selected.Player.tolist(),
                         ['Player 1', 'Player 3', 'Player 0', 'Player 2', 'Player 0', 'Player 3'])
        summary = summarize(predictions, data)
        self.assertEqual(summary.Predicted_2nd.tolist(), ['Player 3', 'Player 0'])
        self.assertEqual(summary.Predicted_3rd.tolist(), ['Player 0', 'Player 3'])

    def test_history_is_strictly_past_and_counts_consecutive_wins(self):
        data = self.load([self.row(Season=2022, Age=26),
                          self.row(Season=2023, Age=27),
                          self.row(Season=2024, Age=28, Awards='MVP-2'),
                          self.row(Season=2025, Age=29, Awards='')])
        before = add_voting_history(data)
        current = before.loc[before.Season.eq(2024)].iloc[0]
        self.assertEqual(current.Career_MVP_Wins, 2)
        self.assertEqual(current.Consecutive_MVP_Wins, 2)
        self.assertEqual(current.Previous_MVP_Winner, 1)
        self.assertEqual(before.loc[before.Season.eq(2025)].iloc[0].Consecutive_MVP_Wins, 0)
        data.loc[data.Season.ge(2024), 'MVP_Rank_LABEL_ONLY'] = 1
        after = add_voting_history(data)
        columns = list(CONTEXT_FEATURES[2:])
        pd.testing.assert_frame_equal(before.loc[before.Season.le(2024), columns],
                                      after.loc[after.Season.le(2024), columns])

    def test_nonwinning_voting_rank_cannot_affect_future_features(self):
        data = self.load([self.row(Season=2023, Age=27, Awards='MVP-2'),
                          self.row(Season=2024, Age=28, Awards='')])
        before = add_voting_history(data)
        data.loc[data.Season.eq(2023), 'MVP_Rank_LABEL_ONLY'] = 9
        after = add_voting_history(data)
        pd.testing.assert_frame_equal(before[list(CONTEXT_FEATURES[2:])],
                                      after[list(CONTEXT_FEATURES[2:])])
        pd.testing.assert_frame_equal(
            make_features(before).loc[before.Season.eq(2024)],
            make_features(after).loc[after.Season.eq(2024)])
        self.assertNotIn('Previous_MVP_Score', make_features(after).columns)

    def test_team_context_weights_stints_and_ignores_playoff_results(self):
        rows = [self.row(G=30), self.row(Team='BBB', G=10),
                self.row(Team='2TM', G=40)]
        with tempfile.TemporaryDirectory() as directory:
            stats = Path(directory) / 'stats.csv'
            teams = Path(directory) / 'teams.csv'
            pd.DataFrame(rows).to_csv(stats, index=False)
            team_data = pd.DataFrame([
                dict(Season=2024, Team='AAA', Wins=60, Losses=20, Won_Title=False),
                dict(Season=2024, Team='BBB', Wins=20, Losses=60, Won_Title=True)])
            team_data.to_csv(teams, index=False)
            before = add_team_context(load_players(stats), stats, teams)
            self.assertAlmostEqual(before.iloc[0].Team_Win_Percentage, .625)
            self.assertAlmostEqual(before.iloc[0].Team_Wins, 50)
            team_data['Won_Title'] = ~team_data.Won_Title
            team_data.to_csv(teams, index=False)
            pd.testing.assert_frame_equal(before, add_team_context(load_players(stats), stats, teams))


if __name__ == '__main__':
    unittest.main()
