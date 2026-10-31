import tempfile
import unittest
from pathlib import Path

import pandas as pd

from nba_predictor.nba_download_team_ratings import TEAM_RATING_FEATURES, normalize_ratings
from nba_predictor.nba_team_ratings import add_team_ratings
from nba_predictor.nba_champion_best_player import TOP_PLAYER_FEATURES
from nba_predictor.nba_champion_advanced import TEAM_ADVANCED_FEATURES
from nba_predictor.nba_champion_first_time import FIRST_TIME_FEATURES
from nba_predictor.nba_predict_champions_2010_2026 import available_features as logistic_features
from nba_predictor.nba_predict_champions_xgboost_2010_2026 import available_features as xgb_features


class TeamRatingTests(unittest.TestCase):
    def raw(self):
        return pd.DataFrame([dict(season=2024, lg='NBA', abbreviation='BKN',
                                  o_rtg=120, d_rtg=110, playoffs=True),
                             dict(season=1973, lg='ABA', abbreviation='ABC',
                                  o_rtg=999, d_rtg=1, playoffs=True)])

    def test_nba_only_and_net_calculated_from_actual_team_ratings(self):
        data = normalize_ratings(self.raw())
        self.assertEqual(data.Team.tolist(), ['BRK'])
        self.assertEqual(data.Team_Net_Rating.tolist(), [10])

    def test_identity_join_and_missing_season_are_audited(self):
        ratings = normalize_ratings(self.raw())
        metrics = pd.DataFrame([dict(Season=2024, Team='BRK'), dict(Season=2025, Team='BRK')])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'ratings.csv'
            ratings.to_csv(path, index=False)
            result, audit = add_team_ratings(metrics, path)
        self.assertEqual(len(result), 2)
        self.assertEqual(audit.Eligible.tolist(), [True, False])
        self.assertEqual(result.Team_Offensive_Rating.iloc[0], 120)

    def test_postseason_and_duplicate_ratings_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'regular-season'):
            normalize_ratings(self.raw().assign(isPlayoff=True))
        ratings = normalize_ratings(self.raw())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'ratings.csv'
            pd.concat([ratings, ratings]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                add_team_ratings(pd.DataFrame([dict(Season=2024, Team='BRK')]), path)

    def test_all_requested_features_fit_under_new_default_cap(self):
        features = ('Win_Percentage', *TEAM_RATING_FEATURES, *TEAM_ADVANCED_FEATURES,
                    *TOP_PLAYER_FEATURES, *FIRST_TIME_FEATURES)
        frame = pd.DataFrame({feature: range(50) for feature in features})
        for selector in (logistic_features, xgb_features):
            chosen = selector(frame, pd.Series(True, index=frame.index), .35, 59)
            self.assertEqual(set(chosen), set(features))

    def test_third_player_columns_cannot_enter_either_model(self):
        frame = pd.DataFrame({column: range(50) for column in
                              (*TEAM_RATING_FEATURES, 'Third_Player_PER', 'Third_Player_BPM',
                               'Third_Player_PTS_Per_Game')})
        for selector in (logistic_features, xgb_features):
            chosen = selector(frame, pd.Series(True, index=frame.index), .35, 59)
            self.assertEqual(set(chosen), set(TEAM_RATING_FEATURES))


if __name__ == '__main__':
    unittest.main()
