"""Date-search boundaries, holdout isolation, and fit-cache regression checks."""
import argparse
from copy import deepcopy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from nba_champion_window_search import fit_window, leaderboard, training_windows, wilson_lower


class WindowSearchTests(unittest.TestCase):
    def test_all_default_training_windows_obey_chronology(self):
        windows = list(training_windows(1973, 2025, 10, 1))
        self.assertEqual(len(windows), 946)
        self.assertEqual(windows[0], (1973, 1982))
        self.assertEqual(windows[-1], (2015, 2024))
        self.assertTrue(all(start >= 1973 and cutoff-start+1 >= 10 and cutoff < 2025
                            for start, cutoff in windows))
        self.assertEqual(sum(2025-cutoff for _, cutoff in windows), 14190)

    def fits(self):
        def prediction(year, correct):
            return dict(Season=year, Actual_Champion='AAA', Actual_Rank=1 if correct else 4,
                        Correct=correct, Actual_In_Top3=correct)
        return [dict(Model=model, Train_Start=1973, Train_End=1982, Features_Used=5,
                     Predictions=[prediction(1983, model=='logistic'),
                                  prediction(1984, True), prediction(1985, False)])
                for model in ('logistic', 'xgboost')]

    def test_later_test_results_never_change_leaderboard_order(self):
        fits = self.fits()
        before = leaderboard(fits, 1984, 1985, 1)
        changed = deepcopy(fits)
        for fit in changed:
            fit['Predictions'][-1].update(Correct=True, Actual_In_Top3=True, Actual_Rank=1)
        after = leaderboard(changed, 1984, 1985, 1)
        identity = ['Model','Train_Start','Train_End','Validation_Start','Validation_End','Rank']
        pd.testing.assert_frame_equal(before[identity], after[identity])
        self.assertFalse(before.Later_Test_Top1.equals(after.Later_Test_Top1))

    def test_validation_and_later_test_do_not_overlap(self):
        result = leaderboard(self.fits(), 1984, 1985, 1)
        self.assertTrue((result.Later_Test_Start == result.Validation_End + 1).all())
        self.assertTrue((result.Validation_Start > result.Train_End).all())
        self.assertEqual(len(result), 4)

    def test_ranking_accounts_for_sample_size(self):
        self.assertGreater(wilson_lower(8, 10), wilson_lower(1, 1))

    def test_fit_sees_only_training_rows_and_resumes_from_cache(self):
        frame = pd.DataFrame([dict(Season=year, Team=team, Win_Percentage=score,
                                    _label=int(team=='DDD'))
                              for year in range(2000, 2013)
                              for team, score in zip(('AAA','BBB','CCC','DDD'),(.2,.4,.6,.8))])
        zscores = frame[['Win_Percentage']]
        calls = []

        class Spy:
            def set_params(self, **kwargs):
                return self

            def fit(self, x, y):
                calls.append((x.index.tolist(), y.index.tolist()))
                return self

            def predict_proba(self, x):
                scores = x.Win_Percentage.to_numpy()
                return np.column_stack([1-scores, scores])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'fit_cache').mkdir()
            args = argparse.Namespace(output_dir=root, min_feature_coverage=.35, max_features=56)
            with patch('nba_champion_window_search.make_xgb_pipeline', return_value=Spy()):
                fit = fit_window('xgboost', 2000, 2009, frame, zscores,
                                 {year:'DDD' for year in range(2000,2013)}, args)
                resumed = fit_window('xgboost', 2000, 2009, frame, zscores, {}, args)
            self.assertEqual(fit, resumed)
            self.assertEqual(len(calls), 1)
            expected = frame.index[frame.Season.le(2009)].tolist()
            self.assertEqual(calls[0], (expected, expected))
            self.assertEqual([row['Season'] for row in fit['Predictions']], [2010,2011,2012])


if __name__ == '__main__':
    unittest.main()
