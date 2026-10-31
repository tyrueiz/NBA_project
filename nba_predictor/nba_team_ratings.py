"""Join actual regular-season team efficiencies, independent of player proxies."""
from pathlib import Path

import numpy as np
import pandas as pd

from nba_predictor.nba_advanced_stats import team_key
from nba_predictor.nba_download_team_ratings import TEAM_RATING_FEATURES


def add_team_ratings(metrics: pd.DataFrame, path: Path):
    ratings = pd.read_csv(path, low_memory=False)
    required = {'Season', 'Team', *TEAM_RATING_FEATURES}
    if not required.issubset(ratings):
        raise ValueError(f'Team ratings CSV requires {sorted(required)}')
    if 'isPlayoff' in ratings and not ratings.isPlayoff.astype(str).str.lower().isin(['false', '0']).all():
        raise ValueError('Team ratings must contain regular-season data only')
    ratings = ratings[['Season', 'Team', *TEAM_RATING_FEATURES]].copy()
    for column in ('Season', *TEAM_RATING_FEATURES):
        ratings[column] = pd.to_numeric(ratings[column], errors='coerce')
    ratings['_rating_team'] = ratings.Team.map(team_key)
    if ratings.duplicated(['Season', '_rating_team']).any():
        raise ValueError('Duplicate team-season rating rows')
    net = ratings[TEAM_RATING_FEATURES[0]] - ratings[TEAM_RATING_FEATURES[1]]
    if not np.isclose(net, ratings[TEAM_RATING_FEATURES[2]], atol=.15, equal_nan=True).all():
        raise ValueError('Net rating is inconsistent with offensive minus defensive rating')
    result = metrics.drop(columns=list(TEAM_RATING_FEATURES), errors='ignore').copy()
    result['_rating_team'] = result.Team.map(team_key)
    result = result.merge(ratings[['Season', '_rating_team', *TEAM_RATING_FEATURES]],
                          on=['Season', '_rating_team'], how='left', validate='one_to_one')
    audit = result[['Season', 'Team', *TEAM_RATING_FEATURES]].copy()
    audit['Eligible'] = np.isfinite(audit[list(TEAM_RATING_FEATURES)]).all(axis=1)
    result = result.drop(columns='_rating_team')
    return result, audit
