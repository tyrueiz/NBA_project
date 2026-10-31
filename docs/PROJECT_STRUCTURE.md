# Project structure

Run all commands from the repository root. The Python modules form the
`nba_predictor` package, so no installation or custom `PYTHONPATH` is needed.

| Location | Purpose |
| --- | --- |
| `nba_predictor/nba_unified_scraper_*.py` | Regular-season data collection |
| `nba_predictor/nba_download_*.py` | Advanced statistics, team ratings and playoff history downloads |
| `nba_predictor/nba_advanced_stats.py`, `nba_team_ratings.py` | Shared normalization and team-rating joins |
| `nba_predictor/nba_champion_*.py` | Championship feature builders and training-window search |
| `nba_predictor/nba_predict_champions_*.py` | Logistic and XGBoost championship commands |
| `nba_predictor/nba_mvp_common.py`, `nba_predict_mvp_*.py` | Shared MVP logic and prediction commands |
| `nba_predictor/nba_compare_*.py` | Controlled roster-depth and playoff-experience comparisons |
| `nba_predictor/nba_champions_vs_league.py` | Team metrics and historical champion analysis |
| `nba_predictor/render_champion_leaderboard.py` | Leaderboard rendering |
| `tests/` | Offline tests |
| `docs/` | Model context and organization guide |
| `data/nba_updated/` | Local datasets and download caches |
| `outputs/` | Local predictions, analyses, comparisons and fit caches |

In rows listing several filenames, all Python files live under `nba_predictor/`.

## Command migration

Replace `python3 NAME.py` with `python3 -m nba_predictor.NAME`. For example:

```bash
python3 -m nba_predictor.nba_predict_champions_xgboost_2010_2026 \
  --train-start 1983 --train-end 2002 \
  --predict-start 2010 --predict-end 2026 \
  --max-features 73 \
  --include-playoff-experience --no-playoff-continuity --no-roster-depth \
  --output-dir outputs/nba_champion_predictions_actual_experience

python3 -m nba_predictor.nba_predict_mvp_linear --help
python3 -m nba_predictor.nba_predict_mvp_xgboost --help
python3 -m nba_predictor.nba_champion_window_search --help
python3 -m unittest discover -s tests -v
```

The prediction command writes into the named output directory. Choose a new
name if you want to keep the previous run.

All default input/output paths now match the folder layout. Explicit paths
are still accepted and are resolved relative to your working directory.

## Existing local files

The original `nba_updated/` directory moved to `data/nba_updated/`.
All original generated directories moved to `outputs/`, retaining their names
and file contents. Historical reports and manifests therefore still contain
some original paths; interpret them using this relocation. Old cache keys may
change because code and input paths are part of the fingerprint.

Downloaded datasets and generated outputs remain ignored by Git. Their folder
README files are tracked. A fresh clone needs downloaded data and generated
team metrics before prediction. Controlled comparisons also require the saved
baseline metadata specified by `--baseline-details`; local saved baselines
have been preserved under `outputs/`.

The pre-existing stray `g` in the no-money scraper's argument definition is
preserved as an uncommitted local edit. It causes that script to fail to parse;
use the standard scraper until that separate edit is corrected.
