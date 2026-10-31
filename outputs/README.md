# Local generated outputs

All existing prediction, analysis, window-search, and comparison directories
were moved here with their original names. Their contents are ignored by Git.

- `nba_champions_analysis/`: prepared team metrics and historical analysis.
- `nba_champion_predictions*/`: championship predictions and feature audits.
- `nba_mvp_predictions*/`: MVP predictions.
- `nba_champion_window_search/`: training-window leaderboard and fit cache.
- `nba_roster_depth_experiments*/`: controlled depth comparisons.
- `nba_playoff_experience*/`: playoff-history experiments and checks.

The verified playoff comparison is in
`nba_playoff_experience_comparison_verified_1983_2002/REPORT.md`.
Saved reports and manifests retain their original historical path strings;
resolve former `nba_updated/...` paths under `data/` and former generated
output directories under `outputs/`. File contents and model results were not
rewritten during migration. Code/path fingerprints can invalidate cached fits
when experiments are rerun, so a cache recomputation can occur.

For a fresh analysis, run from the project root:

```bash
python3 -m nba_predictor.nba_champions_vs_league --input-dir data/nba_updated
```

Use a new `--output-dir outputs/NAME` to preserve an earlier experiment.
