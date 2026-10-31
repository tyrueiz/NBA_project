# Local datasets

`nba_updated/` contains the regular-season exports, advanced player statistics,
team ratings, playoff histories, and downloaded source caches. Dataset contents
are ignored by Git; the reorganization preserved the existing files.

From the project root, use the collection modules:

```bash
python3 -m nba_predictor.nba_unified_scraper_1947_2027 --help
python3 -m nba_predictor.nba_download_advanced --help
python3 -m nba_predictor.nba_download_team_ratings --help
python3 -m nba_predictor.nba_download_playoff_history --help
```

The default dataset directory is `data/nba_updated/`. These commands with
`--help` describe available options without downloading data.
