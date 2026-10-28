#!/usr/bin/env python3
"""XGBoost NBA champion backtest, trained strictly on seasons ending <= 2009.

Inputs (from NBA champions analysis and original scraper):
  nba_champions_analysis/team_metrics_all_seasons.csv
  nba_champions_analysis/champions_vs_league_all_stats.csv (optional label fallback)
  nba_updated/NBA_team_seasons_1947_2027.csv
  nba_updated/NBA_stats_1947_2027.csv (optional, for roster distribution)

For EACH season 2010..2026, predict the title winner using REGULAR-SEASON
stats of that season, but NEVER train on its championship outcome, postseason
stats, or outcomes after 2009. This is a post-regular-season retrospective
prediction, NOT a preseason prediction. The 2010–2026 winners are used
ONLY AFTER prediction for optional scoring, NEVER for model training.

Install: python3 -m pip install pandas numpy scikit-learn xgboost
Run: python3 nba_predict_champions_xgboost_2010_2026.py \
        --analysis-dir nba_champions_analysis --input-dir nba_updated

No network requests are made.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

ALIASES = {
    'BKN':'BRK','BRK':'BKN','CHA':'CHO','CHO':'CHA',
    'PHX':'PHO','PHO':'PHX','GS':'GSW','NOL':'NOH','NOK':'NOH','NOH':'NOK',
}
DISTRIBUTION_STATS = ('PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV', '3P')
METRIC_EXCLUDE = {
    'Season', 'Team', 'team_name', 'Team_Games', 'Number_of_Players',
    'Wins', 'Losses', 'Won_Title', 'Payroll', 'salary', 'Salary',
    'Season_Status', 'Data_Collected_UTC', 'Age_Of_Franchise',
}
LEAKAGE_PATTERN = re.compile(
    r'playoff|postseason|title|champion|finals?|series|seed|\brank\b|'
    r'won_|round|advance|result|_id$|win_total|loss_total', re.I
)


def locate(directory: Path, glob_pattern: str, required=True) -> Path | None:
    candidates = sorted(directory.glob(glob_pattern), key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    if not candidates and required:
        raise FileNotFoundError(f'No {glob_pattern} in {directory}. Run the scraper/analysis first.')
    return candidates[0] if candidates else None


def parse_flag(s: pd.Series) -> pd.Series:
    return s.astype('string').str.strip().str.lower().isin(('1','true','t','yes','y'))


def normalize_frame(df: pd.DataFrame, label: str) -> pd.DataFrame:
    df.columns = df.columns.astype(str).str.strip()
    if not {'Season','Team'}.issubset(df):
        raise ValueError(f'{label} missing Season/Team columns: {list(df.columns)}')
    df = df.copy()
    df['Season'] = pd.to_numeric(df['Season'], errors='coerce')
    df = df.dropna(subset=['Season'])
    df['Season'] = df['Season'].astype(int)
    df['Team'] = df['Team'].astype('string').str.strip().str.upper()
    df = df.loc[~df['Team'].isin(['', 'NAN', 'NONE', '<NA>'])].copy()
    return df


def season_alias_map(frame: pd.DataFrame):
    known = set(zip(frame.Season, frame.Team))
    def translate(season, code):
        if (season, code) in known: return code
        alt = ALIASES.get(code)
        return alt if alt and (season, alt) in known else None
    return translate


def load_champion_labels(team_seasons: pd.DataFrame, comparisons: pd.DataFrame | None):
    """Use playoff results ONLY for training labels <=2009, and post-hoc evaluation."""
    frames = []
    if 'Won_Title' in team_seasons:
        frames.append(team_seasons.loc[parse_flag(team_seasons.Won_Title), ['Season','Team']])
    if 'Playoff_Result' in team_seasons:
        frames.append(team_seasons.loc[
            team_seasons.Playoff_Result.astype('string').str.strip().str.casefold().eq('champion').fillna(False),
            ['Season','Team']])
    if comparisons is not None and not comparisons.empty:
        if 'Champion_Code' in comparisons:
            labels = comparisons[['Season','Champion_Code']].rename(columns={'Champion_Code':'Team'})
            frames.append(labels)
    if not frames:
        return {}
    joined = pd.concat(frames, ignore_index=True).dropna().drop_duplicates()
    joined['Season'] = pd.to_numeric(joined['Season'], errors='coerce')
    joined = joined.dropna(subset=['Season'])
    joined['Season'] = joined['Season'].astype(int)
    joined['Team'] = joined.Team.astype(str).str.strip().str.upper()
    collision = joined.groupby('Season').Team.nunique()
    bad = collision[collision > 1]
    if not bad.empty:
        raise ValueError('Conflicting championship labels in the inputs: ' + str(bad.head(10).to_dict()))
    return joined.set_index('Season').Team.to_dict()


def add_roster_concentration(metrics: pd.DataFrame, stats_path: Path | None):
    """Compute distribution features for ALL teams, not just champions.

    Each player-table row is a team-stint of regular-season TOTALS.
    Exact duplicates are removed. TOT / 2TM / 3TM aggregate rows are excluded.
    """
    if stats_path is None:
        print('No player CSV; using team aggregate statistics only.')
        return metrics, []
    roster = normalize_frame(pd.read_csv(stats_path, low_memory=False), 'player stats')
    roster = roster.loc[~roster.Team.str.fullmatch(r'TOT|[2-9]TM', na=False)].drop_duplicates()
    translate = season_alias_map(metrics)
    roster['Team'] = [translate(y, t) for y,t in zip(roster.Season, roster.Team)]
    roster = roster.dropna(subset=['Team'])
    existing = [c for c in DISTRIBUTION_STATS if c in roster]
    if not existing:
        print('Player data found, but no configured distribution stats; skipping roster features.')
        return metrics, []
    for c in existing:
        roster[c] = pd.to_numeric(roster[c], errors='coerce')
    # A player's season total must be counted once per team stint. A player who
    # changes teams legitimately appears once in each team.
    counts = roster.groupby(['Season','Team'], sort=False)
    output = []
    for (season,team), group in counts:
        row = {'Season':season, 'Team':team}
        for stat in existing:
            vals = group[stat].dropna().clip(lower=0)
            # Only include if most roster rows have the statistic available.
            if len(vals) < max(3, int(np.ceil(len(group)*.70))) or vals.sum() <= 0:
                continue
            weights = vals / vals.sum()
            row[f'Roster_{stat}_Top3Share'] = float(weights.nlargest(3).sum())
            row[f'Roster_{stat}_HHI'] = float((weights * weights).sum())
        output.append(row)
    if not output:
        return metrics, []
    derived = pd.DataFrame(output)
    result = metrics.merge(derived, on=['Season','Team'], how='left', validate='one_to_one')
    features = [c for c in derived.columns if c not in ('Season','Team')]
    print(f'Added {len(features)} roster distribution feature columns to all teams.')
    return result, features


def available_features(metrics: pd.DataFrame, train_mask: pd.Series,
                       extra: list[str], min_train_nonmissing: float, max_features: int):
    # Explicitly exclude all championship / postseason variables and identifiers.
    options = []
    for c in metrics:
        if c in METRIC_EXCLUDE or LEAKAGE_PATTERN.search(c):
            continue
        if c in ('Win_Percentage', *extra) or c in (
            'FG','FGA','FG%','3P','3PA','3P%','2P','2PA','2P%',
            'eFG%','FT','FTA','FT%','PTS','TRB','AST','STL','BLK',
            'TOV','ORB','DRB','PF','MP','GS','Age','Trp-Dbl',
            'TS%','PER','WS','WS/48','BPM','OBPM','DBPM','ORtg','DRtg'
        ):
            v = pd.to_numeric(metrics[c], errors='coerce')
            valid = v.loc[train_mask].dropna()
            coverage = v.loc[train_mask].notna().mean()
            if coverage >= min_train_nonmissing and len(valid) > 10 and valid.std() > 1e-10:
                options.append((c, coverage))
    # Favor broad historical coverage: features available only in new eras should
    # not dominate the small number of training champion labels.
    options.sort(key=lambda x: (x[0] == 'Win_Percentage', x[1]), reverse=True)
    return [c for c,_ in options[:max_features]]


def to_season_zscores(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Season-relative features use contemporaneous regular-season standings.

    The league average for test year X uses ONLY regular-season stats of year X,
    not its playoff outcomes. The model and imputer are fitted before 2010 only.
    """
    out = pd.DataFrame(index=frame.index)
    for c in columns:
        s = pd.to_numeric(frame[c], errors='coerce')
        mean = s.groupby(frame.Season).transform('mean')
        std = s.groupby(frame.Season).transform(lambda v: v.std(ddof=0))
        out[c] = ((s-mean) / std.where(std > 1e-10)).replace([np.inf,-np.inf], np.nan)
    return out


def evaluate_period(table: pd.DataFrame, predictions: pd.DataFrame, labels: dict, period: str):
    rows = []
    for season, sample in predictions.groupby('Season', sort=True):
        actual = labels.get(int(season))
        ranked = sample.sort_values('Rank')
        winner = ranked.iloc[0].Team
        actual_in_set = bool(actual in sample.Team.values) if actual else False
        rows.append({
            'Evaluation_Period':period, 'Season':int(season),
            'Predicted_Champion':winner,'Actual_Champion':actual,
            'Correct':(winner == actual) if actual_in_set else pd.NA,
            'Actual_In_Top3':(actual in ranked.head(3).Team.values) if actual_in_set else pd.NA,
            'Actual_Team_In_Input':actual_in_set if actual else pd.NA,
            'Teams_Scored':len(sample),
        })
    return pd.DataFrame(rows)


def make_xgb_pipeline(y: pd.Series, args: argparse.Namespace):
    """Train a shallow, regularized tree ensemble for rare positive titles."""
    n_positive = int(y.sum())
    n_negative = len(y) - n_positive
    if n_positive < 1 or n_negative < 1:
        raise ValueError('Need both champions and non-champions in training data.')
    model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=7.0,
        reg_alpha=0.1,
        scale_pos_weight=n_negative / n_positive,
        tree_method='hist',
        random_state=42,
        n_jobs=2,
    )
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', model),
    ])


def optional_logistic_comparison(results: pd.DataFrame, logistic_dir: Path, output_dir: Path):
    path = logistic_dir / 'predicted_champions_2010_2026.csv'
    if not path.exists():
        print(f'Logistic Regression comparison skipped ({path} not found).')
        return
    previous = pd.read_csv(path)
    need = {'Season', 'Predicted_Champion'}
    if not need.issubset(previous.columns):
        print(f'Logistic Regression comparison skipped: {path} missing columns.')
        return
    comp = results[['Season', 'Predicted_Champion', 'Actual_Champion', 'Correct']].rename(
        columns={'Predicted_Champion':'XGBoost_Predicted_Champion', 'Correct':'XGBoost_Correct'})
    base = previous[['Season', 'Predicted_Champion']].rename(
        columns={'Predicted_Champion':'Logistic_Predicted_Champion'})
    comp = comp.merge(base, on='Season', how='left', validate='one_to_one')
    comp['Logistic_Correct'] = [
        (pred == truth) if pd.notna(pred) and pd.notna(truth) else pd.NA
        for pred,truth in zip(comp.Logistic_Predicted_Champion, comp.Actual_Champion)
    ]
    comp.to_csv(output_dir / 'xgboost_vs_logistic.csv', index=False)
    print('Saved side-by-side model comparison: xgboost_vs_logistic.csv')


def main():
    parser = argparse.ArgumentParser(description='XGBoost trained only on 1947–2009; predict 2010–2026.')
    parser.add_argument('--analysis-dir', type=Path, default=Path('nba_champions_analysis'))
    parser.add_argument('--input-dir', type=Path, default=Path('nba_updated'))
    parser.add_argument('--output-dir', type=Path, default=Path('nba_champion_predictions_xgboost'))
    parser.add_argument('--logistic-dir', type=Path, default=Path('nba_champion_predictions'),
                        help='Existing Logistic Regression output directory for optional comparison')
    parser.add_argument('--n-estimators', type=int, default=240)
    parser.add_argument('--max-depth', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=0.04)
    parser.add_argument('--train-end', type=int, default=2009, help='Must be 2009 or earlier to prevent leakage')
    parser.add_argument('--predict-start', type=int, default=2010)
    parser.add_argument('--predict-end', type=int, default=2026)
    parser.add_argument('--min-team-coverage', type=float, default=.75)
    parser.add_argument('--min-feature-coverage', type=float, default=.35)
    parser.add_argument('--max-features', type=int, default=32)
    parser.add_argument('--no-roster', action='store_true', help='Exclude player distribution features')
    parser.add_argument('--no-wins', action='store_true', help='Exclude regular-season win percentage feature')
    args = parser.parse_args()
    if args.train_end > 2009 or args.predict_start < 2010 or args.predict_end > 2026:
        parser.error('This backtest forbids training after 2009 and limits predictions to 2010–2026.')
    if args.predict_start > args.predict_end: parser.error('Invalid prediction range')
    if args.n_estimators < 1 or not (1 <= args.max_depth <= 10) or not (0 < args.learning_rate <= 1):
        parser.error('Invalid XGBoost hyperparameters.')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metric_path = args.analysis_dir / 'team_metrics_all_seasons.csv'
    if not metric_path.exists():
        raise FileNotFoundError(f'Missing {metric_path}. Run nba_champions_vs_league.py first.')
    source_path = locate(args.input_dir, 'NBA_team_seasons_*.csv')
    comparison_path = args.analysis_dir / 'champions_vs_league_all_stats.csv'
    stats_path = locate(args.input_dir, 'NBA_stats_*.csv', required=False) if not args.no_roster else None

    print('Team metrics:',metric_path)
    print('Season/team results:',source_path)
    print('Player totals:',stats_path or '(not used)')
    metrics = normalize_frame(pd.read_csv(metric_path, low_memory=False), 'team metrics')
    seasons = normalize_frame(pd.read_csv(source_path, low_memory=False), 'team results')
    if metrics.duplicated(['Season','Team']).any():
        raise ValueError('Duplicate team-season keys in team_metrics_all_seasons.csv')
    if seasons.duplicated(['Season','Team']).any():
        raise ValueError('Duplicate team-season keys in NBA_team_seasons file')

    comparisons = None
    if comparison_path.exists() and comparison_path.stat().st_size:
        try:
            comparisons = pd.read_csv(comparison_path, low_memory=False)
        except pd.errors.EmptyDataError:
            pass
    labels = load_champion_labels(seasons, comparisons)
    historical_labeled = sorted(y for y in labels if y <= args.train_end)
    if len(historical_labeled) < 10:
        raise ValueError(
            f'Only {len(historical_labeled)} labeled champions up to {args.train_end}; '
            'at least 10 required. Your prior analysis reported zero champions. '
            'Check Won_Title or Playoff_Result in NBA_team_seasons_*.csv, and '
            'champions_vs_league_all_stats.csv. Never manufacture labels.'
        )
    print(f'Train window: through {args.train_end} (historical winners known: {len(historical_labeled)}).')
    print(f'Prediction window: {args.predict_start}–{args.predict_end}.')

    translate = season_alias_map(metrics)
    labels = {season: translate(season, team) or team for season,team in labels.items()}
    # Results are used solely to identify missing team-year coverage and labels.
    # No playoff results of prediction years enter the model feature matrix.
    expected = seasons.groupby('Season').Team.nunique().to_dict()
    metrics = metrics.loc[metrics.Season.between(1947,args.predict_end)].copy()
    metrics, roster_features = add_roster_concentration(metrics, stats_path)
    if args.no_wins:
        metrics = metrics.drop(columns=['Win_Percentage'],errors='ignore')

    audits=[]
    qualified_years=[]
    for year, group in metrics.groupby('Season', sort=True):
        if year > args.train_end and not args.predict_start <= year <= args.predict_end:
            continue
        total = int(expected.get(year, len(group)))
        coverage = len(group)/total if total else 0
        title = labels.get(year)
        represented = title in set(group.Team) if title else False
        status = ('TRAIN' if year <= args.train_end else 'PREDICT')
        reason = 'OK'
        if coverage < args.min_team_coverage:
            reason = f'coverage {coverage:.1%} < required {args.min_team_coverage:.0%}'
        if year <= args.train_end and not represented:
            reason = 'missing historical champion in league stats' if reason == 'OK' else reason+'; missing champion'
        if year > args.train_end and 'Season_Status' in seasons:
            src_season = seasons.loc[seasons.Season.eq(year)]
            if src_season.Season_Status.astype(str).str.lower().eq('in progress').any():
                reason = 'season in progress'
        if reason == 'OK': qualified_years.append(year)
        audits.append({'Season':year,'Period':status,'League_Teams':total,'Teams_With_Data':len(group),
                       'Team_Coverage':coverage, 'Historical_Champion_Available':represented,
                       'Eligible':reason=='OK','Reason':reason})
    audit = pd.DataFrame(audits)
    audit.to_csv(args.output_dir/'season_data_coverage.csv',index=False)
    good_years = set(qualified_years)
    frame = metrics.loc[metrics.Season.isin(good_years)].copy().reset_index(drop=True)
    train = frame.Season.le(args.train_end)
    target = frame.Season.between(args.predict_start,args.predict_end)
    if frame.loc[train,'Season'].nunique() < 10:
        raise ValueError('Fewer than ten complete training seasons. See season_data_coverage.csv')
    if not target.any():
        raise ValueError('No eligible 2010–2026 test seasons. See season_data_coverage.csv')
    frame['is_champion'] = [int(labels.get(y)==t) if y<=args.train_end else np.nan
                             for y,t in zip(frame.Season,frame.Team)]
    candidates = available_features(frame, train, roster_features,
                                    args.min_feature_coverage, args.max_features)
    if not candidates:
        raise ValueError('No usable historical regular-season stats. Review team_metrics_all_seasons.csv')
    zscores = to_season_zscores(frame, candidates)
    # This class label exists ONLY in pre-2010 training rows; test is never used for fitting.
    class_label = frame.loc[train,'is_champion'].astype(int)
    if class_label.nunique() != 2:
        raise ValueError('Historical training records lack both champions and non-champions.')
    pipe = make_xgb_pipeline(class_label, args)
    pipe.fit(zscores.loc[train,candidates], class_label)
    print(f'Model trained on {int(train.sum())} pre-2010 team-seasons across '
          f'{frame.loc[train,"Season"].nunique()} seasons, using {len(candidates)} features.')

    future = frame.loc[target, ['Season','Team']].copy()
    if 'team_name' in frame:
        future['team_name']=frame.loc[target,'team_name'].values
    # These are classification scores, NOT calibrated probabilities of winning a title.
    future['Model_Score'] = pipe.predict_proba(zscores.loc[target,candidates])[:,1]
    future['Rank'] = future.groupby('Season').Model_Score.rank(ascending=False,method='first').astype(int)
    future['Predicted_Winner'] = future.Rank.eq(1)
    # Evaluation-only truth: added after predicting; never passed to fit or transform.
    future['Actual_Champion_EVALUATION_ONLY'] = [labels.get(y) for y in future.Season]
    future['Is_Actual_Champion_EVALUATION_ONLY'] = [
        bool(labels.get(y)==t) if labels.get(y) else pd.NA
        for y,t in zip(future.Season,future.Team)
    ]
    future = future.sort_values(['Season','Rank'])
    future.to_csv(args.output_dir/'team_predictions_2010_2026.csv', index=False)

    results = evaluate_period(frame, future, labels, '2010–2026 out-of-period backtest')
    results.to_csv(args.output_dir/'predicted_champions_2010_2026.csv',index=False)
    importance = pipe.named_steps['classifier'].feature_importances_
    pd.DataFrame({'Feature':candidates, 'Importance':importance}).sort_values(
        'Importance',ascending=False).to_csv(args.output_dir/'feature_importances.csv',index=False)

    # Pre-2010 time-ordered validation: never include 2010+ winners in the fit.
    years = sorted(frame.loc[train,'Season'].unique())
    cut = max(1,int(len(years)*.7))
    earlier=set(years[:cut]); later=set(years[cut:])
    hist_val = {'train_seasons':len(earlier),'validation_seasons':len(later),
                'top1_accuracy':None,'top3_accuracy':None}
    if len(earlier) >= 10 and len(later) >= 3:
        before = train & frame.Season.isin(earlier)
        later_mask = train & frame.Season.isin(later)
        validator = make_xgb_pipeline(frame.loc[before,'is_champion'].astype(int), args)
        validator.fit(zscores.loc[before,candidates],frame.loc[before,'is_champion'].astype(int))
        val = frame.loc[later_mask,['Season','Team']].copy()
        val['Model_Score'] = validator.predict_proba(zscores.loc[later_mask,candidates])[:,1]
        val['Rank'] = val.groupby('Season').Model_Score.rank(ascending=False,method='first').astype(int)
        check = evaluate_period(frame,val,labels,'pre-2010 time split')
        scored_hist = check.loc[check.Correct.notna()]
        if len(scored_hist):
            hist_val['top1_accuracy'] = float(scored_hist.Correct.astype(bool).mean())
            hist_val['top3_accuracy'] = float(scored_hist.Actual_In_Top3.astype(bool).mean())
        check.to_csv(args.output_dir/'historical_validation.csv', index=False)

    scored = results.loc[results.Correct.notna()]
    if len(scored):
        top1=float(scored.Correct.astype(bool).mean())
        top3=float(scored.Actual_In_Top3.astype(bool).mean())
        print(f'2010–2026 BACKTEST (post-hoc only): top-1 {top1:.1%} '
              f'({int(scored.Correct.astype(bool).sum())}/{len(scored)}) | top-3 {top3:.1%}.')
    else:
        print('Predictions generated; no actual champions available for scoring. '
              'Predictions remain valid as a retrospective model output.')
    optional_logistic_comparison(results, args.logistic_dir, args.output_dir)
    config = {
        'algorithm':'XGBClassifier (XGBoost gradient-boosted decision trees)',
        'hyperparameters':pipe.named_steps['classifier'].get_params(),
        'training_max_season':args.train_end,
        'training_min_season':int(frame.loc[train,'Season'].min()),
        'years_for_predictions':[args.predict_start,args.predict_end],
        'label_is_visible_to_model_in_2010_2026':False,
        'uses_same_season_regular_season_stats_for_predictions':True,
        'postseason_features_used':False,
        'number_of_training_seasons':int(frame.loc[train,'Season'].nunique()),
        'number_of_predicted_seasons':int(frame.loc[target,'Season'].nunique()),
        'features':candidates,
        'pre_2010_validation':hist_val,
        'note':'XGBoost binary classification scores are not calibrated NBA title probabilities. The top-ranked team in each season is the pick.',
    }
    (args.output_dir/'model_details.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(f'Saved predictions and diagnostics to: {args.output_dir.resolve()}')
    print(results[['Season','Predicted_Champion','Actual_Champion','Correct']].to_string(index=False))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError,FileNotFoundError) as exc:
        sys.exit(f'ERROR: {exc}')
