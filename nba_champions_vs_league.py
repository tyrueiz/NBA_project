#!/usr/bin/env python3
"""Compare each historical NBA/BAA champion to its own season's league.

Consumes local CSV exports from nba_unified_scraper_1947_2027.py (with or
without money scraping). NO web requests are made by this analysis script.

Outputs:
  champions_vs_league_all_stats.csv            one row/champion/year/metric
  champion_player_distribution_all_stats.csv   one row/player/champion/metric
  champion_distribution_summary.csv            player-level concentration
  team_metrics_all_seasons.csv                 useful intermediate table
  analysis_coverage_report.csv                 missing seasons and coverage
  player_name_collision_audit.csv              distinct players sharing a name

Counting team stats are sums of non-aggregate player-team season TOTALS,
divided by a team's regular-season games. League average is the mean of the
league's team-per-game rates (not a mean over players). Shooting percentages
are rebuilt from the relevant team numerator and denominator. Rates lacking
such components are labeled as minutes-weighted player estimates.

Run:
  python -m pip install pandas numpy matplotlib
  python nba_champions_vs_league.py --input-dir nba_updated
  python nba_champions_vs_league.py --input-dir nba_updated --plots --plot-years 2024 2025 2026

Years are season END years; 2024 means 2023-24. 2026-27 is omitted until
its champion is known. Missing stats in early eras remain missing.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Stats identifiers and metadata NOT meaningful as team performance metrics.
NON_STATS = {
    "Season", "seasonStartYear", "Year", "season", "Rk", "Player-additional",
    "ID", "PlayerID", "player_id", "bbref_id", "Data_Collected_UTC", "index",
}
PLAYER_ONLY_STATS = {"G"}  # Sum of players' games is NOT team games.
# Historical Basketball-Reference team abbreviation changes. Each is accepted
# only if the target franchise code really exists in THAT PARTICULAR season.
ALIASES = {
    "BKN": "BRK", "BRK": "BKN", "CHA": "CHO", "CHO": "CHA",
    "PHX": "PHO", "PHO": "PHX", "GS": "GSW",
    "NOH": "NOK", "NOK": "NOH", "NOL": "NOH",
}
RATE_COMPONENTS = {
    "FG%": ("FG", "FGA"), "3P%": ("3P", "3PA"),
    "2P%": ("2P", "2PA"), "FT%": ("FT", "FTA"),
}


def norm_header(col: Any) -> str:
    return str(col).strip()


def find_latest(folder: Path, kind: str) -> Path:
    patterns = {
        "stats": "NBA_stats_*.csv",
        "teams": "NBA_team_seasons_*.csv",
    }
    candidates = [p for p in folder.glob(patterns[kind])
                  if "with_playoffs" not in p.name and "salaries" not in p.name]
    if not candidates:
        raise FileNotFoundError(
            f"No {kind} export in {folder}. Supply --{kind} /path/to/file.csv "
            "or run the unified scraper first."
        )
    def score(p: Path):
        years = [int(x) for x in re.findall(r"(?:19|20)\d{2}", p.stem)]
        return (max(years) if years else 0, -min(years) if years else 0, p.stat().st_mtime)
    return max(candidates, key=score)


def truthy(col: pd.Series) -> pd.Series:
    return col.astype("string").str.strip().str.lower().isin(["true", "1", "yes"])


def disambiguate_same_name_players(stats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep distinct people with the same name on one team in the same season.

    Historical example: Washington's 1988-89 roster had two Charles Jones
    players (ages 31/C and 27/PF). Use a source player ID if present; otherwise
    use available age/position evidence. Ambiguous conflicts still raise rather
    than silently losing or double-counting rows.
    """
    key = ["Season", "Team", "Player"]
    collisions = stats.loc[stats.duplicated(key, keep=False)]
    audit = []
    if collisions.empty:
        return stats, pd.DataFrame(columns=[
            "Season", "Team", "Original_Player", "Disambiguated_Player",
            "Age", "Pos", "Identification_Method"])

    stats = stats.copy()
    for (season, team, name), group in collisions.groupby(key, sort=True, dropna=False):
        ages = (pd.to_numeric(group["Age"], errors="coerce") if "Age" in group
                else pd.Series(np.nan, index=group.index))
        positions = (group["Pos"].astype("string").str.strip().replace("", pd.NA)
                     if "Pos" in group else pd.Series(pd.NA, index=group.index, dtype="string"))
        labels = None
        method = None
        # When the source includes true person identifiers, prefer them.
        for id_col in ("bbref_id", "PlayerID", "player_id", "Player-additional"):
            if id_col in group:
                ids = group[id_col].astype("string").str.strip().replace("", pd.NA)
                if ids.notna().all() and ids.is_unique:
                    labels = [f"ID {val}" for val in ids]
                    method = id_col
                    break
        if labels is None:
            if ages.notna().all() and positions.notna().all():
                candidates = [f"{pos}, age {age:g}" for pos, age in zip(positions, ages)]
                if len(set(candidates)) == len(candidates):
                    labels, method = candidates, "position + age"
            if labels is None and ages.notna().all() and ages.is_unique:
                labels = [f"age {age:g}" for age in ages]
                method = "age"
            if labels is None and positions.notna().all() and positions.is_unique:
                labels = [f"position {pos}" for pos in positions]
                method = "position"
        if labels is None:
            sample_cols = [c for c in ("Season", "Team", "Player", "Age", "Pos", "G", "MP", "PTS")
                           if c in group.columns]
            raise ValueError(
                "Ambiguous same-name player records; cannot safely identify distinct "
                f"players for {season} {team} {name}. Do not drop either row blindly.\n"
                f"{group[sample_cols].to_string(index=False)}\n"
                "Provide distinct player IDs or resolve this collision in the CSV.")

        for (idx, row), label in zip(group.iterrows(), labels):
            distinct_name = f"{name} [{label}]"
            stats.at[idx, "Player"] = distinct_name
            audit.append({
                "Season": int(season), "Team": team, "Original_Player": name,
                "Disambiguated_Player": distinct_name,
                "Age": ages.loc[idx], "Pos": positions.loc[idx],
                "Identification_Method": method,
            })
    if stats.duplicated(key).any():
        raise ValueError("Player-name disambiguation left duplicate player-team-season keys")
    return stats, pd.DataFrame(audit)


def checked_files(stats_path: Path, teams_path: Path):
    stats = pd.read_csv(stats_path, low_memory=False)
    teams = pd.read_csv(teams_path, low_memory=False)
    stats.columns = [norm_header(c) for c in stats.columns]
    teams.columns = [norm_header(c) for c in teams.columns]
    expected_stats = {"Season", "Team", "Player"}
    expected_teams = {"Season", "Team", "Won_Title", "Wins", "Losses"}
    if not expected_stats.issubset(stats.columns):
        raise ValueError(f"Stats file missing {sorted(expected_stats - set(stats.columns))}")
    if not expected_teams.issubset(teams.columns):
        raise ValueError(f"Team file missing {sorted(expected_teams - set(teams.columns))}")
    for f in (stats, teams):
        f["Season"] = pd.to_numeric(f["Season"], errors="coerce")
        f.dropna(subset=["Season"], inplace=True)
        f["Season"] = f["Season"].astype(int)
        f["Team"] = f["Team"].astype(str).str.strip().str.upper()
    if teams.duplicated(["Season", "Team"]).any():
        bad = teams[teams.duplicated(["Season", "Team"], keep=False)].head(5)
        raise ValueError(f"Duplicate team-season keys in team results:\n{bad}")
    teams["Won_Title"] = truthy(teams["Won_Title"])
    teams["Wins"] = pd.to_numeric(teams["Wins"], errors="coerce")
    teams["Losses"] = pd.to_numeric(teams["Losses"], errors="coerce")
    teams["Team_Games"] = teams["Wins"] + teams["Losses"]
    # Optional team-season payroll comes from the FULL scraper; compare it with
    # same-season teams only (nominal USD; no inflation adjustment required).
    if "payroll" in teams:
        money = teams["payroll"].astype("string").str.replace(r"[$,]", "", regex=True)
        teams["Payroll"] = pd.to_numeric(money, errors="coerce")
    teams["Win_Percentage"] = teams["Wins"] / teams["Team_Games"].replace(0, np.nan)
    # Prefer the boolean title flag, not name matching or rank assumptions.
    completed = teams[teams.get("Season_Status", pd.Series("Completed", index=teams.index))
                      .astype("string").str.lower().ne("in progress")].copy()
    winner_counts = completed.loc[completed["Won_Title"]].groupby("Season").size()
    invalid = winner_counts[winner_counts.ne(1)]
    if not invalid.empty:
        raise ValueError(f"More than one champion in seasons {invalid.index.tolist()}")
    # Current season cannot be included until an actual winner is recorded.
    stats = stats.loc[~stats["Team"].str.fullmatch(r"TOT|[2-9]TM", case=False, na=False)].copy()
    stats["Player"] = stats["Player"].astype(str).str.replace("*", "", regex=False).str.strip()
    stats = stats.loc[stats["Player"].ne("") & stats["Player"].ne("nan")].copy()
    stats = stats.drop_duplicates()
    stats, name_collision_audit = disambiguate_same_name_players(stats)
    if not name_collision_audit.empty:
        print(f"Disambiguated {len(name_collision_audit)} same-name player rows "
              f"across {len(name_collision_audit.groupby(['Season', 'Team', 'Original_Player']))} "
              "player-name collisions; see player_name_collision_audit.csv")
    # Match aliases only when the actual team code exists in that season.
    known = set(zip(teams["Season"], teams["Team"]))
    def map_code(year: int, code: str) -> str | None:
        if (year, code) in known:
            return code
        substitute = ALIASES.get(code)
        if substitute and (year, substitute) in known:
            return substitute
        return None
    stats["Team_Matched"] = [map_code(int(y), c) for y, c in zip(stats["Season"], stats["Team"])]
    mismatches = stats[stats["Team_Matched"].isna()]
    if not mismatches.empty:
        examples = mismatches[["Season", "Team"]].drop_duplicates().head(12).values.tolist()
        print(f"Warning: {len(mismatches)} player rows have unmatched team-season codes; "
              f"excluded. Examples: {examples}", file=sys.stderr)
    stats = stats.dropna(subset=["Team_Matched"]).copy()
    stats["Team"] = stats["Team_Matched"]
    stats.drop(columns=["Team_Matched"], inplace=True)
    return stats, teams, name_collision_audit


def metric_columns(stats: pd.DataFrame):
    """Identify ALL numeric stats in the source (not identifiers or rankings)."""
    skip = NON_STATS | {"Age"} | {"G"}
    columns = []
    for c in stats.columns:
        if c in skip or c.startswith("Unnamed") or c.startswith("_"):
            continue
        if c in {"Team", "Player", "Pos", "Season_Status", "team_name"}:
            continue
        col = pd.to_numeric(stats[c], errors="coerce")
        if col.notna().any():
            stats[c] = col
            columns.append(c)
    for c in ("Age", "G"):
        if c in stats.columns:
            vals = pd.to_numeric(stats[c], errors="coerce")
            if vals.notna().any():
                stats[c] = vals
                columns.append(c)
    # Nonstat numeric fields if data was sourced from an enriched CSV.
    columns = [c for c in columns if c not in {"Won_Title", "Wins", "Losses",
                "Playoff_Round_Reached", "Made_Playoffs", "Win_Percentage",
                "Playoff_Stage_Of_Season", "Total_Playoff_Stages_That_Season",
                "Playoff_Series_Won", "payroll", "salary"}]
    return columns


def metric_kind(stat: str) -> str:
    if stat == "G":
        return "player_only"
    if stat == "Age":
        return "minute_weighted_player_mean"
    if ("%" in stat or stat.lower().endswith(("_pct", "_percentage"))
            or stat in {"PER", "BPM", "OBPM", "DBPM", "WS/48", "ORtg", "DRtg",
                        "USG", "USG%", "TS%"}):
        return "shooting_rate" if stat in RATE_COMPONENTS or stat in {"eFG%", "TS%"} else "minute_weighted_player_mean"
    return "per_team_game"


def weighted_rate(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return np.nan
    return float(np.average(values[valid].astype(float), weights=weights[valid].astype(float)))


def aggregate_team(group: pd.DataFrame, stats: list[str], missing_limit: float) -> dict:
    games = float(group["Team_Games"].iloc[0])
    result = {"Season": int(group["Season"].iloc[0]),
              "Team": group["Team"].iloc[0],
              "team_name": group["team_name"].iloc[0],
              "Team_Games": games,
              "Number_of_Players": len(group),
              "Wins": group["Wins"].iloc[0],
              "Losses": group["Losses"].iloc[0],
              "Win_Percentage": group["Win_Percentage"].iloc[0],
              "Won_Title": bool(group["Won_Title"].iloc[0])}
    if "Payroll" in group:
        result["Payroll"] = group["Payroll"].iloc[0]
    raw_totals = {}
    for stat in stats:
        if stat in PLAYER_ONLY_STATS:
            continue
        series = group[stat]
        coverage = series.notna().mean()
        if coverage < 1 - missing_limit or not games or np.isnan(games):
            raw_totals[stat] = np.nan
        elif metric_kind(stat) == "per_team_game":
            raw_totals[stat] = series.sum(min_count=1)
        elif metric_kind(stat) == "minute_weighted_player_mean":
            if "MP" in group:
                raw_totals[stat] = weighted_rate(series, group["MP"])
            else:
                raw_totals[stat] = np.nan
        else:
            raw_totals[stat] = np.nan
    # Rebuild team shooting percentages from player/team counting totals,
    # not the (incorrect) arithmetic mean of player shooting percentages.
    for rate, (made, attempts) in RATE_COMPONENTS.items():
        if rate not in stats:
            continue
        numerator = raw_totals.get(made)
        denominator = raw_totals.get(attempts)
        raw_totals[rate] = (float(numerator) / float(denominator)
                            if pd.notna(numerator) and pd.notna(denominator)
                            and denominator > 0 else np.nan)
    if "eFG%" in stats:
        fg, threes, attempts = (raw_totals.get(x) for x in ("FG", "3P", "FGA"))
        raw_totals["eFG%"] = ((fg + 0.5 * threes) / attempts
                              if all(pd.notna(x) for x in (fg, threes, attempts))
                              and attempts > 0 else np.nan)
    if "TS%" in stats:
        pts, attempts, ft = (raw_totals.get(x) for x in ("PTS", "FGA", "FTA"))
        denom = 2 * (attempts + 0.44 * ft) if all(pd.notna(x) for x in (attempts, ft)) else 0
        raw_totals["TS%"] = pts / denom if pd.notna(pts) and denom > 0 else np.nan
    for stat in stats:
        if stat in PLAYER_ONLY_STATS:
            continue
        total = raw_totals.get(stat, np.nan)
        result[stat] = (total / games if metric_kind(stat) == "per_team_game" and pd.notna(total)
                        else total)
    return result


def make_team_metrics(stats: pd.DataFrame, teams: pd.DataFrame, metrics: list[str], missing_limit: float):
    core_columns = ["Season", "Team", "team_name", "Team_Games", "Wins", "Losses",
                    "Win_Percentage", "Won_Title"]
    if "Payroll" in teams:
        core_columns.append("Payroll")
    core = teams[core_columns].copy()
    subset = stats.merge(core, on=["Season", "Team"], how="inner", validate="many_to_one")
    calculated = [aggregate_team(g, metrics, missing_limit)
                  for _, g in subset.groupby(["Season", "Team"], sort=True)]
    table = pd.DataFrame(calculated)
    if table.empty:
        raise ValueError("No player-team records match the team result file; check year/team codes.")
    return subset, table


def concentration(values: pd.Series) -> dict:
    vals = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if vals.empty:
        return {"Top_Player_Share": np.nan, "Top_3_Share": np.nan,
                "HHI": np.nan, "Gini": np.nan}
    vals = vals.clip(lower=0)
    total = vals.sum()
    if total <= 0:
        return {"Top_Player_Share": np.nan, "Top_3_Share": np.nan,
                "HHI": np.nan, "Gini": np.nan}
    shares = vals / total
    ordered = np.sort(vals.to_numpy())
    n = len(ordered)
    gini = float(2 * (np.arange(1, n + 1) * ordered).sum() / (n * total) - (n + 1) / n)
    return {"Top_Player_Share": float(shares.max()),
            "Top_3_Share": float(shares.nlargest(3).sum()),
            "HHI": float(np.square(shares).sum()), "Gini": gini}


def make_outputs(player_team: pd.DataFrame, metrics_table: pd.DataFrame,
                 original_teams: pd.DataFrame, stats: list[str],
                 min_team_coverage: float):
    metric_names = [c for c in stats if c not in PLAYER_ONLY_STATS] + ["Wins", "Losses", "Win_Percentage"]
    if "Payroll" in metrics_table and metrics_table["Payroll"].notna().any():
        metric_names.append("Payroll")
    champions = original_teams.loc[original_teams["Won_Title"]].copy()
    compare = []
    player_details = []
    summary = []
    coverage = []
    roster_lookup = {(int(y), t): g for (y, t), g in player_team.groupby(["Season", "Team"])}
    for season, league_results in original_teams.groupby("Season"):
        winners = champions.loc[champions["Season"].eq(season)]
        if len(winners) != 1:
            coverage.append({"Season": season, "Status": "SKIPPED: champion unknown",
                             "Number_of_League_Teams": len(league_results), "Teams_With_Stats": 0})
            continue
        winner = winners.iloc[0]
        team = winner["Team"]
        season_metrics = metrics_table.loc[metrics_table["Season"].eq(season)]
        champ_metrics = season_metrics.loc[season_metrics["Team"].eq(team)]
        valid_team_count = len(season_metrics)
        base_coverage = valid_team_count / len(league_results)
        if champ_metrics.empty or base_coverage < min_team_coverage:
            coverage.append({"Season": season, "Status": "SKIPPED: missing champion or insufficient team coverage",
                             "Number_of_League_Teams": len(league_results),
                             "Teams_With_Stats": valid_team_count,
                             "Team_Coverage": base_coverage})
            continue
        champ = champ_metrics.iloc[0]
        key = (int(season), team)
        players = roster_lookup[key]
        coverage.append({"Season": season, "Status": "Analyzed",
                         "Number_of_League_Teams": len(league_results),
                         "Teams_With_Stats": valid_team_count,
                         "Team_Coverage": base_coverage,
                         "Champion": champ["team_name"],
                         "Champion_Player_Rows": len(players)})
        for stat in metric_names:
            if stat not in season_metrics:
                continue
            available = season_metrics[["Team", stat]].dropna()
            if len(available) / len(league_results) < min_team_coverage:
                continue
            if pd.isna(champ[stat]):
                continue
            cv = float(champ[stat])
            lm = float(available[stat].mean())
            diff = cv - lm
            compare.append({
                "Season": int(season), "Champion_Code": team,
                "Champion": champ["team_name"], "Stat": stat,
                "Aggregation": ("regular_season_team_stat" if stat in {"Wins", "Losses", "Win_Percentage"}
                                else "team_season_nominal_USD" if stat == "Payroll"
                                else metric_kind(stat)),
                "Champion_Value": cv, "League_Average": lm,
                "Difference": diff,
                "Percent_Difference": (100 * diff / abs(lm) if lm != 0 else np.nan),
                "Champion_Percentile": 100 * float((available[stat] <= cv).mean()),
                "League_Teams_With_Stat": len(available),
                "League_Teams_Total": len(league_results),
                "League_StdDev": float(available[stat].std(ddof=0)),
            })
        # All player-stat columns, including G and Age, are distributed on
        # the champion roster even where not meaningful as team aggregates.
        for stat in stats:
            vals = players[list(dict.fromkeys(["Player", "Pos", "G", "MP", stat]))].copy() if all(
                c in players.columns for c in ("Pos", "G", "MP")) else None
            if vals is None:
                vals = players[["Player", stat]].copy()
                if "G" not in vals:
                    vals["G"] = pd.NA
                if "MP" not in vals:
                    vals["MP"] = pd.NA
                if "Pos" not in vals:
                    vals["Pos"] = pd.NA
            vals = vals.loc[vals[stat].notna()]
            if vals.empty:
                continue
            is_count = metric_kind(stat) == "per_team_game"
            roster_total = vals[stat].sum() if is_count else np.nan
            cc = concentration(vals[stat]) if is_count else {
                "Top_Player_Share": np.nan, "Top_3_Share": np.nan,
                "HHI": np.nan, "Gini": np.nan}
            best_idx = vals[stat].idxmax()
            weighted_mean = (weighted_rate(vals[stat], vals["MP"])
                             if not is_count else np.nan)
            summary.append({
                "Season": season, "Champion": champ["team_name"], "Team": team,
                "Stat": stat, "Metric_Type": metric_kind(stat),
                "Players_With_Stat": len(vals), "Min": vals[stat].min(),
                "P25": vals[stat].quantile(.25), "Median": vals[stat].median(),
                "Mean_Player_Value": vals[stat].mean(),
                "Minutes_Weighted_Mean": weighted_mean,
                "P75": vals[stat].quantile(.75), "Max": vals[stat].max(),
                "StdDev": vals[stat].std(ddof=0),
                "Largest_Value_Player": vals.loc[best_idx, "Player"],
                "Largest_Player_Value": vals.loc[best_idx, stat],
                "Roster_Total": roster_total, **cc,
            })
            for _, row in vals.iterrows():
                v = float(row[stat])
                g = pd.to_numeric(row["G"], errors="coerce")
                player_details.append({
                    "Season": int(season), "Champion": champ["team_name"],
                    "Team": team, "Player": row["Player"],
                    "Pos": row["Pos"], "Games_Played": g,
                    "Minutes_Played": row["MP"],
                    "Stat": stat, "Metric_Type": metric_kind(stat),
                    "Player_Season_Value": v,
                    "Player_Per_Game_Played": (v / g if is_count and pd.notna(g) and g > 0 else np.nan),
                    "Team_Total": roster_total if is_count else np.nan,
                    "Share_Of_Team_Total": (v / roster_total if is_count and roster_total > 0 else np.nan),
                    "Player_Percentile_Within_Team": 100 * float((vals[stat] <= v).mean()),
                })
    return (pd.DataFrame(compare), pd.DataFrame(player_details),
            pd.DataFrame(summary), pd.DataFrame(coverage))


def make_plots(comparison: pd.DataFrame, details: pd.DataFrame, folder: Path,
               years: list[int] | None, metrics: list[str], top_n: int):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Plots requested but matplotlib unavailable: pip install matplotlib", file=sys.stderr)
        return
    if comparison.empty or details.empty:
        return
    years = years if years else [int(comparison["Season"].max())]
    folder.mkdir(parents=True, exist_ok=True)
    for season in years:
        cmp = comparison.loc[comparison["Season"].eq(season)]
        det = details.loc[details["Season"].eq(season)]
        if cmp.empty:
            print(f"No champion analysis for {season}; skipping plots", file=sys.stderr)
            continue
        winner = cmp["Champion"].iloc[0]
        selected = cmp[cmp["Stat"].isin(metrics)].copy()
        if not selected.empty:
            selected = selected.loc[selected["Percent_Difference"].notna()].copy()
            if not selected.empty:
                fig, ax = plt.subplots(figsize=(9, max(4, len(selected) * .5)))
                ax.barh(selected["Stat"], selected["Percent_Difference"])
                ax.axvline(0, linewidth=1)
                ax.set_title(f"{season} {winner}: difference from league team average")
                ax.set_xlabel("Difference from that season's average team (%)")
                fig.tight_layout()
                fig.savefig(folder / f"{season}_champion_vs_league.png", dpi=160)
                plt.close(fig)
        for stat in metrics:
            vals = det.loc[det["Stat"].eq(stat)].nlargest(top_n, "Player_Season_Value")
            if vals.empty:
                continue
            vals = vals.sort_values("Player_Season_Value")
            fig, ax = plt.subplots(figsize=(9, max(4, .35 * len(vals) + 1)))
            ax.barh(vals["Player"], vals["Player_Season_Value"])
            is_pct = "%" in stat or stat.lower().endswith("_pct")
            ax.set_xlabel(f"{stat} — player season {'rate (0–1)' if is_pct else 'total / value'}")
            ax.set_title(f"{season} {winner}: top {len(vals)} players in {stat}")
            fig.tight_layout()
            name = re.sub(r"[^a-zA-Z0-9_-]", "_", stat)
            fig.savefig(folder / f"{season}_{name}_player_distribution.png", dpi=160)
            plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", default="nba_updated", help="Unified scraper output directory")
    p.add_argument("--stats", help="Override player stats CSV path")
    p.add_argument("--teams", help="Override NBA_team_seasons CSV path")
    p.add_argument("--outdir", default="nba_champions_analysis", help="Output folder")
    p.add_argument("--min-team-coverage", type=float, default=.90,
                   help="Min share of teams with player data to benchmark a season/stat [0..1]")
    p.add_argument("--max-missing-player-fraction", type=float, default=.05,
                   help="Max missing-player fraction for a team metric [0..1]")
    p.add_argument("--plots", action="store_true", help="Create optional PNGs (default: latest completed champion)")
    p.add_argument("--plot-years", type=int, nargs="*", help="Plot these champion season-end years")
    p.add_argument("--plot-all-years", action="store_true",
                   help="Plot every analyzed champion (creates many PNG images)")
    p.add_argument("--plot-stats", nargs="+", default=["PTS", "TRB", "AST", "FG%", "3P%"],
                   help="Which stats to visualize (all stats are ALWAYS included in CSVs)")
    p.add_argument("--top-players", type=int, default=15, help="Max players per PNG")
    args = p.parse_args(argv)
    for value in (args.min_team_coverage, args.max_missing_player_fraction):
        if not 0 <= value <= 1:
            p.error("Coverage/fraction must be between 0 and 1")
    input_dir = Path(args.input_dir)
    stats_path = Path(args.stats) if args.stats else find_latest(input_dir, "stats")
    teams_path = Path(args.teams) if args.teams else find_latest(input_dir, "teams")
    folder = Path(args.outdir)
    folder.mkdir(parents=True, exist_ok=True)
    print(f"Reading player stats: {stats_path}")
    print(f"Reading team/champions: {teams_path}")
    stats, teams, name_collision_audit = checked_files(stats_path, teams_path)
    cols = metric_columns(stats)
    print(f"Identified {len(cols)} player statistics: {', '.join(cols)}")
    if not cols:
        raise ValueError("No numeric player stats identified")
    player_team, team_metrics = make_team_metrics(stats, teams, cols,
                                                  args.max_missing_player_fraction)
    cmp, dist, dist_summary, coverage = make_outputs(
        player_team, team_metrics, teams, cols, args.min_team_coverage)
    files = {
        "team_metrics_all_seasons.csv": team_metrics,
        "champions_vs_league_all_stats.csv": cmp,
        "champion_player_distribution_all_stats.csv": dist,
        "champion_distribution_summary.csv": dist_summary,
        "analysis_coverage_report.csv": coverage,
        "player_name_collision_audit.csv": name_collision_audit,
    }
    for filename, frame in files.items():
        path = folder / filename
        frame.to_csv(path, index=False, float_format="%.8g")
        print(f"Saved {path}: {len(frame):,} rows")
    print(f"Analyzed {cmp['Season'].nunique() if not cmp.empty else 0} champions "
          "(seasons without complete results/data reported in coverage file).")
    if args.plots:
        plotted_years = (sorted(int(y) for y in cmp["Season"].unique())
                         if args.plot_all_years and not cmp.empty else args.plot_years)
        make_plots(cmp, dist, folder / "plots", plotted_years, args.plot_stats, args.top_players)
    return 0 if not cmp.empty else 1


if __name__ == "__main__":
    sys.exit(main())
