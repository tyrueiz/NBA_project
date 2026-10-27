#!/usr/bin/env python3
"""NBA scraper: player stats, player salaries, team payroll, and NBA standings/playoffs.

Adapted from two scripts prepared in this conversation and from the workflow in
https://github.com/logan-lauton/nba_webscrape . This is not an official NBA API.

Every year option is a *season-ending year*: 2024 = the 2023-24 NBA season.
Defaults: playoff histories, player stats, salaries, and payrolls are all
requested for season-ending years 1947-2027. The 2026-27 season
is IN PROGRESS: live team standings and player statistics may change, and the
playoff outcomes are deliberately left blank. Reliable public salary/payroll
records are NOT available for every season. Money requests before the 1984-85
season are skipped by default (reported as no_verified_source), and all other
missing data are left blank. Never interpret a missing salary as $0.

Basketball-Reference says its historical salaries start in 1984-85 and are
incomplete. HoopsHype coverage may be more limited. Use --earliest-money-year
to adjust when money requests should begin; this is NOT an evidence guarantee.

Sources: Basketball-Reference (stats, standings, postseason), HoopsHype
(player salaries and team payroll). Follow site terms and robots instructions;
stop on 403, 429, access challenges. No access-control evasion.

Install: python -m pip install pandas requests beautifulsoup4 lxml
Example: python nba_unified_scraper_1947_2027.py --kind all
"""

import argparse
import re
import sys
import time
import unicodedata
from io import StringIO
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

BBREF = "https://www.basketball-reference.com"
HOOPSHYPE = "https://hoopshype.com"
SERIES_SCORE_RE = re.compile(r"\(\s*\d+\s*[-–]\s*\d+\s*\)")
TEAM_LINK_RE = re.compile(r"^/teams/([A-Z]{3})/(\d{4})\.html")
ROUND_RE = re.compile(
    r"(?:Eastern|Western|Central)?\s*(?:Conference|Division)?\s*"
    r"(?:First Round|Second Round|Quarterfinals|Semifinals|Semi-Finals|Finals|Tiebreaker|Tie Breaker)",
    flags=re.I,
)
# Allow Basketball-Reference and HoopsHype to publish different abbreviations
# for the SAME franchise in the SAME season. Historical franchise changes are
# handled by using the historical full name instead of assuming a modern code.
DEFAULT_EARLIEST_MONEY_YEAR = 1985  # Historical salary evidence begins around 1984-85.
CURRENT_SEASON_END = 2027
LAST_COMPLETED_SEASON_END = 2026

CODE_ALIASES = {
    "BKN": "BRK", "BRK": "BRK", "CHA": "CHO", "CHO": "CHO",
    "PHX": "PHO", "PHO": "PHO", "GSW": "GSW", "GS": "GSW",
}


def norm(value):
    return re.sub(r"\s+", " ", str(value)).strip().rstrip("*").strip()


def name_key(value):
    value = unicodedata.normalize("NFKD", norm(value)).casefold()
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", value)


def http_get(session, url):
    domain = urlparse(url).netloc
    if domain not in {"www.basketball-reference.com", "hoopshype.com", "www.hoopshype.com"}:
        raise ValueError(f"Unexpected source host: {domain}")
    response = session.get(url, timeout=30)
    if response.status_code in (403, 429):
        raise PermissionError(f"HTTP {response.status_code}: {url}; do not bypass restrictions")
    response.raise_for_status()
    if "captcha" in response.text[:3000].lower():
        raise PermissionError(f"Access challenge at {url}")
    return response.text


def table_scopes(html):
    page = BeautifulSoup(html, "lxml")
    yield page
    for comment in page.find_all(string=lambda x: isinstance(x, Comment)):
        if "<table" in str(comment):
            yield BeautifulSoup(str(comment), "lxml")


def dataframe_tables(html):
    for scope in table_scopes(html):
        # Avoid yielding table descendants twice: each scope represents one
        # real-page or HTML-comment fragment. Duplicate rows are not expected.
        for tag in scope.select("table"):
            try:
                for df in pd.read_html(StringIO(str(tag)), displayed_only=False):
                    df = df.copy()
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [next((str(x).strip() for x in reversed(parts)
                                            if not str(x).startswith("Unnamed")), "Unnamed")
                                      for parts in df.columns.to_flat_index()]
                    else:
                        df.columns = [str(c).strip() for c in df.columns]
                    yield df
            except (ValueError, ImportError):
                continue


def season_urls(year):
    league = "BAA" if year < 1950 else "NBA"
    return (
        f"{BBREF}/leagues/{league}_{year}_standings.html",
        f"{BBREF}/playoffs/{league}_{year}.html",
    )


def get_stats(session, year):
    league = "BAA" if year < 1950 else "NBA"
    url = f"{BBREF}/leagues/{league}_{year}_totals.html"
    for frame in dataframe_tables(http_get(session, url)):
        cols = set(frame.columns)
        if {"Player", "G", "PTS"}.issubset(cols) and ("Team" in cols or "Tm" in cols):
            frame = frame.rename(columns={"Tm": "Team"}).copy()
            frame = frame.loc[frame["Player"].astype(str).ne("Player")].copy()
            frame["Player"] = frame["Player"].astype(str).str.replace("*", "", regex=False).str.strip()
            frame = frame.loc[frame["Player"].ne("") & frame["Player"].ne("nan")]
            frame = frame.drop(columns=["Rk"], errors="ignore")
            frame.insert(0, "Season", year)
            frame.insert(1, "seasonStartYear", year - 1)
            if not frame.empty:
                return frame.reset_index(drop=True)
    raise ValueError(f"Player totals table not identified: {url}")


def salary_year_col(columns, start):
    short_end = str((start + 1) % 100).zfill(2)
    label = re.compile(rf"^{start}\s*[/\-]\s*(?:{short_end}|{start + 1})$")
    return next((c for c in columns if label.match(str(c).strip())), None)


def parse_dollars(value):
    if pd.isna(value):
        return pd.NA
    matched = re.search(r"\$?\s*([0-9][0-9,]*)", str(value))
    return int(matched.group(1).replace(",", "")) if matched else pd.NA


def get_money(session, year, kind):
    start = year - 1
    if kind == "salaries":
        url = f"{HOOPSHYPE}/salaries/players/{start}-{year}/"
        label, out_label, number = "Player", "playerName", "salary"
    elif kind == "payroll":
        url = f"{HOOPSHYPE}/salaries/{start}-{year}/"
        label, out_label, number = "Team", "team", "payroll"
    else:
        raise ValueError(f"Unknown money dataset: {kind}")
    for frame in dataframe_tables(http_get(session, url)):
        names = [c for c in frame if str(c).casefold() == label.casefold()]
        target = salary_year_col(frame.columns, start)
        if not names or target is None:
            continue
        frame = frame.loc[:, [names[0], target]].copy()
        frame.columns = [out_label, number]
        frame[out_label] = frame[out_label].astype(str).str.strip()
        frame[number] = frame[number].map(parse_dollars).astype("Int64")
        frame = frame.loc[frame[out_label].ne("") & frame[out_label].ne(label)].dropna(subset=[number])
        frame.insert(0, "Season", year)
        frame.insert(1, "seasonStartYear", year - 1)
        if not frame.empty:
            return frame.reset_index(drop=True)
    raise ValueError(f"Could not find {label} / {start}-{year} salary table: {url}")


def standings_teams(html, year):
    """Return historical teams with BBRef code, name, wins and losses."""
    teams = {}
    for scope in table_scopes(html):
        for tr in scope.select("tr"):
            w = tr.select_one('[data-stat="wins"]')
            l = tr.select_one('[data-stat="losses"]')
            if w is None or l is None:
                continue
            for a in tr.find_all("a", href=True):
                m = TEAM_LINK_RE.match(a["href"])
                if not m or int(m.group(2)) != year:
                    continue
                try:
                    wins = int(norm(w.get_text(" ", strip=True)))
                    losses = int(norm(l.get_text(" ", strip=True)))
                except ValueError:
                    continue
                teams[m.group(1)] = {
                    "team_name": norm(a.get_text(" ", strip=True)),
                    "Wins": wins, "Losses": losses,
                }
                break
    if len(teams) < (20 if year >= 1980 else 6):
        raise ValueError(f"{year} standings: found only {len(teams)} teams")
    return teams


def parse_round(label):
    low = label.lower()
    if "tiebreak" in low or "tie break" in low:
        return "Tiebreaker", 0
    if "quarterfinal" in low or "first round" in low:
        return "First Round", 1
    if "second round" in low or "semi" in low:
        return "Semifinals", 2
    if "final" in low and ("conference" in low or "division" in low):
        return "Conference Finals", 3
    if "final" in low:
        return "Finals", 4
    raise ValueError(f"Unrecognized round: {label}")


def playoff_series(html, teams):
    # Bracket format and names vary by year: match only official team names for
    # that season; this deliberately fails if the site no longer uses the same
    # series-summary structure.
    to_code = {norm(v["team_name"]).casefold(): k for k, v in teams.items()}
    choices = "|".join(sorted((re.escape(x) for x in to_code), key=len, reverse=True))
    matchup = re.compile(rf"(?P<winner>{choices})\s+over\s+(?P<loser>{choices})\s*"
                         rf"(?P<score>\(\s*\d+\s*[-–]\s*\d+\s*\))", re.I)
    series = {}
    for scope in table_scopes(html):
        for tr in scope.select("tr"):
            text = norm(tr.get_text(" ", strip=True))
            if " over " not in text or not SERIES_SCORE_RE.search(text):
                continue
            m = matchup.search(text)
            if not m:
                continue
            before = text[:m.start()]
            labels = list(ROUND_RE.finditer(before))
            if not labels:
                continue
            raw_label = labels[-1].group(0).strip()
            round_name, rank = parse_round(raw_label)
            if rank == 0:
                continue
            winner = to_code[norm(m.group("winner")).casefold()]
            loser = to_code[norm(m.group("loser")).casefold()]
            series[(rank, winner, loser)] = {
                "rank": rank, "round": round_name, "source_round": raw_label,
                "winner": winner, "loser": loser,
            }
    return list(series.values())


def parse_playoff_year(year, standings_html, playoff_html):
    teams = standings_teams(standings_html, year)
    series = playoff_series(playoff_html, teams)
    if not series:
        raise ValueError(f"{year}: no playoff series found")
    if year >= 1984:
        counts = {r: sum(s["rank"] == r for s in series) for r in range(1, 5)}
        if counts != {1: 8, 2: 4, 3: 2, 4: 1}:
            raise ValueError(f"{year}: incomplete playoff bracket: {counts}")
    finals = [s for s in series if s["rank"] == 4]
    if len(finals) != 1:
        raise ValueError(f"{year}: expected one Finals series, found {len(finals)}")
    champion = finals[0]["winner"]
    eliminated = {}
    for s in series:
        if s["loser"] in eliminated:
            raise ValueError(f"{year}: duplicate eliminated team: {s['loser']}")
        eliminated[s["loser"]] = s
    entrants = {s["winner"] for s in series} | set(eliminated)
    if champion in eliminated or not entrants.issubset(teams):
        raise ValueError(f"{year}: inconsistent postseason winners/standings")
    if year >= 1984 and len(entrants) != 16:
        raise ValueError(f"{year}: expected 16 playoff entrants, found {len(entrants)}")
    stages = sorted({s["rank"] for s in series})
    chronological_stage = {x: i + 1 for i, x in enumerate(stages)}
    max_stages = len(stages)
    exit_names = {1: "First Round Exit", 2: "Second Round Exit",
                  3: "Conference Finals Exit", 4: "Finalist"}
    stand_url, post_url = season_urls(year)
    rows = []
    for code, team in sorted(teams.items()):
        loss = eliminated.get(code)
        if code == champion:
            status, raw_round, reached, stage = "Champion", "Finals", 4, max_stages
        elif loss:
            raw_round = loss["source_round"]
            reached = loss["rank"]
            stage = chronological_stage[reached]
            status = (exit_names[reached] if year >= 1984 or reached == 4
                      else raw_round + " Exit")
        else:
            status, raw_round, reached, stage = "Missed Playoffs", "", 0, 0
        rows.append({
            "Season": year,
            "seasonStartYear": year - 1,
            "Season_Status": "Completed",
            "Team": code,
            "team_name": team["team_name"],
            "Wins": team["Wins"],
            "Losses": team["Losses"],
            "Win_Percentage": (team["Wins"] / (team["Wins"] + team["Losses"])
                               if team["Wins"] + team["Losses"] else pd.NA),
            "Playoff_Result": status,
            "Playoff_Round": raw_round,
            "Playoff_Round_Reached": reached,
            "Playoff_Stage_Of_Season": stage,
            "Total_Playoff_Stages_That_Season": max_stages,
            "Made_Playoffs": code in entrants,
            "Won_Title": code == champion,
            "Playoff_Series_Won": sum(s["winner"] == code for s in series),
            "source_standings": stand_url,
            "source_playoffs": post_url,
        })
    result = pd.DataFrame(rows)
    if (result.duplicated(["Season", "Team"]).any() or
            result["Won_Title"].sum() != 1 or
            result["Playoff_Result"].eq("Finalist").sum() != 1):
        raise ValueError(f"{year}: result validity check failed")
    return result


def get_current_standings(session, year):
    """Get current standings only. Never infer playoffs from an unfinished season.

    The historical 'playoffs' collection represents a team-season outcome table;
    for the active season it is a *standings snapshot*. All postseason outcomes
    remain missing instead of being incorrectly labeled Missed Playoffs.
    """
    if year != CURRENT_SEASON_END:
        raise ValueError(f"Only {CURRENT_SEASON_END} is configured as an active season")
    stand_url, _ = season_urls(year)
    teams = standings_teams(http_get(session, stand_url), year)
    if len(teams) != 30:
        raise ValueError(f"{year}: expected all 30 NBA teams, found {len(teams)}")
    rows = []
    for code, team in sorted(teams.items()):
        wins, losses = team["Wins"], team["Losses"]
        rows.append({
            "Season": year,
            "seasonStartYear": year - 1,
            "Season_Status": "In Progress",
            "Team": code,
            "team_name": team["team_name"],
            "Wins": wins,
            "Losses": losses,
            "Win_Percentage": wins / (wins + losses) if wins + losses else pd.NA,
            "Playoff_Result": pd.NA,
            "Playoff_Round": pd.NA,
            "Playoff_Round_Reached": pd.NA,
            "Playoff_Stage_Of_Season": pd.NA,
            "Total_Playoff_Stages_That_Season": pd.NA,
            "Made_Playoffs": pd.NA,
            "Won_Title": pd.NA,
            "Playoff_Series_Won": pd.NA,
            "source_standings": stand_url,
            "source_playoffs": pd.NA,
        })
    return pd.DataFrame(rows)


def get_playoffs(session, year, delay):
    standings_url, playoff_url = season_urls(year)
    standings = http_get(session, standings_url)
    time.sleep(delay)
    playoffs = http_get(session, playoff_url)
    return parse_playoff_year(year, standings, playoffs)


def output_filename(kind, year):
    return f"{kind}_{year}.csv"


def valid_cached(frame, year, kind):
    if frame.empty or "Season" not in frame or not frame["Season"].eq(year).all():
        return False
    if kind == "playoffs":
        if not {"Playoff_Result", "Team", "Wins", "Losses"}.issubset(frame.columns):
            return False
        if frame.duplicated(["Season", "Team"]).any():
            return False
        if year == CURRENT_SEASON_END:
            return (len(frame) == 30
                    and "Season_Status" in frame
                    and frame["Season_Status"].eq("In Progress").all()
                    and frame["Playoff_Result"].isna().all()
                    and frame["Won_Title"].isna().all())
        return (len(frame) >= 6 and "Won_Title" in frame
                and frame["Won_Title"].astype(str).str.lower().eq("true").sum() == 1
                and frame["Playoff_Result"].eq("Finalist").sum() == 1)
    return True


def prepare_season_frame(frame, year, collected_at=None):
    """Add consistent completed/live flag and snapshot timestamp to source data."""
    frame = frame.copy()
    frame["Season_Status"] = ("In Progress" if year == CURRENT_SEASON_END
                              else "Completed")
    if "Data_Collected_UTC" not in frame.columns:
        frame["Data_Collected_UTC"] = collected_at if collected_at else pd.NA
    return frame


def safe_collect(session, kind, years, outdir, delay, refresh, statuses, earliest_money_year=1985):
    """Save one complete season at a time; source access denial stops this kind."""
    saved = []
    for year in years:
        path = outdir / output_filename(kind, year)
        # Active-season snapshots must be refreshed on each run.
        if year != CURRENT_SEASON_END and path.exists() and not refresh:
            try:
                cached = pd.read_csv(path)
                if valid_cached(cached, year, kind):
                    statuses.append({"kind": kind, "Season": year, "status": "cached", "detail": ""})
                    saved.append(path)
                    print(f"Using cache: {path.name}", flush=True)
                    continue
            except (ValueError, OSError, KeyError):
                pass
        # Systematic player salary / payroll records are not established for
        # the 1947-1984 period. Preserve the requested years in the status
        # report, rather than invent dollar figures or hitting dozens of
        # predictable missing URLs.
        if kind in {"salaries", "payroll"} and year < earliest_money_year:
            statuses.append({
                "kind": kind, "Season": year, "status": "no_verified_source",
                "detail": ("Historical money source not verified for this year; "
                           "no salary/payroll value is inferred"),
            })
            continue
        print(f"Fetching {kind}: {year - 1}-{year} ...", flush=True)
        try:
            if kind == "playoffs":
                data = (get_current_standings(session, year) if year == CURRENT_SEASON_END
                        else get_playoffs(session, year, delay))
            elif kind == "stats":
                data = get_stats(session, year)
            else:
                data = get_money(session, year, kind)
            data = prepare_season_frame(data, year,
                                        datetime.now(timezone.utc).isoformat(timespec="seconds"))
            if not valid_cached(data, year, kind):
                raise ValueError(f"{kind} {year}: output failed validation")
            data.to_csv(path, index=False)
            saved.append(path)
            statuses.append({"kind": kind, "Season": year, "status": "ok", "detail": ""})
            print(f"  Saved {len(data):,} rows -> {path.name}", flush=True)
            time.sleep(delay)
        except PermissionError as ex:
            statuses.append({"kind": kind, "Season": year, "status": "blocked", "detail": str(ex)})
            print(f"  Access restricted; stopping {kind}: {ex}", file=sys.stderr)
            # Stop fetching this source; do not repeatedly try successive years.
            for skipped in range(year + 1, years.stop):
                statuses.append({"kind": kind, "Season": skipped,
                                 "status": "not_attempted", "detail": "Stopped after source access restriction"})
            break
        except requests.HTTPError as ex:
            code = ex.response.status_code if ex.response is not None else None
            status = "not_found" if code == 404 else "failed"
            statuses.append({"kind": kind, "Season": year, "status": status, "detail": str(ex)})
            print(f"  Unable to retrieve {kind} {year}: {ex}", file=sys.stderr)
            time.sleep(delay)
        except (requests.RequestException, ValueError, KeyError, TypeError) as ex:
            statuses.append({"kind": kind, "Season": year, "status": "failed", "detail": str(ex)})
            print(f"  Unable to verify {kind} {year}: {ex}", file=sys.stderr)
            time.sleep(delay)
    if saved:
        total = pd.concat((prepare_season_frame(pd.read_csv(p),
                                              int(p.stem.rsplit("_", 1)[-1]))
                           for p in saved), ignore_index=True)
        merged_path = outdir / f"NBA_{kind}_{years.start}_{years.stop - 1}.csv"
        total.to_csv(merged_path, index=False)
        print(f"Collected {len(saved)}/{len(years)} seasons => {merged_path.name}")
        return total
    return None


def match_teams(frame, outcomes, source_col):
    """Join using SAME-SEASON official team name or BBRef ID (never cross-era).

    The first attempt uses names, then BBRef's source code, then a narrowly
    scoped alias table. Multi-team aggregate codes (TOT, 2TM, etc.) stay NA.
    """
    other = frame.copy()
    ref = outcomes.copy()
    other["_team_match"] = other[source_col].map(name_key)
    ref["_team_match"] = ref["team_name"].map(name_key)
    name_aliases = {"laclippers": "losangelesclippers", "lalakers": "losangeleslakers"}
    other["_team_match"] = other["_team_match"].replace(name_aliases)
    # Team abbreviations in stats may not match the historical official code.
    # Translate common code variants only within the same season.
    expected = ref[["Season", "Team"]].drop_duplicates()
    map_by_season = {}
    for year, season_set in expected.groupby("Season"):
        codes = set(season_set["Team"])
        for code in codes:
            map_by_season[(int(year), code.casefold())] = code
        for alias, formal in CODE_ALIASES.items():
            if formal in codes:
                map_by_season[(int(year), alias.casefold())] = formal
    short_form = other.apply(
        lambda x: map_by_season.get((int(x["Season"]), str(x[source_col]).casefold()))
        if pd.notna(x["Season"]) else None,
        axis=1,
    )
    code_to_name = ref.set_index(["Season", "Team"])["_team_match"].to_dict()
    for idx, code in short_form.items():
        if code is not None:
            yr = int(other.at[idx, "Season"])
            other.at[idx, "_team_match"] = code_to_name[(yr, code)]
    # Join only outcome columns, avoiding clashes in field names.
    attrs = [c for c in ref.columns if c not in {"seasonStartYear", "Team", "team_name", "_team_match"}]
    cols_to_add = [c for c in attrs if c != "Season" and c not in other.columns]
    return other.merge(ref[["Season", "_team_match", *cols_to_add]],
                       on=["Season", "_team_match"], how="left", validate="many_to_one").drop(columns="_team_match")


def build_outputs(outdir, data, first_data_year, last_data_year, playoff_start, playoff_end):
    outcomes = data.get("playoffs")
    if outcomes is not None:
        if outcomes.duplicated(["Season", "Team"]).any():
            raise ValueError("Duplicate season/team keys in historical outcomes")
        # Assemble one team-season row per actual franchise/year.
        team = outcomes.copy()
        payroll = data.get("payroll")
        if payroll is not None:
            payroll_join = payroll[["Season", "team", "payroll"]].copy()
            payroll_join = payroll_join.drop_duplicates(["Season", "team"])
            # Attach payroll to the correctly matched historical team-season.
            # Match full names first, then exact season-specific team codes.
            ids = outcomes[["Season", "Team", "team_name"]].copy()
            ids["_join_key"] = ids["team_name"].map(name_key)
            payroll_join["_join_key"] = payroll_join["team"].map(name_key).replace({
                "laclippers": "losangelesclippers", "lalakers": "losangeleslakers"})
            ids["_season_name"] = list(zip(ids["Season"], ids["_join_key"]))
            name_to_code = ids.set_index("_season_name")["Team"].to_dict()
            payroll_join["_TeamCode"] = [name_to_code.get((int(s), key))
                                          for s, key in zip(payroll_join["Season"], payroll_join["_join_key"])]
            # Permit input already stored as season-specific BBRef abbreviation.
            known = set(zip(outcomes["Season"], outcomes["Team"]))
            for idx, r in payroll_join.loc[payroll_join["_TeamCode"].isna()].iterrows():
                direct = str(r["team"]).upper()
                if (r["Season"], direct) in known:
                    payroll_join.at[idx, "_TeamCode"] = direct
            unmatched = int(payroll_join["_TeamCode"].isna().sum())
            if unmatched:
                print(f"Payroll: {unmatched} team-seasons could not match historical team names/codes")
            payroll_join = payroll_join.dropna(subset=["_TeamCode"])
            if payroll_join.duplicated(["Season", "_TeamCode"]).any():
                raise ValueError("Conflicting payroll entries for the same team and season")
            team = team.merge(payroll_join[["Season", "_TeamCode", "payroll"]],
                              left_on=["Season", "Team"], right_on=["Season", "_TeamCode"],
                              how="left", validate="one_to_one").drop(columns="_TeamCode")
        else:
            # Explicitly show that payroll is missing, NOT zero.
            team["payroll"] = pd.NA
        team_path = outdir / f"NBA_team_seasons_{playoff_start}_{playoff_end}.csv"
        team.to_csv(team_path, index=False)
        print(f"Created {team_path.name} ({len(team):,} team-seasons)")
        for kind, team_col in (("stats", "Team"), ("payroll", "team")):
            source = data.get(kind)
            if source is None:
                continue
            merged = match_teams(source, outcomes, team_col)
            target = outdir / f"NBA_{kind}_with_playoffs_{first_data_year}_{last_data_year}.csv"
            merged.to_csv(target, index=False)
            # For 2027 a correctly joined team still has a missing playoff result.
            # Distinguish pending outcomes from genuinely unmatched teams.
            missing = merged["source_standings"].isna().sum()
            pending = ((merged["Season"] == CURRENT_SEASON_END)
                       & merged["source_standings"].notna()).sum()
            print(f"Created {target.name}; {missing:,} team joins unmatched; "
                  f"{pending:,} current-season playoff outcomes pending")
    # Single player-season stats row (aggregate 2TM/3TM if traded) + salary.
    stats = data.get("stats")
    salaries = data.get("salaries")
    if stats is not None and salaries is not None:
        s = stats.copy()
        s["_name"] = s["Player"].map(name_key)
        s["_aggregate"] = s["Team"].astype(str).str.fullmatch(r"(?:TOT|[2-9]TM)", case=False)
        counts = s.groupby(["Season", "_name"])["_name"].transform("size")
        preferred = s.loc[(counts == 1) | s["_aggregate"]].copy()
        # No duplicating one salary over player rows from multiple teams.
        preferred = preferred.loc[~preferred.duplicated(["Season", "_name"], keep=False)]
        m = salaries.copy()
        m["_name"] = m["playerName"].map(name_key)
        m = m.loc[~m.duplicated(["Season", "_name"], keep=False)]
        combined = preferred.merge(m[["Season", "_name", "salary"]], on=["Season", "_name"],
                                   how="left", validate="one_to_one")
        combined = combined.drop(columns=["_name", "_aggregate"])
        file = outdir / f"NBA_player_season_stats_salaries_{first_data_year}_{last_data_year}.csv"
        combined.to_csv(file, index=False)
        print(f"Created {file.name}: {len(combined):,} player-seasons; "
              f"{combined['salary'].isna().sum():,} salaries unmatched")



def merge_existing_csv(path, outcomes, outdir, team_col=None, season_col=None):
    """Enrich an older Logan Lauton / user-supplied CSV with historical outcomes.

    Accepts 'seasonStartYear' as season start; 'Season', 'Year', 'season'
    mean season END. Unmatched rows are preserved and reported.
    """
    src = pd.read_csv(path)
    if team_col is None:
        team_col = next((c for c in ("team", "Team", "Tm", "team_name") if c in src), None)
    if season_col is None:
        season_col = next((c for c in ("seasonStartYear", "Season", "Year", "season") if c in src), None)
    if not team_col or not season_col or team_col not in src or season_col not in src:
        raise ValueError(f"{path}: no team/season column; specify --team-col and --season-col")
    if "Season" not in src or season_col != "Season":
        vals = pd.to_numeric(src[season_col], errors="coerce")
        src["Season"] = vals + (1 if season_col == "seasonStartYear" else 0)
    src["Season"] = pd.to_numeric(src["Season"], errors="coerce")
    out = match_teams(src, outcomes, team_col)
    target = outdir / f"{Path(path).stem}_with_playoffs.csv"
    out.to_csv(target, index=False)
    print(f"Enriched {target.name}: {len(out):,} original rows, "
          f"{out['Playoff_Result'].isna().sum():,} unmatched")
    return target


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", choices=["all", "stats", "salaries", "payroll", "playoffs"], default="all")
    p.add_argument("--skip-money", action="store_true",
                   help="With --kind all, collect only player stats and team standings/playoffs; "
                        "skip player salaries and team payroll entirely")
    p.add_argument("--playoffs-start", type=int, default=1947, help="First playoff season END year")
    p.add_argument("--playoffs-end", type=int, default=CURRENT_SEASON_END,
                   help="Last team-results season END year (2027 is in progress; playoffs pending)")
    p.add_argument("--data-start", type=int, default=1947, help="First stats/salaries/payroll season END year (default 1947)")
    p.add_argument("--data-end", type=int, default=CURRENT_SEASON_END,
                   help="Last stats/salaries/payroll season END year (2027 includes live stats)")
    p.add_argument("--earliest-money-year", type=int, default=DEFAULT_EARLIEST_MONEY_YEAR,
                   help="First season-ending year to TRY scraping salaries/payrolls (default 1985). "
                        "Years earlier remain explicitly unavailable, not $0.")
    p.add_argument("--outdir", default="nba_updated")
    p.add_argument("--delay", type=float, default=8.0, help="Pause after requests, minimum 6 seconds")
    p.add_argument("--refresh", action="store_true",
                   help="Replace cached historical CSVs (2027 always refreshes)")
    p.add_argument("--merge-existing", action="append", default=[], metavar="CSV",
                   help="Also join historical playoff results onto an existing local CSV (repeatable)")
    p.add_argument("--team-col", help="Team column to use for --merge-existing")
    p.add_argument("--season-col", help="Season column to use for --merge-existing")
    args = p.parse_args(argv)
    if args.skip_money and args.kind in {"salaries", "payroll"}:
        p.error("--skip-money cannot be combined with --kind salaries or payroll")
    if not 1947 <= args.playoffs_start <= args.playoffs_end <= CURRENT_SEASON_END:
        p.error("Playoff ending years must be between 1947 and 2027")
    if not 1947 <= args.data_start <= args.data_end <= CURRENT_SEASON_END:
        p.error("Data ending years must be between 1947 and 2027")
    if args.delay < 6:
        p.error("--delay must be at least 6 seconds")
    if not 1947 <= args.earliest_money_year <= CURRENT_SEASON_END:
        p.error("--earliest-money-year must be between 1947 and 2027")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({
        "User-Agent": "NBA-research-project/1.0 (polite academic scraper)",
        "Accept-Language": "en-US,en;q=0.9",
    })
    kinds = (["playoffs", "stats"] if args.skip_money else
             ["playoffs", "stats", "salaries", "payroll"]) if args.kind == "all" else [args.kind]
    all_data = {}
    log = []
    for kind in kinds:
        years = (range(args.playoffs_start, args.playoffs_end + 1) if kind == "playoffs"
                 else range(args.data_start, args.data_end + 1))
        data = safe_collect(session, kind, years, outdir, args.delay, args.refresh, log,
                            earliest_money_year=args.earliest_money_year)
        if data is not None:
            all_data[kind] = data
    if log:
        report = pd.DataFrame(log)
        report.to_csv(outdir / "download_report.csv", index=False)
        report.groupby(["kind", "status"]).size().rename("seasons").reset_index().to_csv(
            outdir / "coverage_summary.csv", index=False)
        print("Season coverage by dataset:")
        print(report.groupby(["kind", "status"]).size().to_string())
    if all_data:
        try:
            build_outputs(outdir, all_data, args.data_start, args.data_end,
                          args.playoffs_start, args.playoffs_end)
            if args.merge_existing:
                outcomes = all_data.get("playoffs")
                if outcomes is None:
                    raise ValueError("--merge-existing needs playoff results; use --kind all or playoffs")
                for file in args.merge_existing:
                    merge_existing_csv(file, outcomes, outdir, args.team_col, args.season_col)
        except (KeyError, ValueError, pd.errors.MergeError) as e:
            print(f"Some merged outputs could not be made: {e}", file=sys.stderr)
            return 1
    incomplete = any(entry["status"] in {
        "failed", "blocked", "not_attempted", "not_found"
    } for entry in log)
    if incomplete:
        print("PARTIAL: see download_report.csv for missing seasons; cached files are retained.",
              file=sys.stderr)
        return 1
    print("Finished attempting available-source seasons. See coverage_summary.csv "
          "for historical salary/payroll gaps and download_report.csv for each year.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
