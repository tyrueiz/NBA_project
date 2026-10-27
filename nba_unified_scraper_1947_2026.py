
    
  
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
    kinds = ["playoffs", "stats", "salaries", "payroll"] if args.kind == "all" else [args.kind]
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

