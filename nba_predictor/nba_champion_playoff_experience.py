"""Actual player postseason history and continuity of the current personnel.

Only prior seasons enter history. Current regular-season minute weights and
the last listed regular-season coach define the team being predicted.
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd

from nba_predictor.nba_advanced_stats import player_key, team_key

HISTORY_STATS = ('G', 'MP', 'Series_Wins', 'Conference_Finals', 'Finals')
ACTUAL_EXPERIENCE_FEATURES = tuple(f'Roster_Prior_Playoff_{stat}' for stat in HISTORY_STATS) + tuple(
    f'{prefix}_Player_Prior_Playoff_{stat}' for prefix in ('Best', 'Second') for stat in HISTORY_STATS) + (
    'Coach_Prior_Playoff_Games', 'Inexperienced_Roster_Minutes_Share')
CONTINUITY_FEATURES = ('Prior_Playoff_Roster_Overlap_Last_Season', 'Same_Coach_As_Last_Season',
                       'Continuity_Weighted_Series_Wins_Last5',
                       'Continuity_Weighted_Conference_Finals_Last5', 'Continuity_Weighted_Finals_Last5')
PLAYOFF_EXPERIENCE_FEATURES = (*ACTUAL_EXPERIENCE_FEATURES, *CONTINUITY_FEATURES)
EXPERIENCE_RULE = (
    'Actual prior playoff games/minutes and participation in won series, conference finals, and NBA Finals. '
    'Current-roster averages use regular-season team-stint minutes. Player history follows stable NBA ids. '
    'Only seasons before the prediction year count. Conference finals start in 1971. '
    'Team success history uses last-five-season playoff-roster minute overlap, coaching continuity, '
    'and a three-year half-life; a fully changed roster receives zero transferred team success.')


def identity_key(name):
    name = str(name).replace('ı', 'i').replace('ð', 'd').replace('Ð', 'D')
    name = re.sub(r'(?:\s+(?:jr\.?|sr\.?|ii|iii|iv))+$', '', name, flags=re.I)
    return player_key(name).replace('ı', 'i').replace('ð', 'd')


NAME_ALIASES = {'stevesmith': 'stevensmith', 'cliffrobinson': 'clifftrobinson',
                'michaelrayrichardson': 'michealrayrichardson', 'kjmartin': 'kenyonmartin',
                'tinyarchibald': 'natearchibald', 'fatlever': 'lafayettelever',
                'worldbfree': 'worldfree', 'pearlwashington': 'dwaynewashington',
                'dwightdavis': 'doubleddavis', 'lutherrackley': 'lukerackley',
                'dickgarrett': 'eldogarrett', 'ronaldmurray': 'flipmurray',
                'barryclemens': 'johnclemens', 'budstallworth': 'isaacstallworth',
                'molayton': 'dennislayton', 'mannyleaks': 'emanuelleaks',
                'birdaveritt': 'williamaveritt', 'bubbleshawkins': 'roberthawkins',
                'jjanderson': 'mitchellanderson', 'hawkeyewhitney': 'charleswhitney',
                'billhosket': 'wilmerhosket', 'eugenejeter': 'poohjeter',
                'rickymarsh': 'ericmarsh', 'tickyburden': 'lutherburden',
                'geoffcrompton': 'jeffreycrompton', 'budogden': 'carlosogden',
                'gookennedy': 'eugenekennedy', 'moebarr': 'thomasbarr',
                'hutchjones': 'williejones', 'chubbycox': 'johncox',
                'normrichardson': 'normanrichardson', 'cheesejohnson': 'lynbertjohnson',
                'lewalcindor': 'kareemabduljabbar'}


def match_roster_identities(roster, identities):
    """Conservative, auditable crosswalk; unresolved players stay unknown."""
    identities = identities.copy()
    verified = (identities.Verified_NBA_Game_ID.astype(str).str.lower().eq('true')
                if 'Verified_NBA_Game_ID' in identities else pd.to_numeric(identities.nbaFlag, errors='coerce').eq(1))
    identities = identities.loc[verified].copy()
    identities['Player'] = identities.firstName.fillna('') + ' ' + identities.lastName.fillna('')
    identities['_name'] = identities.Player.map(identity_key)
    identities['_surname'] = identities.lastName.map(identity_key)
    identities['_first'] = identities.firstName.map(identity_key)
    identities['_birth'] = pd.to_datetime(identities.birthDate, errors='coerce').dt.year
    dates = pd.to_datetime(identities.birthDate, errors='coerce')
    identities['_birthday_after_feb1'] = dates.dt.month.gt(2) | (dates.dt.month.eq(2) & dates.dt.day.gt(1))
    by_name = {key: group for key, group in identities.groupby('_name')}
    by_surname = {key: group for key, group in identities.groupby('_surname')}
    by_first = {key: group for key, group in identities.groupby('_first')}
    matches = []
    for row in roster[['Player', 'Age', 'Season']].drop_duplicates().itertuples(index=False):
        key = identity_key(row.Player)
        key = NAME_ALIASES.get(key, key)
        candidates = by_name.get(key, identities.iloc[:0])
        method = 'exact normalized name and age'
        if len(candidates):
            age_matches = candidates.loc[(row.Season - row.Age - candidates._birth).abs().le(1)]
            if len(age_matches):
                candidates = age_matches
            elif candidates._birth.notna().any():
                # Some source birth dates / season ages differ by one extra
                # year. Require an exact unique name AND active career match;
                # do not widen the surname fallback for ambiguous names.
                active = candidates.fromYear.add(1).le(row.Season) & (
                    candidates.toYear.isna() | candidates.toYear.add(1).ge(row.Season))
                if candidates.personId.nunique() == 1 and active.all():
                    candidates = candidates.loc[active]
                    method = 'unique exact NBA name and active career; source birth-date discrepancy'
                else:
                    candidates = candidates.loc[(row.Season - row.Age - candidates._birth).abs().le(2) & active]
                    method = 'exact normalized name and active career; age discrepancy <= 2 years'
        if not len(candidates):
            surname = identity_key(str(row.Player).split()[-1])
            candidates = by_surname.get(surname, identities.iloc[:0])
            candidates = candidates.loc[(row.Season - row.Age - candidates._birth).abs().le(1)]
            initial = identity_key(str(row.Player).split()[0])[:1]
            candidates = candidates.loc[candidates._first.str.startswith(initial)]
            method = 'unique surname and birth-year match'
        if not len(candidates):
            # Single names and reversed name order (e.g. Nene / Yi Jianlian).
            token = identity_key(str(row.Player).split()[-1])
            candidates = by_first.get(token, identities.iloc[:0])
            candidates = candidates.loc[(row.Season - row.Age - candidates._birth).abs().le(1)]
            method = 'unique first-name token and birth-year match'
        ids = candidates.personId.drop_duplicates()
        if len(ids) > 1:
            exact_age = row.Season - candidates['_birth'] - candidates['_birthday_after_feb1'].astype(int)
            exact = candidates.loc[exact_age.eq(row.Age)]
            if exact.personId.nunique() == 1:
                candidates = exact
                ids = candidates.personId.drop_duplicates()
                method += '; exact season age disambiguates'
        resolved = len(ids) == 1
        matches.append(dict(Player=row.Player, Age=row.Age, Season=row.Season,
                            Player_ID=int(ids.iloc[0]) if resolved else np.nan,
                            Career_First_Season=float(candidates.fromYear.iloc[0] + 1) if resolved and 'fromYear' in candidates else np.nan,
                            Identity_Match=method if resolved else 'unresolved or ambiguous'))
    return roster.merge(pd.DataFrame(matches), on=['Player', 'Age', 'Season'], how='left', validate='many_to_one')


def roster_overlap(current, previous):
    """Intersection of normalized minute distributions; zero for disjoint rosters."""
    return sum(min(weight, previous.get(player, 0.0)) for player, weight in current.items())


def continuity_weight(overlap, same_coach, years_ago):
    if years_ago < 1:
        raise ValueError('Continuity can use only strictly prior seasons')
    return overlap * (1.0 if same_coach else .5) * .5 ** ((years_ago - 1) / 3)


def add_playoff_experience(metrics, stats_path: Path, history_dir: Path):
    metadata = pd.read_json(history_dir/'metadata.json', typ='series')
    history_start, history_end = int(metadata.history_start), int(metadata.history_end)
    if not bool(metadata.source_coverage_passed):
        raise ValueError('Playoff source coverage has not passed')
    if metrics.Season.max() - 1 > history_end:
        raise ValueError('Playoff history ends before the required prior season')
    players = pd.read_csv(history_dir/'players.csv')
    teams = pd.read_csv(history_dir/'teams.csv')
    coaches = pd.read_csv(history_dir/'coaches.csv')
    codes = pd.read_csv(history_dir/'team_identities.csv')
    identities = pd.read_csv(history_dir/'player_identities.csv', low_memory=False)
    roster = pd.read_csv(stats_path, low_memory=False).drop_duplicates()
    for column in ('Season', 'Age', 'MP'):
        roster[column] = pd.to_numeric(roster[column], errors='coerce')
    roster = roster.loc[roster.Season.isin(metrics.Season) & roster.MP.gt(0) & roster.Age.notna() & roster.Team.notna()].copy()
    roster['Team'] = roster.Team.map(team_key)
    roster = roster.loc[roster.Team.ne('TOT')].copy()
    requested = metrics[['Season', 'Team']].copy()
    requested['Team'] = requested.Team.map(team_key)
    roster = roster.merge(requested, on=['Season', 'Team'], how='inner', validate='many_to_one')
    roster['_name'] = roster.Player.map(player_key)
    if roster.duplicated(['Season', 'Team', '_name', 'Age']).any():
        raise ValueError('Ambiguous current roster stints')
    roster = match_roster_identities(roster, identities)
    # A nickname collision must not double count the same player-team minutes.
    known = roster.loc[roster.Player_ID.notna()]
    if known.duplicated(['Season', 'Team', 'Player_ID']).any():
        raise ValueError('Two roster identities resolve to the same player-team stint')
    roster = roster.merge(codes, on=['Season', 'Team'], how='left', validate='many_to_one')
    if roster.Team_ID.isna().any():
        raise ValueError('Missing current team id in roster history')
    if players.duplicated(['Season', 'Player_ID', 'Team_ID']).any():
        raise ValueError('Duplicate actual player playoff stints')
    for stat in HISTORY_STATS:
        players[stat] = pd.to_numeric(players[stat], errors='coerce')
    if players[list(HISTORY_STATS)].isna().any().any() or players[list(HISTORY_STATS)].lt(0).any().any():
        raise ValueError('Incomplete or invalid actual playoff statistics')
    coach_choice = coaches.loc[coaches.Regular_Games.gt(0)].sort_values('Coach_Order').drop_duplicates(
        ['Season', 'Team'], keep='last').set_index(['Season', 'Team'])
    coach_history = coaches.groupby(['Season', 'Coach_Key']).Playoff_Games.sum().reset_index()
    franchise_codes = codes.dropna(subset=['Team_ID']).set_index(['Season', 'Team_ID']).Team
    past_rosters = {}
    for key, group in players.groupby(['Season', 'Team_ID']):
        past_rosters[key] = dict(zip(group.Player_ID, group.MP / group.MP.sum()))
    team_lookup = teams.set_index(['Season', 'Team_ID'])
    roster_groups = {(int(year), team): group for (year, team), group in roster.groupby(['Season', 'Team'])}
    history, coach_totals, feature_rows, player_audits, comparisons = {}, {}, [], [], []
    # Build from the source beginning, not just the training start. At year Y,
    # update Y-1 before feature construction; never add Y's playoff outcomes.
    minimum = int(metrics.Season.min())
    for year in range(history_start, int(metrics.Season.max()) + 1):
        prior = players.loc[players.Season.eq(year - 1)]
        for player, group in prior.groupby('Player_ID'):
            history[player] = history.get(player, np.zeros(len(HISTORY_STATS))) + group[list(HISTORY_STATS)].sum().to_numpy(dtype=float)
        for row in coach_history.loc[coach_history.Season.eq(year - 1)].itertuples(index=False):
            coach_totals[row.Coach_Key] = coach_totals.get(row.Coach_Key, 0) + row.Playoff_Games
        if year < minimum:
            continue
        for metric in metrics.loc[metrics.Season.eq(year)].itertuples(index=False):
            team = team_key(metric.Team)
            group = roster_groups.get((year, team))
            if group is None or not len(group):
                raise ValueError(f'Missing current roster {year}/{team}')
            total = group.MP.sum()
            matched = group.loc[group.Player_ID.notna() &
                                ~group.Career_First_Season.lt(history_start)].copy()
            coverage = matched.MP.sum() / total
            if coverage < .95:
                missing = group.loc[group.Player_ID.isna(), ['Player', 'MP']].to_dict('records')
                raise ValueError(f'Player identity coverage {coverage:.1%} in {year}/{team}: {missing}')
            values = np.array([history.get(player, np.zeros(len(HISTORY_STATS))) for player in matched.Player_ID])
            for index, stat in enumerate(HISTORY_STATS):
                matched[f'Prior_Playoff_{stat}'] = values[:, index]
            matched['Current_Minutes_Weight'] = matched.MP / total
            # Unknown history is not zero. Aggregate only identified minutes;
            # report the original minute coverage independently.
            row = dict(Season=year, Team=metric.Team, Identity_Minutes_Coverage=coverage)
            for stat in HISTORY_STATS:
                row[f'Roster_Prior_Playoff_{stat}'] = np.average(matched[f'Prior_Playoff_{stat}'], weights=matched.MP)
            row['Inexperienced_Roster_Minutes_Share'] = matched.loc[matched.Prior_Playoff_G.eq(0), 'MP'].sum() / matched.MP.sum()
            for prefix in ('Best', 'Second'):
                name = getattr(metric, f'{prefix}_Player_Name', None)
                selected = matched.loc[matched['_name'].eq(player_key(name))]
                if len(selected) != 1:
                    raise ValueError(f'Cannot resolve selected {prefix.lower()} player {name!r} in {year}/{team}')
                for stat in HISTORY_STATS:
                    row[f'{prefix}_Player_Prior_Playoff_{stat}'] = selected[f'Prior_Playoff_{stat}'].iloc[0]
            coach = coach_choice.loc[(year, team)] if (year, team) in coach_choice.index else None
            if coach is None:
                raise ValueError(f'Missing current coach {year}/{team}')
            row['Coach_Name'] = coach.Coach
            row['Coach_Prior_Playoff_Games'] = coach_totals.get(coach.Coach_Key, 0)
            team_id = int(group.Team_ID.iloc[0])
            prior_team_code = franchise_codes.get((year - 1, team_id), team)
            last_coach = coach_choice.loc[(year - 1, prior_team_code)] if (year - 1, prior_team_code) in coach_choice.index else None
            row['Same_Coach_As_Last_Season'] = int(last_coach is not None and coach.Coach_Key == last_coach.Coach_Key)
            current = dict(zip(matched.Player_ID, matched.MP / total))
            row['Prior_Playoff_Roster_Overlap_Last_Season'] = roster_overlap(current, past_rosters.get((year - 1, team_id), {}))
            for stat in ('Series_Wins', 'Conference_Finals', 'Finals'):
                row[f'Continuity_Weighted_{stat}_Last5'] = 0.0
            # Audit all earlier playoff rosters, including distant title eras.
            # Only the predeclared five-year window enters model features.
            for previous in range(history_start, year):
                if (previous, team_id) not in team_lookup.index:
                    continue
                past = team_lookup.loc[(previous, team_id)]
                past_coach = coach_choice.loc[(previous, past.Team)] if (previous, past.Team) in coach_choice.index else None
                same = past_coach is not None and coach.Coach_Key == past_coach.Coach_Key
                overlap = roster_overlap(current, past_rosters[(previous, team_id)])
                shared = set(current) & set(past_rosters[(previous, team_id)])
                names = matched.set_index('Player_ID').Player
                in_window = previous >= year - 5
                weight = continuity_weight(overlap, same, year - previous) if in_window else 0.0
                for stat in ('Series_Wins', 'Conference_Finals', 'Finals'):
                    row[f'Continuity_Weighted_{stat}_Last5'] += weight * past[stat]
                comparisons.append(dict(Season=year, Team=metric.Team, Historical_Season=previous,
                                        Current_Coach=coach.Coach, Historical_Coach=past_coach.Coach if past_coach is not None else None,
                                        Roster_Overlap=overlap, Same_Coach=bool(same), Transfer_Weight=weight,
                                        Used_In_Model_History=in_window,
                                        Current_Player_Count=len(current), Historical_Player_Count=len(past_rosters[(previous, team_id)]),
                                        Shared_Player_Count=len(shared),
                                        Shared_Players='; '.join(sorted(str(names.get(player, player)) for player in shared)),
                                        Prior_Series_Wins=past.Series_Wins, Prior_Conference_Finals=past.Conference_Finals,
                                        Prior_Finals=past.Finals))
            feature_rows.append(row)
            for column in ('Identity_Minutes_Coverage', 'Coach_Name'):
                matched[column] = row[column]
            player_audits.append(matched)
    audit = pd.DataFrame(feature_rows)
    if len(audit) != len(metrics):
        raise ValueError('Some requested team-seasons precede the available playoff history')
    player_audit = pd.concat(player_audits, ignore_index=True)
    # Include unmatched identities in their own audit rather than silently
    # interpreting them as rookies with zero experience.
    identities_audit = roster[['Season', 'Team', 'Player', 'Age', 'MP', 'Player_ID', 'Identity_Match', 'Career_First_Season']].copy()
    identities_audit['History_Left_Censored'] = identities_audit.Career_First_Season.lt(history_start)
    result = metrics.drop(columns=list(PLAYOFF_EXPERIENCE_FEATURES), errors='ignore').merge(
        audit[['Season', 'Team', *PLAYOFF_EXPERIENCE_FEATURES]], on=['Season', 'Team'], validate='one_to_one')
    return result, audit, player_audit, pd.DataFrame(comparisons), identities_audit


def experience_candidates(frame, train, minimum_coverage, include_continuity=True):
    result = []
    for column in PLAYOFF_EXPERIENCE_FEATURES if include_continuity else ACTUAL_EXPERIENCE_FEATURES:
        if column in frame:
            values = pd.to_numeric(frame.loc[train, column], errors='coerce').replace([np.inf, -np.inf], np.nan)
            if values.notna().mean() >= minimum_coverage and values.notna().sum() > 10 and values.std() > 1e-10:
                result.append(column)
    return result
