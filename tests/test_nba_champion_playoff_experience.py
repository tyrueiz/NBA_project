import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from nba_predictor.nba_advanced_stats import player_key
from nba_predictor.nba_champion_playoff_experience import (ACTUAL_EXPERIENCE_FEATURES, PLAYOFF_EXPERIENCE_FEATURES, add_playoff_experience,
                                             roster_overlap, continuity_weight, match_roster_identities)
from nba_predictor.nba_download_playoff_history import build_history, parse_coaches, parse_season_totals, season_from_game_id
from nba_predictor.nba_predict_champions_2010_2026 import available_features as logistic_features
from nba_predictor.nba_predict_champions_xgboost_2010_2026 import available_features as xgb_features
from nba_predictor.nba_compare_playoff_experience import feature_configurations


class ActualExperienceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stats = self.root/'regular.csv'
        self.metrics = pd.DataFrame([
            dict(Season=2021, Team='AAA', Best_Player_Name='Veteran One', Second_Player_Name='Other Two'),
            dict(Season=2021, Team='BBB', Best_Player_Name='Rookie Three', Second_Player_Name='New Four')])
        self.roster = pd.DataFrame([
            dict(Season=2021, Team=team, Player=name, Age=age, MP=mp)
            for team, name, age, mp in [('AAA', 'Veteran One', 31, 1500), ('AAA', 'Other Two', 30, 500),
                                       ('BBB', 'Rookie Three', 21, 1000), ('BBB', 'New Four', 21, 1000)]])
        self.identities = pd.DataFrame([
            dict(personId=pid, firstName=first, lastName=last, birthDate=birth, nbaFlag=1,
                 fromYear=2011 if pid < 3 else 2020, toYear=np.nan, Verified_NBA_Game_ID=True)
            for pid, first, last, birth in [(1, 'Veteran', 'One', '1990-01-01'), (2, 'Other', 'Two', '1991-01-01'),
                                           (3, 'Rookie', 'Three', '2000-01-01'), (4, 'New', 'Four', '2000-01-01')]])
        self.players = pd.DataFrame([
            dict(Season=2020, Player_ID=1, Team_ID=10, Team='AAA', G=10, MP=300, Series_Wins=1, Conference_Finals=1, Finals=0),
            dict(Season=2020, Player_ID=2, Team_ID=10, Team='AAA', G=5, MP=100, Series_Wins=1, Conference_Finals=0, Finals=0),
            dict(Season=2021, Player_ID=3, Team_ID=20, Team='BBB', G=20, MP=500, Series_Wins=4, Conference_Finals=1, Finals=1)])
        self.teams = pd.DataFrame([
            dict(Season=2020, Team_ID=10, Team='AAA', Playoff_Games=10, Series_Wins=2, Conference_Finals=1, Finals=0),
            dict(Season=2021, Team_ID=20, Team='BBB', Playoff_Games=20, Series_Wins=4, Conference_Finals=1, Finals=1)])
        self.coaches = pd.DataFrame([
            dict(Season=2020, Team='AAA', Coach='Old Coach', Coach_Order=1, Regular_Games=82, Playoff_Games=10),
            dict(Season=2020, Team='BBB', Coach='New Coach', Coach_Order=1, Regular_Games=82, Playoff_Games=0),
            dict(Season=2021, Team='AAA', Coach='Old Coach', Coach_Order=1, Regular_Games=20, Playoff_Games=0),
            dict(Season=2021, Team='AAA', Coach='New Coach', Coach_Order=2, Regular_Games=62, Playoff_Games=999),
            dict(Season=2021, Team='BBB', Coach='Other Coach', Coach_Order=1, Regular_Games=82, Playoff_Games=20)])

    def calculate(self):
        self.roster.to_csv(self.stats, index=False)
        self.identities.to_csv(self.root/'player_identities.csv', index=False)
        self.players.to_csv(self.root/'players.csv', index=False)
        self.teams.to_csv(self.root/'teams.csv', index=False)
        self.coaches.assign(Coach_Key=self.coaches.Coach.map(player_key)).to_csv(self.root/'coaches.csv', index=False)
        pd.DataFrame([dict(Season=y, Team=t, Team_ID=pid) for y in (2020, 2021)
                      for t, pid in [('AAA', 10), ('BBB', 20)]]).to_csv(self.root/'team_identities.csv', index=False)
        (self.root/'metadata.json').write_text(json.dumps(dict(history_start=2010, history_end=2021, source_coverage_passed=True)))
        return add_playoff_experience(self.metrics, self.stats, self.root)

    def test_actual_games_minutes_and_rounds_are_minute_weighted(self):
        frame, audit, _, _, _ = self.calculate()
        first = frame.iloc[0]
        self.assertEqual(first.Best_Player_Prior_Playoff_G, 10)
        self.assertEqual(first.Second_Player_Prior_Playoff_G, 5)
        self.assertEqual(first.Roster_Prior_Playoff_G, 8.75)
        self.assertEqual(first.Roster_Prior_Playoff_MP, 250)
        self.assertEqual(first.Roster_Prior_Playoff_Conference_Finals, .75)
        self.assertEqual(first.Inexperienced_Roster_Minutes_Share, 0)
        self.assertEqual(frame.Inexperienced_Roster_Minutes_Share.iloc[1], 1)
        self.assertEqual(audit.Coach_Name.iloc[0], 'New Coach')
        self.assertEqual(first.Coach_Prior_Playoff_Games, 0)

    def test_future_playoff_outcomes_and_coach_games_do_not_change_inputs(self):
        before = self.calculate()[0]
        self.players.loc[self.players.Season.eq(2021), ['G', 'MP', 'Series_Wins', 'Conference_Finals', 'Finals']] = 9999
        self.teams.loc[self.teams.Season.eq(2021), ['Playoff_Games', 'Series_Wins', 'Finals']] = 9999
        self.coaches.loc[self.coaches.Season.eq(2021), 'Playoff_Games'] = 99999
        after = self.calculate()[0]
        pd.testing.assert_frame_equal(before, after)

    def test_player_keeps_experience_after_trade(self):
        self.roster.loc[self.roster.Player.eq('Veteran One'), 'Team'] = 'BBB'
        self.roster.loc[self.roster.Player.eq('Rookie Three'), 'Team'] = 'AAA'
        self.metrics.loc[self.metrics.Team.eq('AAA'), 'Best_Player_Name'] = 'Rookie Three'
        self.metrics.loc[self.metrics.Team.eq('BBB'), 'Best_Player_Name'] = 'Veteran One'
        frame = self.calculate()[0]
        self.assertEqual(frame.Best_Player_Prior_Playoff_G.tolist(), [0, 10])

    def test_continuity_requires_same_personnel_and_discounts_coach_change(self):
        frame, _, _, comparisons, _ = self.calculate()
        self.assertEqual(frame.Prior_Playoff_Roster_Overlap_Last_Season.iloc[0], 1)
        self.assertEqual(frame.Same_Coach_As_Last_Season.iloc[0], 0)
        self.assertEqual(frame.Continuity_Weighted_Series_Wins_Last5.iloc[0], 1)
        self.assertEqual(frame.Continuity_Weighted_Conference_Finals_Last5.iloc[0], .5)
        self.assertLess(comparisons.Historical_Season.max(), comparisons.Season.min())
        self.assertEqual(roster_overlap({1: .5, 2: .5}, {3: .5, 4: .5}), 0)
        self.assertEqual(continuity_weight(0, True, 1), 0)
        self.assertLess(continuity_weight(1, True, 4), continuity_weight(1, True, 1))
        with self.assertRaises(ValueError):
            continuity_weight(1, True, 0)

    def test_fully_replaced_roster_inherits_no_franchise_success(self):
        self.players.loc[self.players.Season.eq(2020), 'Player_ID'] += 100
        frame = self.calculate()[0]
        self.assertEqual(frame.Prior_Playoff_Roster_Overlap_Last_Season.iloc[0], 0)
        self.assertEqual(frame.Continuity_Weighted_Series_Wins_Last5.iloc[0], 0)
        self.assertEqual(frame.Continuity_Weighted_Conference_Finals_Last5.iloc[0], 0)

    def test_unknown_identity_is_not_a_rookie_and_low_coverage_stops(self):
        self.roster.loc[self.roster.Player.eq('Veteran One'), 'Player'] = 'Unknown Person'
        with self.assertRaisesRegex(ValueError, 'identity coverage'):
            self.calculate()

    def test_prior_careers_outside_source_range_are_not_zero_filled(self):
        self.identities.loc[self.identities.personId.eq(1), 'fromYear'] = 2000
        with self.assertRaisesRegex(ValueError, 'identity coverage'):
            self.calculate()

    def test_combined_rows_and_other_leagues_do_not_enter_roster(self):
        before = self.calculate()[0]
        extra = self.roster.iloc[:1].assign(Team='2TM', MP=999999)
        other_league = self.roster.iloc[:1].assign(Team='ABA', MP=999999)
        self.roster = pd.concat([self.roster, extra, other_league], ignore_index=True)
        pd.testing.assert_frame_equal(before, self.calculate()[0])
        self.roster = pd.concat([self.roster, self.roster.iloc[:1].assign(MP=600)], ignore_index=True)
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            self.calculate()

    def test_experience_is_opt_in_additive_and_consistent_between_models(self):
        frame = pd.DataFrame({c: range(30) for c in ['Win_Percentage', 'PTS', *PLAYOFF_EXPERIENCE_FEATURES]})
        train = pd.Series(True, index=frame.index)
        for selector in (logistic_features, xgb_features):
            base = selector(frame, train, .35, 1)
            self.assertEqual(base, ['Win_Percentage'])
            self.assertEqual(selector(frame, train, .35, 1, False, True), [*base, *PLAYOFF_EXPERIENCE_FEATURES])
            self.assertEqual(selector(frame, train, .35, 1, False, True, False), [*base, *ACTUAL_EXPERIENCE_FEATURES])
        configs = feature_configurations(['Win_Percentage'])
        self.assertFalse(set(configs['Actual']).intersection(set(configs['Proxy']) - set(configs['Actual'])))
        self.assertEqual(configs['Actual_Continuity'][:len(configs['Actual'])], configs['Actual'])


class HistorySourceTests(unittest.TestCase):
    def test_same_name_players_use_exact_season_age_and_ambiguity_stays_unknown(self):
        identities = pd.DataFrame([
            dict(personId=1, firstName='George', lastName='Johnson', birthDate='1947-06-19', nbaFlag=1, fromYear=1969, toYear=1980),
            dict(personId=2, firstName='George', lastName='Johnson', birthDate='1948-12-18', nbaFlag=1, fromYear=1969, toYear=1980)])
        roster = pd.DataFrame([dict(Player='George Johnson', Season=1973, Age=24)])
        result = match_roster_identities(roster, identities)
        self.assertEqual(result.Player_ID.iloc[0], 2)
        identities.loc[0, 'birthDate'] = '1948-12-19'
        self.assertTrue(match_roster_identities(roster, identities).Player_ID.isna().all())

    def test_historical_totals_parser_records_true_totals_not_per_game_minutes(self):
        html = ('<table><tbody><tr><th data-stat="name_display"><a href="/players/a/abc01.html">A Player</a></th>'
                '<td data-stat="age">25</td><td data-stat="team_name_abbr">AAA</td><td data-stat="g">10</td>'
                '<td data-stat="mp">320</td></tr></tbody></table>')
        result = parse_season_totals(html, 1990)
        self.assertEqual(result.minutesPg.iloc[0], 320)
        self.assertEqual(result.playerId.iloc[0], 'abc01')

    def test_coach_selection_records_only_regular_season_order(self):
        html = '<table id="NBA_coaches"><tbody>' + ''.join(
            f'<tr><th data-stat="coach"><a>{name}</a></th><td data-stat="team">AAA</td>'
            f'<td data-stat="cur_g">{g}</td><td data-stat="cur_g_p">{pg}</td></tr>'
            for name, g, pg in [('Old', 20, 0), ('New', 62, 99)]) + '</tbody></table>'
        result = parse_coaches(html, 2021)
        self.assertEqual(result.Coach_Order.tolist(), [1, 2])
        self.assertEqual(result.Regular_Games.tolist(), [20, 62])

    def test_game_seasons_do_not_depend_on_bubble_calendar(self):
        self.assertEqual(season_from_game_id(pd.Series([41900401, 42400407])).tolist(), [2020, 2025])

    def test_series_credit_requires_playing_in_that_series(self):
        games = pd.DataFrame([
            dict(gameId=42400301, Season=2025, gameType='Playoffs', gameLabel='East Conf. Finals',
                 gameDateTimeEst='2025-05-01', hometeamId=1, awayteamId=2, winner=1),
            dict(gameId=42400401, Season=2025, gameType='Playoffs', gameLabel='NBA Finals',
                 gameDateTimeEst='2025-06-01', hometeamId=1, awayteamId=3, winner=1)])
        players = pd.DataFrame([
            dict(gameId=42400301, personId=10, playerteamId=1, numMinutes=10, comment=None),
            dict(gameId=42400401, personId=10, playerteamId=1, numMinutes=np.nan, comment="DNP - Coach's Decision"),
            dict(gameId=42400401, personId=11, playerteamId=1, numMinutes=15, comment=None),
            dict(gameId=42400301, personId=20, playerteamId=2, numMinutes=10, comment=None),
            dict(gameId=42400401, personId=30, playerteamId=np.nan, home=0, numMinutes=15, comment=None)])
        coaches = pd.DataFrame([dict(Season=2025, Team=t, Playoff_Games=g) for t, g in [('AAA', 2), ('BBB', 1), ('CCC', 1)]])
        codes = pd.DataFrame([dict(Season=2025, Team=t, Team_ID=i) for t, i in [('AAA', 1), ('BBB', 2), ('CCC', 3)]])
        annual, teams, _, coverage = build_history(games, players, coaches, codes, 2025, 2025)
        first = annual.loc[annual.Player_ID.eq(10)].iloc[0]
        self.assertEqual(first.G, 1)
        self.assertEqual(first.Series_Wins, 1)
        self.assertEqual(first.Conference_Finals, 1)
        self.assertEqual(first.Finals, 0)
        self.assertTrue(coverage.Eligible.all())


if __name__ == '__main__':
    unittest.main()
