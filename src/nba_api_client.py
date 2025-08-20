from nba_api.stats.endpoints import (
    playercareerstats, playergamelog,
    teamgamelog, leaguegamefinder, commonplayerinfo,
    teamyearbyyearstats, playerdashboardbyyearoveryear,
    playervsplayer, shotchartdetail
)
from nba_api.stats.static import players, teams
import pandas as pd
import time
from typing import Dict, List, Optional, Union
import requests
class NBAAPIClient:
    def __init__(self):
        self.rate_limit_delay = 0.6
        self._disambiguation_candidates = None
        self._disambiguation_query = None
    def _rate_limit(self):
        time.sleep(self.rate_limit_delay)
    def get_player_id(self, player_name: str) -> Optional[int]:
        cleaned_name = self._preprocess_player_name(player_name)
        return self._reverse_search_player(cleaned_name)
    def _preprocess_player_name(self, raw_name: str) -> str:
        import re
        raw_name = raw_name.strip()
        if len(raw_name.split()) <= 2:
            return raw_name
        potential_names = re.findall(r'[A-Z][a-zA-Z\']+(?:\s+[A-Z][a-zA-Z\']+)*', raw_name)
        action_words = {
            'Predict', 'Analyze', 'Compare', 'Show', 'Tell', 'Find', 'Get', 'Display'
        }
        for name in potential_names:
            name_parts = name.split()
            if len(name_parts) >= 2:
                if name_parts[0] not in action_words:
                    return name
                elif len(name_parts) > 2:
                    clean_name = ' '.join(name_parts[1:])
                    if len(clean_name.split()) >= 2:
                        return clean_name
        return raw_name
    def _reverse_search_player(self, player_name: str) -> Optional[int]:
        player_dict = players.find_players_by_full_name(player_name)
        if player_dict and len(player_dict) == 1:
            return player_dict[0]['id']
        all_players = players.get_players()
        player_name_lower = player_name.lower().strip()
        name_parts = player_name_lower.split()
        if not name_parts:
            return None
        exact_matches = []
        partial_matches = []
        for player in all_players:
            player_parts = player['full_name'].lower().split()
            if len(name_parts) == len(player_parts):
                if all(search_part == player_part for search_part, player_part in zip(name_parts, player_parts)):
                    exact_matches.append(player)
                    continue
            if len(name_parts) <= len(player_parts):
                confidence = self._calculate_player_confidence(player, name_parts)
                if confidence > 0:
                    partial_matches.append((player, confidence))
        if exact_matches:
            return exact_matches[0]['id']
        if not partial_matches:
            return None
        partial_matches.sort(key=lambda x: (x[1], x[0]['id']), reverse=True)
        return self._handle_player_disambiguation(partial_matches, player_name)
    def _calculate_player_confidence(self, player: dict, search_name_parts: list) -> float:
        player_parts = player['full_name'].lower().split()
        confidence = 0.0
        if len(search_name_parts) == 0 or len(player_parts) == 0:
            return 0.0
        exact_matches = 0
        partial_matches = 0
        total_search_parts = len(search_name_parts)
        def calculate_similarity(s1: str, s2: str) -> float:
            if s1 == s2:
                return 1.0
            if len(s1) >= 3 and len(s2) >= 3:
                if s1 in s2 or s2 in s1:
                    return 0.8
                longer = s1 if len(s1) > len(s2) else s2
                shorter = s2 if len(s1) > len(s2) else s1
                if len(longer) - len(shorter) <= 1:
                    common_chars = sum(1 for a, b in zip(shorter, longer) if a == b)
                    return common_chars / len(longer)
            return 0.0
        for i, search_part in enumerate(search_name_parts):
            best_match_score = 0.0
            best_position_bonus = 0.0
            for j, player_part in enumerate(player_parts):
                similarity = calculate_similarity(search_part, player_part)
                if similarity >= 1.0:
                    if i == 0 and j == 0:
                        position_bonus = 40.0
                    elif i == len(search_name_parts) - 1 and j == len(player_parts) - 1:
                        position_bonus = 35.0
                    else:
                        position_bonus = 25.0
                    if similarity > best_match_score:
                        best_match_score = similarity
                        best_position_bonus = position_bonus
                elif similarity >= 0.7:
                    if i == 0 and j == 0:
                        position_bonus = 25.0
                    elif i == len(search_name_parts) - 1 and j == len(player_parts) - 1:
                        position_bonus = 20.0
                    else:
                        position_bonus = 15.0
                    if similarity > best_match_score:
                        best_match_score = similarity
                        best_position_bonus = position_bonus * similarity
            if best_match_score >= 1.0:
                exact_matches += 1
                confidence += best_position_bonus
            elif best_match_score >= 0.7:
                partial_matches += 1
                confidence += best_position_bonus
        match_ratio = (exact_matches + partial_matches * 0.8) / total_search_parts
        if match_ratio < 0.4:
            return 0.0
        if len(search_name_parts) == len(player_parts) and exact_matches == total_search_parts:
            confidence += 20.0
        player_id = player['id']
        if player_id > 1000000:
            activity_bonus = 3.0
        elif player_id > 200000:
            activity_bonus = 5.0
        elif player_id > 100000:
            activity_bonus = 2.0
        else:
            activity_bonus = 1.0
        confidence += activity_bonus
        return confidence
    def _handle_player_disambiguation(self, scored_candidates: list, original_query: str) -> Optional[int]:
        if not scored_candidates:
            return None
        if len(scored_candidates) == 1:
            return scored_candidates[0][0]['id']
        best_confidence = scored_candidates[0][1]
        high_confidence_candidates = []
        for player, confidence in scored_candidates:
            if confidence >= best_confidence - 5.0:
                high_confidence_candidates.append((player, confidence))
            else:
                break
        if len(high_confidence_candidates) == 1 or \
           (len(high_confidence_candidates) >= 2 and
            high_confidence_candidates[0][1] - high_confidence_candidates[1][1] > 10.0):
            return high_confidence_candidates[0][0]['id']
        return self._request_user_disambiguation(high_confidence_candidates, original_query)
    def _request_user_disambiguation(self, candidates: list, original_query: str) -> Optional[int]:
        self._disambiguation_candidates = candidates
        self._disambiguation_query = original_query
        return None
    def get_disambiguation_options(self) -> Optional[Dict]:
        if not self._disambiguation_candidates:
            return None
        options = []
        for player, confidence in self._disambiguation_candidates:
            options.append({
                'id': player['id'],
                'name': player['full_name'],
                'confidence': round(confidence, 1)
            })
        return {
            'query': self._disambiguation_query,
            'options': options
        }
    def clear_disambiguation(self):
        self._disambiguation_candidates = None
        self._disambiguation_query = None
    def get_team_id(self, team_name: str) -> Optional[int]:
        team_dict = teams.find_teams_by_full_name(team_name)
        if team_dict:
            return team_dict[0]['id']
        all_teams = teams.get_teams()
        for team in all_teams:
            if team_name.lower() in team['full_name'].lower() or team_name.lower() in team['nickname'].lower():
                return team['id']
        return None
    def resolve_player_name(self, query_name: str) -> str:
        nickname_mappings = {
            'lebron': 'LeBron James',
            'bronny': 'Bronny James',
            'curry': 'Stephen Curry',
            'steph': 'Stephen Curry',
            'kd': 'Kevin Durant',
            'podz': 'Brandin Podziemski',
            'giannis': 'Giannis Antetokounmpo',
            'luka': 'Luka Doncic',
            'tatum': 'Jayson Tatum',
            'ad': 'Anthony Davis',
            'pg13': 'Paul George',
            'cp3': 'Chris Paul',
            'dwight': 'Dwight Howard',
            'westbrook': 'Russell Westbrook',
            'russ': 'Russell Westbrook',
            'ayo': 'Ayo Dosunmu',
            'dosunmu': 'Ayo Dosunmu',
            'coby': 'Coby White'
        }
        query_lower = query_name.lower().strip()
        words = query_lower.split()
        if len(words) == 1 and query_lower in nickname_mappings:
            return nickname_mappings[query_lower]
        return query_name
    def get_player_career_stats(self, player_id: int) -> pd.DataFrame:
        self._rate_limit()
        career = playercareerstats.PlayerCareerStats(player_id=player_id)
        return career.get_data_frames()[0]
    def get_player_season_stats(self, player_id: int, season: str) -> pd.DataFrame:
        self._rate_limit()
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        return gamelog.get_data_frames()[0]
    def get_player_info(self, player_id: int) -> Dict:
        self._rate_limit()
        info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
        return info.get_data_frames()[0].to_dict('records')[0]
    def get_team_season_stats(self, team_id: int, season: str) -> pd.DataFrame:
        self._rate_limit()
        team_stats = teamgamelog.TeamGameLog(team_id=team_id, season=season)
        return team_stats.get_data_frames()[0]
    def get_team_year_by_year(self, team_id: int) -> pd.DataFrame:
        self._rate_limit()
        team_years = teamyearbyyearstats.TeamYearByYearStats(team_id=team_id)
        return team_years.get_data_frames()[0]
    def get_player_vs_player_stats(self, player_id: int, vs_player_id: int, season: Optional[str] = None) -> pd.DataFrame:
        self._rate_limit()
        comparison = playervsplayer.PlayerVsPlayer(
            player_id=player_id,
            vs_player_id=vs_player_id,
            season=season or "2023-24"
        )
        return comparison.get_data_frames()[0]
    def get_advanced_player_stats(self, player_id: int, season: str) -> Dict[str, pd.DataFrame]:
        self._rate_limit()
        dashboard = playerdashboardbyyearoveryear.PlayerDashboardByYearOverYear(
            player_id=player_id,
            season=season
        )
        return {
            'general': dashboard.get_data_frames()[0],
            'shooting': dashboard.get_data_frames()[1] if len(dashboard.get_data_frames()) > 1 else pd.DataFrame()
        }
    def search_games(self, player_id: Optional[int] = None, team_id: Optional[int] = None,
                    season: Optional[str] = None, season_type: str = "Regular Season") -> pd.DataFrame:
        self._rate_limit()
        finder = leaguegamefinder.LeagueGameFinder(
            player_id_nullable=player_id,
            team_id_nullable=team_id,
            season_nullable=season,
            season_type_nullable=season_type
        )
        return finder.get_data_frames()[0]
    def get_current_season(self) -> str:
        return "2023-24"
    def get_available_seasons(self, start_year: int = 1996) -> List[str]:
        current_year = 2024
        seasons = []
        for year in range(start_year, current_year):
            seasons.append(f"{year}-{str(year + 1)[2:]}")
        return seasons