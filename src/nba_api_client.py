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
        # Use reverse search strategy with confidence scoring for all searches
        # This ensures proper last name matching and disambiguation
        return self._reverse_search_player(player_name)
    
    def _reverse_search_player(self, player_name: str) -> Optional[int]:
        """
        Reverse search: Start with perfect last name matches, build confidence levels
        """
        # First try perfect exact match using NBA API (for cases like "LeBron James")
        player_dict = players.find_players_by_full_name(player_name)
        if player_dict and len(player_dict) == 1:  # Only if exactly one perfect match
            return player_dict[0]['id']
        
        all_players = players.get_players()
        player_name_lower = player_name.lower().strip()
        name_parts = player_name_lower.split()
        
        if not name_parts:
            return None
            
        # Extract last name - this MUST match perfectly
        last_name = name_parts[-1]
        
        # Step 1: Find all players with PERFECT last name match
        last_name_candidates = []
        for player in all_players:
            player_parts = player['full_name'].lower().split()
            if len(player_parts) >= 2 and last_name == player_parts[-1]:
                last_name_candidates.append(player)
        
        if not last_name_candidates:
            return None  # No perfect last name match = no results
        
        # Step 2: Score confidence for each candidate
        scored_candidates = []
        for player in last_name_candidates:
            confidence = self._calculate_player_confidence(player, name_parts)
            scored_candidates.append((player, confidence))
        
        # Step 3: Sort by confidence (highest first), then by recency (ID)
        scored_candidates.sort(key=lambda x: (x[1], x[0]['id']), reverse=True)
        
        # Step 4: Check for disambiguation needed
        return self._handle_player_disambiguation(scored_candidates, player_name)
    
    def _calculate_player_confidence(self, player: dict, search_name_parts: list) -> float:
        """
        Calculate confidence score for a player match
        Perfect last name match is prerequisite (already filtered)
        """
        player_parts = player['full_name'].lower().split()
        confidence = 0.0
        
        # Base confidence for perfect last name match
        confidence += 50.0
        
        # Bonus for matching first name
        if len(search_name_parts) >= 2 and len(player_parts) >= 2:
            search_first = search_name_parts[0]
            player_first = player_parts[0]
            
            if search_first == player_first:
                confidence += 40.0  # Perfect first name match
            elif search_first in player_first or player_first in search_first:
                confidence += 20.0  # Partial first name match
        
        # Bonus for matching middle names/initials
        if len(search_name_parts) > 2 and len(player_parts) > 2:
            search_middle = search_name_parts[1:-1]  # Everything except first and last
            player_middle = player_parts[1:-1]
            
            for s_mid in search_middle:
                for p_mid in player_middle:
                    if s_mid == p_mid:
                        confidence += 10.0
        
        # Career activity bonus - prioritize players who are more likely to be actively discussed
        # This is tricky because higher ID doesn't always mean "better" player
        # Instead, we'll use ID ranges to identify likely active vs historical players
        player_id = player['id']
        
        if player_id > 1000000:  # Recent draft classes (2017+)
            activity_bonus = 5.0
        elif player_id > 200000:  # Modern era players (2000s+)  
            activity_bonus = 8.0  # Sweet spot for current stars
        elif player_id > 100000:  # 1990s-2000s players
            activity_bonus = 3.0
        else:  # Historical players
            activity_bonus = 1.0
        
        confidence += activity_bonus
        
        return confidence
    
    def _handle_player_disambiguation(self, scored_candidates: list, original_query: str) -> Optional[int]:
        """
        Handle cases where multiple high-confidence matches exist
        """
        if not scored_candidates:
            return None
            
        # If only one candidate, return it
        if len(scored_candidates) == 1:
            return scored_candidates[0][0]['id']
        
        # Get the top candidates
        best_confidence = scored_candidates[0][1]
        high_confidence_candidates = []
        
        # Consider candidates within 5 points of the best as "high confidence"
        for player, confidence in scored_candidates:
            if confidence >= best_confidence - 5.0:
                high_confidence_candidates.append((player, confidence))
            else:
                break  # List is sorted, so we can break here
        
        # If there's a clear winner (>10 point gap to second place), return it
        if len(high_confidence_candidates) == 1 or \
           (len(high_confidence_candidates) >= 2 and 
            high_confidence_candidates[0][1] - high_confidence_candidates[1][1] > 10.0):
            return high_confidence_candidates[0][0]['id']
        
        # Multiple high-confidence matches - need user disambiguation
        return self._request_user_disambiguation(high_confidence_candidates, original_query)
    
    def _request_user_disambiguation(self, candidates: list, original_query: str) -> Optional[int]:
        """
        Request user to choose between multiple high-confidence player matches
        This will be handled by returning None and letting the error handler present options
        """
        # Store disambiguation data for the error handler to use
        self._disambiguation_candidates = candidates
        self._disambiguation_query = original_query
        return None
    
    def get_disambiguation_options(self) -> Optional[Dict]:
        """
        Get stored disambiguation options from the last search
        """
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
        """Clear stored disambiguation data"""
        self._disambiguation_candidates = None
        self._disambiguation_query = None
    
    def get_team_id(self, team_name: str) -> Optional[int]:
        team_dict = teams.find_teams_by_full_name(team_name)
        if team_dict:
            return team_dict[0]['id']
        
        # Try partial match
        all_teams = teams.get_teams()
        for team in all_teams:
            if team_name.lower() in team['full_name'].lower() or team_name.lower() in team['nickname'].lower():
                return team['id']
        return None
    
    def resolve_player_name(self, query_name: str) -> str:
        # First try nickname mappings (only for clear nicknames, not ambiguous names)
        nickname_mappings = {
            'lebron': 'LeBron James',
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
        
        # Only use nickname mapping for clear nicknames, not common last names
        if query_lower in nickname_mappings:
            return nickname_mappings[query_lower]
        
        # For everything else, return as-is and let the reverse search handle it
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