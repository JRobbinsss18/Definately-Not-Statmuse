from typing import Dict, List, Tuple, Optional, Any
from .nba_api_client import NBAAPIClient
from .llm_processor import LLMProcessor, QueryParsing
from .ml_predictor import MLPredictor
from .visualizations import NBAVisualizer
from .constants import *
import plotly.graph_objects as go

class QueryHandler:
    def __init__(self):
        self.nba_client = NBAAPIClient()
        self.llm_processor = LLMProcessor()
        self.ml_predictor = MLPredictor()
        self.visualizer = NBAVisualizer()
        
    def process_user_query(self, user_query: str, progress_callback=None) -> Dict[str, Any]:
        try:
            if progress_callback:
                progress_callback("Understanding your question", STEP_PARSE_QUERY, TOTAL_PROGRESS_STEPS)
            
            query_details = self.llm_processor.parse_query(user_query)
            
            if progress_callback:
                progress_callback("Getting NBA data", STEP_FETCH_DATA, TOTAL_PROGRESS_STEPS)
            
            if query_details.query_type == "compare_players":
                result = self._handle_player_comparison(query_details, progress_callback)
            elif query_details.query_type == "compare_teams":
                result = self._handle_team_comparison(query_details, progress_callback)
            elif query_details.query_type == "explain_comparison":
                result = self._handle_explanation_comparison(query_details, progress_callback)
            elif query_details.query_type == "explain_analysis":
                result = self._handle_explanation_analysis(query_details, progress_callback)
            elif query_details.query_type == "analyze_player":
                result = self._handle_player_analysis(query_details, progress_callback)
            elif query_details.query_type == "analyze_team":
                result = self._handle_team_analysis(query_details, progress_callback)
            elif query_details.query_type in ["predict_player", "predict_team"]:
                result = self._handle_prediction_request(query_details, progress_callback)
            else:
                result = self._handle_general_analysis(query_details, progress_callback)
            
            if progress_callback:
                progress_callback("Writing analysis", STEP_GENERATE_ANALYSIS, TOTAL_PROGRESS_STEPS)
            
            result['analysis'] = self.llm_processor.generate_analysis_response(
                user_query, result.get('data', {}), query_details
            )
            
            result['query_type'] = query_details.query_type
            result['parsed_query'] = query_details
            result['original_query'] = user_query
            
            return result
            
        except Exception as e:
            return {
                'error': f"Something went wrong: {str(e)}",
                'analysis': "Sorry, I couldn't process your question.",
                'visualizations': [],
                'data': {}
            }
    
    def _handle_player_comparison(self, query_details: QueryParsing, progress_callback=None) -> Dict[str, Any]:
        if len(query_details.players) < 2:
            return self._create_error_response("I need two players to compare")
        
        player_one_name = self.nba_client.resolve_player_name(query_details.players[0])
        player_two_name = self.nba_client.resolve_player_name(query_details.players[1])
        
        player_one_id = self.nba_client.get_player_id(player_one_name)
        player_two_id = self.nba_client.get_player_id(player_two_name)
        
        if not player_one_id or not player_two_id:
            return self._create_error_response("Couldn't find one or both players")
        
        actual_player_one_name = player_one_name
        actual_player_two_name = player_two_name
        
        try:
            from nba_api.stats.static import players as nba_players
            all_players = nba_players.get_players()
            for player in all_players:
                if player['id'] == player_one_id:
                    actual_player_one_name = player['full_name']
                elif player['id'] == player_two_id:
                    actual_player_two_name = player['full_name']
        except:
            pass
        
        if progress_callback:
            progress_callback("Getting player stats", STEP_FETCH_STATS, TOTAL_PROGRESS_STEPS)
        
        player_one_stats = self.nba_client.get_player_career_stats(player_one_id)
        player_two_stats = self.nba_client.get_player_career_stats(player_two_id)
        
        if progress_callback:
            progress_callback("Creating charts", STEP_CREATE_VIZ, TOTAL_PROGRESS_STEPS)
        
        visualizations = []
        
        if query_details.attributes:
            requested_stats = query_details.attributes
            stats_for_radar = requested_stats
        else:
            requested_stats = ['PTS', 'REB', 'AST', 'FG_PCT', 'STL', 'BLK']
            stats_for_radar = ['PTS', 'REB', 'AST', 'FG_PCT', 'STL', 'BLK']
        
        comparison_chart = self.visualizer.create_player_comparison_chart(
            player_one_stats, player_two_stats, actual_player_one_name, actual_player_two_name,
            stats_for_radar
        )
        visualizations.append({
            'chart': comparison_chart, 
            'title': f"{actual_player_one_name} vs {actual_player_two_name} Performance Radar"
        })
        
        stat_display_names = {
            'PTS': 'Points', 'PPG': 'Points Per Game',
            'REB': 'Rebounds', 'RPG': 'Rebounds Per Game', 
            'AST': 'Assists', 'APG': 'Assists Per Game',
            'FG_PCT': 'Field Goal %', 'STL': 'Steals', 'BLK': 'Blocks',
            'FG3_PCT': '3-Point %', 'FT_PCT': 'Free Throw %',
            'TOV': 'Turnovers', 'MIN': 'Minutes'
        }
        
        for stat in requested_stats:
            viz_stat = stat
            if stat == 'PPG':
                viz_stat = 'PTS'
            elif stat == 'RPG': 
                viz_stat = 'REB'
            elif stat == 'APG':
                viz_stat = 'AST'
            
            season_chart = self.visualizer.create_season_aligned_comparison(
                player_one_stats, player_two_stats, actual_player_one_name, actual_player_two_name, viz_stat
            )
            
            display_name = stat_display_names.get(stat, stat)
            visualizations.append({
                'chart': season_chart,
                'title': f"{actual_player_one_name} vs {actual_player_two_name} {display_name} by Season"
            })
        
        if progress_callback:
            progress_callback("Analyzing data", STEP_PROCESS_DATA, TOTAL_PROGRESS_STEPS)
        
        return {
            'data': {
                'player1_stats': player_one_stats,
                'player2_stats': player_two_stats,
                'player1_name': actual_player_one_name,
                'player2_name': actual_player_two_name
            },
            'visualizations': visualizations
        }
    
    def _handle_explanation_comparison(self, query_details: QueryParsing, progress_callback=None) -> Dict[str, Any]:
        if len(query_details.players) < 2:
            return self._create_error_response("I need two players to explain the comparison")
        
        comparison_result = self._handle_player_comparison(query_details, progress_callback)
        
        if 'error' in comparison_result:
            return comparison_result
        
        comparison_result['query_type'] = 'explain_comparison'
        comparison_result['explanation_type'] = 'player_comparison'
        
        return comparison_result
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        return {
            'error': error_message,
            'analysis': f"Error: {error_message}",
            'visualizations': [],
            'data': {}
        }
    
    def export_to_pdf(self, query_result: Dict[str, Any]) -> str:
        try:
            from .pdf_exporter import NBAReportExporter
            exporter = NBAReportExporter()
            
            query = query_result.get('original_query', 'NBA Query')
            analysis = query_result.get('analysis', 'No analysis available')
            query_type = query_result.get('query_type', 'general')
            data = query_result.get('data', {})
            
            pdf_path = exporter.export_query_report(query, analysis, query_type, data)
            return pdf_path
            
        except Exception as e:
            raise Exception(f"PDF export failed: {str(e)}")