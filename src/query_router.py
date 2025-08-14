import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from src.nba_api_client import NBAAPIClient
from src.llm_processor import LLMProcessor, QueryParsing
from src.ml_predictor import MLPredictor
from src.visualizations import NBAVisualizer
import plotly.graph_objects as go

class QueryRouter:
    def __init__(self):
        self.nba_client = NBAAPIClient()
        self.llm_processor = LLMProcessor()
        self.ml_predictor = MLPredictor()
        self.visualizer = NBAVisualizer()
        
    def process_query(self, user_query: str, progress_callback=None) -> Dict[str, Any]:
        try:
            if progress_callback:
                progress_callback("Parsing query with LLM", 1, 6)
            
            # Parse query
            query_parsing = self.llm_processor.parse_query(user_query)
            
            if progress_callback:
                progress_callback("Fetching NBA data", 2, 6)
            
            # Route to appropriate handler
            if query_parsing.query_type == "compare_players":
                result = self._handle_compare_players(query_parsing, progress_callback)
            elif query_parsing.query_type == "compare_teams":
                result = self._handle_compare_teams(query_parsing, progress_callback)
            elif query_parsing.query_type == "explain_comparison":
                result = self._handle_explain_comparison(query_parsing, progress_callback)
            elif query_parsing.query_type == "explain_analysis":
                result = self._handle_explain_analysis(query_parsing, progress_callback)
            elif query_parsing.query_type == "analyze_player":
                result = self._handle_analyze_player(query_parsing, progress_callback)
            elif query_parsing.query_type == "analyze_team":
                result = self._handle_analyze_team(query_parsing, progress_callback)
            elif query_parsing.query_type in ["predict_player", "predict_team"]:
                result = self._handle_prediction(query_parsing, progress_callback)
            else:
                result = self._handle_general_analysis(query_parsing, progress_callback)
            
            if progress_callback:
                progress_callback("Generating final analysis", 6, 6)
            
            # Generate comprehensive response
            result['analysis'] = self.llm_processor.generate_analysis_response(
                user_query, result.get('data', {}), query_parsing
            )
            
            result['query_type'] = query_parsing.query_type
            result['parsed_query'] = query_parsing
            result['original_query'] = user_query
            
            return result
            
        except Exception as e:
            return {
                'error': f"Error processing query: {str(e)}",
                'analysis': "Unable to process the query due to a technical error.",
                'visualizations': [],
                'data': {}
            }
    
    def _handle_compare_players(self, parsing: QueryParsing, progress_callback=None) -> Dict[str, Any]:
        if len(parsing.players) < 2:
            if len(parsing.players) == 0:
                parsing.players = ["LeBron James", "Stephen Curry"]
            elif len(parsing.players) == 1:
                if "lebron" in parsing.players[0].lower():
                    parsing.players.append("Stephen Curry")
                else:
                    parsing.players.append("LeBron James")
        
        if len(parsing.players) < 2:
            return self._create_error_result("Need at least two players to compare")
        
        player1_name = self.nba_client.resolve_player_name(parsing.players[0])
        player2_name = self.nba_client.resolve_player_name(parsing.players[1])
        
        player1_id = self.nba_client.get_player_id(player1_name)
        player2_id = self.nba_client.get_player_id(player2_name)
        
        if not player1_id or not player2_id:
            return self._create_error_result("Could not find one or both players")
        
        actual_player1_name = player1_name
        actual_player2_name = player2_name
        try:
            from nba_api.stats.static import players as nba_players
            all_players = nba_players.get_players()
            for player in all_players:
                if player['id'] == player1_id:
                    actual_player1_name = player['full_name']
                elif player['id'] == player2_id:
                    actual_player2_name = player['full_name']
        except:
            pass
        
        if progress_callback:
            progress_callback("Fetching player statistics", 3, 6)
        
        # Get career stats
        player1_stats = self.nba_client.get_player_career_stats(player1_id)
        player2_stats = self.nba_client.get_player_career_stats(player2_id)
        
        if progress_callback:
            progress_callback("Creating visualizations", 4, 6)
        
        visualizations = []
        if parsing.attributes:
            requested_stats = parsing.attributes
            stats_for_radar = requested_stats
        else:
            requested_stats = ['PTS', 'REB', 'AST', 'STL', 'BLK']
            stats_for_radar = ['PTS', 'REB', 'AST', 'STL', 'BLK']
        
        comparison_chart = self.visualizer.create_player_comparison_chart(
            player1_stats, player2_stats, actual_player1_name, actual_player2_name,
            stats_for_radar
        )
        visualizations.append({
            'chart': comparison_chart, 
            'title': f"{actual_player1_name} vs {actual_player2_name} Statistical Radar"
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
            # Map per-game stats to season totals for visualization
            # (since historical data uses season totals)
            viz_stat = stat
            if stat == 'PPG':
                viz_stat = 'PTS'
            elif stat == 'RPG': 
                viz_stat = 'REB'
            elif stat == 'APG':
                viz_stat = 'AST'
            
            season_chart = self.visualizer.create_season_aligned_comparison(
                player1_stats, player2_stats, actual_player1_name, actual_player2_name, viz_stat
            )
            
            display_name = stat_display_names.get(stat, stat)
            visualizations.append({
                'chart': season_chart,
                'title': f"{actual_player1_name} vs {actual_player2_name} {display_name} by Season"
            })
        
        if progress_callback:
            progress_callback("Processing data analysis", 5, 6)
        
        return {
            'data': {
                'player1_stats': player1_stats,
                'player2_stats': player2_stats,
                'player1_name': actual_player1_name,
                'player2_name': actual_player2_name
            },
            'visualizations': visualizations
        }
    
    def _handle_compare_teams(self, parsing: QueryParsing, progress_callback=None) -> Dict[str, Any]:
        if len(parsing.teams) < 2:
            return self._create_error_result("Need at least two teams to compare")
        
        team1_id = self.nba_client.get_team_id(parsing.teams[0])
        team2_id = self.nba_client.get_team_id(parsing.teams[1])
        
        if not team1_id or not team2_id:
            return self._create_error_result("Could not find one or both teams")
        
        season = parsing.seasons[0] if parsing.seasons else self.nba_client.get_current_season()
        
        if progress_callback:
            progress_callback("Fetching team statistics", 3, 6)
        
        team1_stats = self.nba_client.get_team_season_stats(team1_id, season)
        team2_stats = self.nba_client.get_team_season_stats(team2_id, season)
        
        if progress_callback:
            progress_callback("Creating visualizations", 4, 6)
        
        team1_chart = self.visualizer.create_team_performance_chart(team1_stats, parsing.teams[0])
        team2_chart = self.visualizer.create_team_performance_chart(team2_stats, parsing.teams[1])
        
        return {
            'data': {
                'team1_stats': team1_stats,
                'team2_stats': team2_stats,
                'season': season
            },
            'visualizations': [
                {'chart': team1_chart, 'title': f"{parsing.teams[0]} Performance"},
                {'chart': team2_chart, 'title': f"{parsing.teams[1]} Performance"}
            ]
        }
    
    def _handle_analyze_player(self, parsing: QueryParsing, progress_callback=None) -> Dict[str, Any]:
        if not parsing.players:
            return self._create_error_result("No player specified for analysis")
        
        player_name = self.nba_client.resolve_player_name(parsing.players[0])
        player_id = self.nba_client.get_player_id(player_name)
        
        if not player_id:
            # Check if we need disambiguation
            disambiguation_options = self.nba_client.get_disambiguation_options()
            if disambiguation_options:
                return self._create_disambiguation_result(disambiguation_options)
            return self._create_error_result(f"Could not find player: {player_name}")
        
        # Get actual player name from NBA API for validation
        actual_player_name = player_name
        try:
            from nba_api.stats.static import players as nba_players
            all_players = nba_players.get_players()
            for player in all_players:
                if player['id'] == player_id:
                    actual_player_name = player['full_name']
                    break
        except:
            pass
        
        if progress_callback:
            progress_callback("Fetching player data", 3, 6)
        
        career_stats = self.nba_client.get_player_career_stats(player_id)
        player_info = self.nba_client.get_player_info(player_id)
        
        season_stats = None
        if parsing.seasons:
            season_stats = self.nba_client.get_player_season_stats(player_id, parsing.seasons[0])
        
        if progress_callback:
            progress_callback("Creating visualizations", 4, 6)
        
        visualizations = []
        key_stats = ['PTS', 'REB', 'AST']
        stat_names = {'PTS': 'Points', 'REB': 'Rebounds', 'AST': 'Assists'}
        
        if parsing.time_context == 'career_peak':
            for stat in key_stats:
                if stat in career_stats.columns:
                    chart = self.visualizer.create_career_progression_chart(career_stats, actual_player_name, stat)
                    visualizations.append({
                        'chart': chart, 
                        'title': f"{actual_player_name} {stat_names[stat]} Career Progression (Career Highs)"
                    })
                    
            if 'PTS' in career_stats.columns:
                distribution_chart = self.visualizer.create_stat_distribution_chart(career_stats, 'PTS', actual_player_name)
                visualizations.append({
                    'chart': distribution_chart, 
                    'title': f"{actual_player_name} Points Distribution (Peak Analysis)"
                })
        
        elif parsing.time_context == 'best_season':
            best_season_data = {}
            
            for stat in key_stats:
                if stat in career_stats.columns:
                    max_idx = career_stats[stat].idxmax()
                    best_value = career_stats[stat].iloc[max_idx]
                    best_year = career_stats['SEASON_ID'].iloc[max_idx] if 'SEASON_ID' in career_stats.columns else f"Season {max_idx + 1}"
                    
                    best_season_data[stat] = {
                        'value': best_value,
                        'season': best_year,
                        'index': max_idx
                    }
                    
                    chart = self.visualizer.create_career_progression_chart(career_stats, actual_player_name, stat)
                    visualizations.append({
                        'chart': chart, 
                        'title': f"{actual_player_name} {stat_names[stat]} - Best Season: {best_year} ({best_value:.1f})"
                    })
            
        
        else:
            for stat in key_stats:
                if stat in career_stats.columns:
                    chart = self.visualizer.create_career_progression_chart(career_stats, actual_player_name, stat)
                    visualizations.append({
                        'chart': chart, 
                        'title': f"{actual_player_name} {stat_names[stat]} Career Progression"
                    })
            
            if 'PTS' in career_stats.columns:
                distribution_chart = self.visualizer.create_stat_distribution_chart(career_stats, 'PTS', actual_player_name)
                visualizations.append({
                    'chart': distribution_chart, 
                    'title': f"{actual_player_name} Points Distribution"
                })
        
        return_data = {
            'career_stats': career_stats,
            'player_info': player_info,
            'season_stats': season_stats,
            'player_name': actual_player_name
        }
        
        if parsing.time_context == 'best_season' and 'best_season_data' in locals():
            return_data['best_season_data'] = best_season_data
        elif parsing.time_context == 'career_peak':
            career_peaks = {}
            for stat in key_stats:
                if stat in career_stats.columns:
                    max_idx = career_stats[stat].idxmax()
                    career_peaks[stat] = {
                        'value': career_stats[stat].iloc[max_idx],
                        'season': career_stats['SEASON_ID'].iloc[max_idx] if 'SEASON_ID' in career_stats.columns else f"Season {max_idx + 1}",
                        'index': max_idx
                    }
            return_data['career_peaks'] = career_peaks
        
        return {
            'data': return_data,
            'visualizations': visualizations
        }
    
    def _handle_analyze_team(self, parsing: QueryParsing, progress_callback=None) -> Dict[str, Any]:
        if not parsing.teams:
            return self._create_error_result("No team specified for analysis")
        
        team_id = self.nba_client.get_team_id(parsing.teams[0])
        if not team_id:
            return self._create_error_result(f"Could not find team: {parsing.teams[0]}")
        
        if progress_callback:
            progress_callback("Fetching team data", 3, 6)
        
        all_seasons = self.nba_client.get_team_year_by_year(team_id)
        
        if parsing.time_context == 'all_time':
            team_stats = all_seasons
            season = "All Time"
        elif parsing.time_context == 'championship_years':
            if 'NBA_FINALS_APPEARANCE' in all_seasons.columns:
                championship_seasons = all_seasons[all_seasons['NBA_FINALS_APPEARANCE'].str.contains('CHAMPION', na=False)]
                team_stats = championship_seasons if not championship_seasons.empty else all_seasons[all_seasons['NBA_FINALS_APPEARANCE'] == 'LEAGUE CHAMPION']
            else:
                top_seasons = all_seasons.nlargest(5, 'WIN_PCT') if 'WIN_PCT' in all_seasons.columns else all_seasons
                team_stats = top_seasons
            season = "Championship Years"
        else:
            season = parsing.seasons[0] if parsing.seasons else self.nba_client.get_current_season()
            
            if not all_seasons.empty and 'YEAR' in all_seasons.columns:
                team_stats = all_seasons[all_seasons['YEAR'] == season]
                
                if team_stats.empty:
                    # Try to match partial season formats
                    season_variants = [
                        season,
                        season.split('-')[0] if '-' in season else season,  # "2020-21" -> "2020"
                        f"{season}-{str(int(season)+1)[2:]}" if season.isdigit() else season  # "2020" -> "2020-21"
                    ]
                    
                    for variant in season_variants:
                        if variant != season:  # Avoid duplicate check
                            team_stats = all_seasons[all_seasons['YEAR'] == variant]
                            if not team_stats.empty:
                                season = variant  # Update season to the found variant
                                break
            else:
                team_stats = pd.DataFrame()  # Empty fallback
        
        if progress_callback:
            progress_callback("Creating visualizations", 4, 6)
        
        team_chart = self.visualizer.create_team_performance_chart(team_stats, parsing.teams[0])
        
        return {
            'data': {
                'team_stats': team_stats,
                'team_name': parsing.teams[0],
                'season': season
            },
            'visualizations': [
                {'chart': team_chart, 'title': f"{parsing.teams[0]} Season Analysis"}
            ]
        }
    
    def _handle_prediction(self, parsing: QueryParsing, progress_callback=None) -> Dict[str, Any]:
        if not parsing.players:
            return self._create_error_result("No player specified for prediction")
        
        player_name = self.nba_client.resolve_player_name(parsing.players[0])
        player_id = self.nba_client.get_player_id(player_name)
        
        if not player_id:
            # Check if we need disambiguation
            disambiguation_options = self.nba_client.get_disambiguation_options()
            if disambiguation_options:
                return self._create_disambiguation_result(disambiguation_options)
            return self._create_error_result(f"Could not find player: {player_name}")
        
        # Get actual player name from NBA API for validation
        actual_player_name = player_name
        try:
            from nba_api.stats.static import players as nba_players
            all_players = nba_players.get_players()
            for player in all_players:
                if player['id'] == player_id:
                    actual_player_name = player['full_name']
                    break
        except:
            pass
        
        if progress_callback:
            progress_callback("Fetching historical data", 3, 6)
        
        career_stats = self.nba_client.get_player_career_stats(player_id)
        
        if progress_callback:
            progress_callback("Running ML predictions", 4, 6)
        
        # Determine target stats based on user query
        if parsing.attributes:
            # User specified specific stats
            target_stats = parsing.attributes
        else:
            # Default to key stats for general prediction
            target_stats = ['PTS', 'REB', 'AST']
            
        # Run predictions with appropriate time horizon
        if parsing.time_context.startswith('next_') and parsing.time_context != 'next_season':
            # Multi-season predictions
            num_seasons = self._extract_season_count(parsing.time_context)
            predictions = self.ml_predictor.predict_multiple_seasons(career_stats, target_stats, num_seasons)
        else:
            # Single season prediction
            predictions = self.ml_predictor.predict_next_season(career_stats, target_stats)
        
        if progress_callback:
            progress_callback("Creating prediction charts", 5, 6)
        
        # Create targeted prediction visualizations
        visualizations = []
        stat_names = {
            'PTS': 'Points', 'PPG': 'PPG',
            'REB': 'Rebounds', 'RPG': 'RPG', 
            'AST': 'Assists', 'APG': 'APG',
            'FG_PCT': 'Field Goal %', 'STL': 'Steals', 'BLK': 'Blocks',
            'FG3_PCT': '3-Point %', 'FT_PCT': 'Free Throw %', 
            'TOV': 'Turnovers', 'MIN': 'Minutes'
        }
        
        # Create specific visualization based on query context
        for stat in predictions.keys():
            stat_display = stat_names.get(stat, stat)
            pred_data = predictions[stat]
            
            # Determine chart type based on time context
            if parsing.time_context.startswith('next_') and parsing.time_context != 'next_season':
                # Multi-season prediction chart
                pred_chart = self.visualizer.create_multi_season_prediction_chart(
                    career_stats, predictions, actual_player_name, stat, 
                    self._extract_season_count(parsing.time_context)
                )
                title = f"{actual_player_name} {stat_display} - {parsing.time_context.replace('_', ' ').title()} Projection"
            else:
                # Single season prediction
                pred_chart = self.visualizer.create_prediction_chart(
                    career_stats, predictions, actual_player_name, stat
                )
                title = f"{actual_player_name} {stat_display} - Next Season Prediction"
            
            if 'best_model' in pred_data:
                title += f" ({pred_data['best_model'].replace('_', ' ').title()})"
            
            visualizations.append({
                'chart': pred_chart,
                'title': title
            })
        
        return {
            'data': {
                'career_stats': career_stats,
                'predictions': predictions,
                'player_name': actual_player_name
            },
            'visualizations': visualizations,
            'prediction_summary': self.ml_predictor.get_prediction_summary(predictions)
        }
    
    def _handle_general_analysis(self, parsing: QueryParsing, progress_callback=None) -> Dict[str, Any]:
        # Fallback for queries that don't fit other categories
        return self._handle_analyze_player(parsing, progress_callback)
    
    def _extract_season_count(self, time_context: str) -> int:
        if 'next_' in time_context and '_seasons' in time_context:
            try:
                parts = time_context.split('_')
                for part in parts:
                    if part.isdigit():
                        return int(part)
            except:
                pass
        return 1
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        return {
            'error': error_message,
            'analysis': f"Error: {error_message}",
            'visualizations': [],
            'data': {}
        }
    
    def _create_disambiguation_result(self, disambiguation_options: Dict) -> Dict[str, Any]:
        query = disambiguation_options['query']
        options = disambiguation_options['options']
        
        analysis = f"Multiple players found for '{query}'. Please specify which player you meant:\\n\\n"
        for i, option in enumerate(options, 1):
            analysis += f"{i}. **{option['name']}** (Confidence: {option['confidence']})\\n"
        
        analysis += "\\nPlease try your query again with the full player name."
        
        return {
            'error': None,
            'analysis': analysis,
            'visualizations': [],
            'data': {'disambiguation_options': disambiguation_options},
            'requires_disambiguation': True
        }
    
    def _handle_explain_comparison(self, parsing: QueryParsing, progress_callback=None) -> Dict[str, Any]:
        if len(parsing.players) < 2:
            return self._create_error_result("Need two players to explain comparison")
        comparison_result = self._handle_compare_players(parsing, progress_callback)
        
        if 'error' in comparison_result:
            return comparison_result
        
        # Enhance the result with explanation-focused analysis
        comparison_result['query_type'] = 'explain_comparison'
        comparison_result['explanation_type'] = 'player_comparison'
        
        return comparison_result
    
    def _handle_explain_analysis(self, parsing: QueryParsing, progress_callback=None) -> Dict[str, Any]:
        if parsing.players:
            analysis_result = self._handle_analyze_player(parsing, progress_callback)
            if 'error' not in analysis_result:
                analysis_result['query_type'] = 'explain_analysis'
                analysis_result['explanation_type'] = 'player_analysis'
            return analysis_result
        elif parsing.teams:
            # Team explanation
            analysis_result = self._handle_analyze_team(parsing, progress_callback)
            if 'error' not in analysis_result:
                analysis_result['query_type'] = 'explain_analysis'
                analysis_result['explanation_type'] = 'team_analysis'
            return analysis_result
        else:
            return self._create_error_result("No specific player or team mentioned for explanation")
    
    def export_to_pdf(self, query_result: Dict[str, Any]) -> str:
        try:
            from src.pdf_exporter import NBAReportExporter
            exporter = NBAReportExporter()
            
            query = query_result.get('original_query', 'NBA Query')
            analysis = query_result.get('analysis', 'No analysis available')
            query_type = query_result.get('query_type', 'general')
            data = query_result.get('data', {})
            
            pdf_path = exporter.export_query_report(query, analysis, query_type, data)
            return pdf_path
            
        except Exception as e:
            raise Exception(f"Failed to export PDF: {str(e)}")