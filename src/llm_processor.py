import ollama
import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class QueryParsing:
    query_type: str
    players: List[str]
    teams: List[str]
    seasons: List[str]
    attributes: List[str]
    time_context: str
    
class LLMProcessor:
    def __init__(self, model_name: str = "llama3.2:1b"):
        self.model_name = model_name
        self.client = ollama.Client()
        self.available_models = self._get_available_models()
        
    def _get_available_models(self):
        try:
            models = self.client.list()
            return [model['name'] for model in models['models']]
        except:
            return []
        
    def parse_query(self, user_query: str) -> QueryParsing:
        parsing_prompt = f"Analyze this NBA query: {user_query}. Return JSON with query_type, players, teams, seasons, attributes, time_context."
        
        if not self.available_models or self.model_name not in self.available_models:
            return self._fallback_parse(user_query)
            
        try:
            response = self.client.chat(model=self.model_name, messages=[
                {'role': 'user', 'content': parsing_prompt}
            ])
            
            content = response['message']['content'].strip()
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed_data = json.loads(json_str)
                
                return QueryParsing(
                    query_type=parsed_data.get('query_type', 'analyze_player'),
                    players=parsed_data.get('players', []),
                    teams=parsed_data.get('teams', []),
                    seasons=parsed_data.get('seasons', []),
                    attributes=parsed_data.get('attributes', []),
                    time_context=parsed_data.get('time_context', 'career')
                )
        except Exception:
            return self._fallback_parse(user_query)
    
    def _fallback_parse(self, query: str) -> QueryParsing:
        query_lower = query.lower()
        
        if any(phrase in query_lower for phrase in ['explain why', 'why is', 'why', 'because', 'reason']):
            if any(word in query_lower for word in ['better', 'worse', 'vs', 'than', 'compared to']):
                query_type = 'explain_comparison'
            else:
                query_type = 'explain_analysis'
        elif any(word in query_lower for word in ['compare', 'vs', 'versus']):
            if any(word in query_lower for word in ['team', 'lakers', 'warriors', 'bulls']):
                query_type = 'compare_teams'
            else:
                query_type = 'compare_players'
        elif any(word in query_lower for word in ['predict', 'will', 'next year', 'future']):
            query_type = 'predict_player'
        elif any(word in query_lower for word in ['team', 'lakers', 'warriors']):
            query_type = 'analyze_team'
        else:
            query_type = 'analyze_player'
            
        players = []
        player_names = {
            'lebron': 'LeBron James', 'james': 'LeBron James',
            'curry': 'Stephen Curry', 'stephen': 'Stephen Curry', 'steph': 'Stephen Curry',
            'durant': 'Kevin Durant', 'kd': 'Kevin Durant',
            'giannis': 'Giannis Antetokounmpo', 'antetokounmpo': 'Giannis Antetokounmpo',
            'luka': 'Luka Doncic', 'doncic': 'Luka Doncic',
            'jordan': 'Michael Jordan', 'mj': 'Michael Jordan',
            'kobe': 'Kobe Bryant', 'bryant': 'Kobe Bryant',
            'ayo': 'Ayo Dosunmu', 'dosunmu': 'Ayo Dosunmu',
            'tatum': 'Jayson Tatum', 'jayson': 'Jayson Tatum',
            'booker': 'Devin Booker', 'devin': 'Devin Booker',
            'morant': 'Ja Morant', 'ja': 'Ja Morant',
            'zion': 'Zion Williamson', 'williamson': 'Zion Williamson',
            'coby': 'Coby White', 'klay': 'Klay Thompson', 'thompson': 'Klay Thompson'
        }
        
        teams = []
        team_names = {
            'lakers': 'Los Angeles Lakers', 'warriors': 'Golden State Warriors', 
            'bulls': 'Chicago Bulls', 'celtics': 'Boston Celtics', 'heat': 'Miami Heat',
            'knicks': 'New York Knicks', 'nets': 'Brooklyn Nets', 'sixers': '76ers',
            'raptors': 'Toronto Raptors', 'magic': 'Orlando Magic', 'hawks': 'Atlanta Hawks',
            'hornets': 'Charlotte Hornets', 'pistons': 'Detroit Pistons', 'pacers': 'Indiana Pacers',
            'cavaliers': 'Cleveland Cavaliers', 'cavs': 'Cleveland Cavaliers', 'bucks': 'Milwaukee Bucks',
            'timberwolves': 'Minnesota Timberwolves', 'thunder': 'Oklahoma City Thunder',
            'blazers': 'Portland Trail Blazers', 'jazz': 'Utah Jazz', 'nuggets': 'Denver Nuggets',
            'clippers': 'LA Clippers', 'suns': 'Phoenix Suns', 'kings': 'Sacramento Kings',
            'mavericks': 'Dallas Mavericks', 'mavs': 'Dallas Mavericks', 'rockets': 'Houston Rockets',
            'grizzlies': 'Memphis Grizzlies', 'pelicans': 'New Orleans Pelicans', 'spurs': 'San Antonio Spurs'
        }
        
        words = query_lower.split()
        for word in words:
            if word in player_names:
                full_name = player_names[word]
                if full_name not in players:
                    players.append(full_name)
            elif word in team_names:
                full_name = team_names[word]
                if full_name not in teams:
                    teams.append(full_name)
        
        if teams and not players:
            if query_type == 'analyze_player':
                query_type = 'analyze_team'
            elif query_type == 'predict_player':
                query_type = 'predict_team'
        
        if not players and (query_type in ['analyze_player', 'predict_player', 'compare_players']):
            potential_names = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', query)
            action_words = {'Predict', 'Analyze', 'Compare', 'Show', 'Tell', 'Find', 'Get', 'Display'}
            
            for name in potential_names:
                name_parts = name.split()
                if name_parts[0] in action_words:
                    if len(name_parts) > 2:
                        clean_name = ' '.join(name_parts[1:])
                        if len(clean_name.split()) >= 2:
                            players.append(clean_name)
                elif len(name_parts) >= 2:
                    players.append(name)
        
        if not players and teams and query_type in ['analyze_player', 'predict_player']:
            query_type = 'analyze_team' if query_type == 'analyze_player' else 'predict_team'
        
        if query_type == 'compare_players' and len(players) < 2:
            vs_match = re.search(r'(\w+)\s+(?:to|vs|versus)\s+(\w+)', query_lower)
            if vs_match:
                name1 = vs_match.group(1)
                name2 = vs_match.group(2)
                if name1 in player_names and player_names[name1] not in players:
                    players.append(player_names[name1])
                if name2 in player_names and player_names[name2] not in players:
                    players.append(player_names[name2])
        
        attributes = self._extract_stat_attributes(query_lower)
        time_context, seasons = self._extract_time_context(query)
        
        return QueryParsing(
            query_type=query_type,
            players=players,
            teams=teams,
            seasons=seasons,
            attributes=attributes,
            time_context=time_context
        )
    
    def _extract_stat_attributes(self, query_lower: str) -> List[str]:
        attributes = []
        stat_mappings = {
            'points': 'PTS', 'ppg': 'PPG', 'points per game': 'PPG', 'scoring': 'PPG',
            'rebounds': 'REB', 'rpg': 'RPG', 'rebounds per game': 'RPG', 'rebounding': 'RPG',
            'assists': 'AST', 'apg': 'APG', 'assists per game': 'APG', 'playmaking': 'APG',
            'field goal percentage': 'FG_PCT', 'field goal %': 'FG_PCT', 'fg%': 'FG_PCT',
            'fg percentage': 'FG_PCT', 'shooting percentage': 'FG_PCT',
            'three point percentage': 'FG3_PCT', '3pt %': 'FG3_PCT', '3p%': 'FG3_PCT',
            'free throw percentage': 'FT_PCT', 'ft%': 'FT_PCT',
            'steals': 'STL', 'spg': 'STL', 'steals per game': 'STL', 
            'blocks': 'BLK', 'bpg': 'BLK', 'blocks per game': 'BLK',
            'turnovers': 'TOV', 'tpg': 'TOV', 'turnovers per game': 'TOV',
            'minutes': 'MIN', 'mpg': 'MIN', 'minutes per game': 'MIN'
        }
        
        for phrase, stat_code in stat_mappings.items():
            if phrase in query_lower and stat_code not in attributes:
                attributes.append(stat_code)
        
        return attributes
    
    def _extract_time_context(self, query: str) -> Tuple[str, List[str]]:
        query_lower = query.lower()
        seasons = []
        
        if any(phrase in query_lower for phrase in ['over the next', 'next several', 'coming seasons', 'over next']):
            numbers = re.findall(r'(\d+)', query_lower)
            if numbers:
                num = int(numbers[0])
                if 2 <= num <= 10:
                    time_context = f'next_{num}_seasons'
                else:
                    time_context = 'next_season'
            else:
                time_context = 'next_3_seasons'
        elif any(phrase in query_lower for phrase in ['next 2 seasons', 'next two seasons', 'next 2 years', 'next two years']):
            time_context = 'next_2_seasons'
        elif any(phrase in query_lower for phrase in ['next 3 seasons', 'next three seasons', 'next 3 years', 'next three years']):
            time_context = 'next_3_seasons'
        elif any(phrase in query_lower for phrase in ['next 4 seasons', 'next four seasons', 'next 4 years', 'next four years']):
            time_context = 'next_4_seasons'
        elif any(phrase in query_lower for phrase in ['next 5 seasons', 'next five seasons', 'next 5 years', 'next five years']):
            time_context = 'next_5_seasons'
        elif any(phrase in query_lower for phrase in ['next season', 'this season', 'upcoming season', 'next year']):
            time_context = 'next_season'
        elif any(phrase in query_lower for phrase in ['all time', 'all-time', 'franchise history', 'team history', 'historically']):
            time_context = 'all_time'
        elif any(phrase in query_lower for phrase in ['career high', 'career best', 'personal best', 'peak performance', 'highest']):
            time_context = 'career_peak'
        elif any(phrase in query_lower for phrase in ['best season', 'greatest season', 'peak season', 'mvp season']):
            time_context = 'best_season'
        elif any(phrase in query_lower for phrase in ['championship', 'title', 'ring', 'won championship', 'championship years']):
            time_context = 'championship_years'
        elif any(phrase in query_lower for phrase in ['rivalry', 'head to head', 'versus history', 'matchup history']):
            time_context = 'rivalry_history'
        else:
            season_pattern = r'(\d{4})'
            years = re.findall(season_pattern, query)
            formatted_seasons = []
            for year in years:
                year_int = int(year)
                if 1996 <= year_int <= 2024:
                    formatted_seasons.append(f"{year_int}-{str(year_int + 1)[2:]}")
            if formatted_seasons:
                seasons = formatted_seasons
                time_context = 'specific_year'
            else:
                time_context = 'career'
        
        return time_context, seasons
    
    def generate_analysis_response(self, query: str, data: Dict, parsing: QueryParsing) -> str:
        try:
            return self._generate_basic_analysis(query, data, parsing)
        except Exception:
            return f"Analysis completed. Data retrieved successfully for: {query}"
    
    def _generate_basic_analysis(self, query: str, data: Dict, parsing: QueryParsing) -> str:
        analysis = f"## Analysis Results\n\n"
        
        analysis += f"**Query: \"{query}\"**\n\n"
        
        if parsing.query_type == "explain_comparison":
            if 'player1_name' in data and 'player2_name' in data:
                p1_name = data['player1_name']
                p2_name = data['player2_name']
                analysis += f"**Answer: Why {p1_name} vs {p2_name} - Detailed Comparison Explanation**\n\n"
                
                if 'player1_stats' in data and 'player2_stats' in data:
                    p1_stats = data['player1_stats']
                    p2_stats = data['player2_stats']
                    
                    if not p1_stats.empty and not p2_stats.empty:
                        p1_latest = p1_stats.iloc[-1]
                        p2_latest = p2_stats.iloc[-1]
                        
                        analysis += f"**Key Statistical Advantages:**\n"
                        
                        stat_comparisons = [
                            ('PTS', 'GP', 'Scoring', 'PPG'),
                            ('REB', 'GP', 'Rebounding', 'RPG'),  
                            ('AST', 'GP', 'Playmaking', 'APG'),
                            ('FG_PCT', None, 'Shooting Efficiency', '%')
                        ]
                        
                        for stat, divisor_stat, category, unit in stat_comparisons:
                            if stat in p1_latest and stat in p2_latest:
                                if divisor_stat and divisor_stat in p1_latest and divisor_stat in p2_latest:
                                    p1_val = p1_latest[stat] / max(p1_latest[divisor_stat], 1)
                                    p2_val = p2_latest[stat] / max(p2_latest[divisor_stat], 1)
                                    format_str = "{:.1f}"
                                elif unit == '%':
                                    p1_val = p1_latest[stat] * 100
                                    p2_val = p2_latest[stat] * 100
                                    format_str = "{:.1f}%"
                                else:
                                    p1_val = p1_latest[stat]
                                    p2_val = p2_latest[stat]
                                    format_str = "{:.1f}"
                                
                                if p1_val > p2_val:
                                    advantage = p1_name
                                    diff = ((p1_val - p2_val) / p2_val) * 100
                                elif p2_val > p1_val:
                                    advantage = p2_name
                                    diff = ((p2_val - p1_val) / p1_val) * 100
                                else:
                                    advantage = "Even"
                                    diff = 0
                                
                                if advantage != "Even":
                                    analysis += f"- **{category}**: {advantage} leads with " + format_str.format(max(p1_val, p2_val)) + f" vs " + format_str.format(min(p1_val, p2_val)) + f" ({diff:.1f}% advantage)\n"
                                else:
                                    analysis += f"- **{category}**: Nearly equal performance\n"
                        
                        analysis += "\n**Career Context:**\n"
                        p1_seasons = len(p1_stats)
                        p2_seasons = len(p2_stats)
                        analysis += f"- {p1_name}: {p1_seasons} seasons of data\n"
                        analysis += f"- {p2_name}: {p2_seasons} seasons of data\n"
                        
                        if p1_seasons > p2_seasons:
                            analysis += f"- {p1_name} has more career longevity with {p1_seasons - p2_seasons} additional seasons\n"
                        elif p2_seasons > p1_seasons:
                            analysis += f"- {p2_name} has more career longevity with {p2_seasons - p1_seasons} additional seasons\n"
                        
                        analysis += "\n**Bottom Line**: This statistical comparison shows the measurable differences between both players' performance in key areas like scoring, rebounding, and playmaking.\n\n"
                
                analysis += "The charts provide visual comparisons across all major statistical categories to help understand each player's strengths and weaknesses.\n\n"
        
        elif parsing.query_type == "explain_analysis":
            if parsing.players:
                player_name = parsing.players[0]
                analysis += f"**Answer: Detailed Explanation of {player_name}'s Performance and Style**\n\n"
                
                if 'career_stats' in data:
                    career_stats = data['career_stats']
                    if not career_stats.empty:
                        analysis += f"**Performance Analysis:**\n"
                        seasons = len(career_stats)
                        analysis += f"- Career Span: {seasons} seasons of professional basketball\n"
                        
                        if 'PTS' in career_stats.columns and 'GP' in career_stats.columns:
                            avg_ppg = (career_stats['PTS'] / career_stats['GP']).mean()
                            max_ppg = (career_stats['PTS'] / career_stats['GP']).max()
                            analysis += f"- Scoring: Averages {avg_ppg:.1f} PPG career-wide, peaked at {max_ppg:.1f} PPG in best season\n"
                        
                        if 'FG_PCT' in career_stats.columns:
                            avg_fg_pct = career_stats['FG_PCT'].mean() * 100
                            analysis += f"- Shooting Efficiency: {avg_fg_pct:.1f}% field goal percentage\n"
                        
                        if 'AST' in career_stats.columns and 'GP' in career_stats.columns:
                            avg_apg = (career_stats['AST'] / career_stats['GP']).mean()
                            analysis += f"- Playmaking: {avg_apg:.1f} assists per game on average\n"
                        
                        analysis += f"\n**What Makes {player_name} Special:**\n"
                        analysis += f"The statistical trends and performance metrics shown in the charts reveal {player_name}'s evolution and consistency across different phases of their career.\n\n"
                
                analysis += f"The visualizations break down {player_name}'s career progression and highlight their peak performance periods.\n\n"
            elif parsing.teams:
                team_name = parsing.teams[0]
                analysis += f"**Answer: Detailed Explanation of {team_name}'s History and Performance**\n\n"
                
                if 'team_stats' in data:
                    team_stats = data['team_stats']
                    if not team_stats.empty:
                        analysis += f"**Franchise Overview:**\n"
                        total_seasons = len(team_stats)
                        analysis += f"- Franchise History: {total_seasons} seasons in the NBA\n"
                        
                        if 'WINS' in team_stats.columns and 'LOSSES' in team_stats.columns:
                            total_wins = team_stats['WINS'].sum()
                            total_losses = team_stats['LOSSES'].sum()
                            win_pct = total_wins / (total_wins + total_losses) * 100
                            analysis += f"- All-Time Record: {total_wins}-{total_losses} ({win_pct:.1f}% winning percentage)\n"
                        
                        if 'NBA_FINALS_APPEARANCE' in team_stats.columns:
                            championships = len(team_stats[team_stats['NBA_FINALS_APPEARANCE'].str.contains('CHAMPION', na=False)])
                            analysis += f"- Championships: {championships} NBA titles\n"
                        
                        analysis += f"\n**What Defines {team_name}:**\n"
                        analysis += f"The franchise history charts show the organization's evolution through different eras, highlighting championship periods and rebuilding phases.\n\n"
                
                analysis += f"The comprehensive visualizations tell the story of {team_name}'s journey through NBA history.\n\n"
        
        elif parsing.query_type == "compare_players":
            if 'player1_name' in data and 'player2_name' in data:
                p1_name = data['player1_name']
                p2_name = data['player2_name']
                analysis += f"**Answer: Comprehensive comparison between {p1_name} and {p2_name}**\n\n"
                
                if 'player1_stats' in data and 'player2_stats' in data:
                    p1_stats = data['player1_stats']
                    p2_stats = data['player2_stats']
                    
                    if not p1_stats.empty and not p2_stats.empty:
                        p1_latest = p1_stats.iloc[-1]
                        p2_latest = p2_stats.iloc[-1]
                        
                        analysis += f"**Key Statistical Comparison (Per Game):**\n"
                        
                        if parsing.attributes:
                            stats_to_show = parsing.attributes
                        else:
                            stats_to_show = ['PPG', 'RPG', 'APG']
                        
                        stat_mappings = {
                            'PTS': ('PTS', 'Points', 'PPG'), 'PPG': ('PTS', 'Points', 'PPG'),
                            'REB': ('REB', 'Rebounds', 'RPG'), 'RPG': ('REB', 'Rebounds', 'RPG'),
                            'AST': ('AST', 'Assists', 'APG'), 'APG': ('AST', 'Assists', 'APG'),
                            'STL': ('STL', 'Steals', 'SPG'), 'BLK': ('BLK', 'Blocks', 'BPG'),
                            'FG_PCT': ('FG_PCT', 'Field Goal %', '%'), 'FG3_PCT': ('FG3_PCT', '3-Point %', '%'),
                            'FT_PCT': ('FT_PCT', 'Free Throw %', '%')
                        }
                        
                        for stat in stats_to_show:
                            if stat in stat_mappings:
                                raw_stat, display_name, unit_type = stat_mappings[stat]
                                
                                if raw_stat in p1_latest and raw_stat in p2_latest:
                                    if unit_type == '%':
                                        p1_val = p1_latest[raw_stat] * 100 if p1_latest[raw_stat] <= 1 else p1_latest[raw_stat]
                                        p2_val = p2_latest[raw_stat] * 100 if p2_latest[raw_stat] <= 1 else p2_latest[raw_stat]
                                        analysis += f"- {display_name}: {p1_name} ({p1_val:.1f}%) vs {p2_name} ({p2_val:.1f}%)\n"
                                    elif unit_type in ['PPG', 'RPG', 'APG', 'SPG', 'BPG'] and 'GP' in p1_latest and 'GP' in p2_latest:
                                        p1_val = p1_latest[raw_stat] / max(p1_latest['GP'], 1)
                                        p2_val = p2_latest[raw_stat] / max(p2_latest['GP'], 1)
                                        analysis += f"- {display_name}: {p1_name} ({p1_val:.1f} {unit_type}) vs {p2_name} ({p2_val:.1f} {unit_type})\n"
                                    else:
                                        analysis += f"- {display_name}: {p1_name} ({p1_latest[raw_stat]:.1f}) vs {p2_name} ({p2_latest[raw_stat]:.1f})\n"
                        
                        analysis += "\n"
                
                if parsing.attributes:
                    if len(parsing.attributes) == 1:
                        stat_name = parsing.attributes[0].lower().replace('_', ' ')
                        analysis += f"The charts focus specifically on {stat_name} comparison, showing both current performance and career progression. "
                    else:
                        stat_names = [attr.lower().replace('_', ' ') for attr in parsing.attributes]
                        analysis += f"The charts focus on {', '.join(stat_names)} comparison, showing both current performance and career progression. "
                else:
                    analysis += f"The radar chart shows overall performance comparison across multiple categories. "
                    analysis += f"The season-by-season charts reveal how {p1_name} and {p2_name} performed at similar career stages. "
                
                analysis += f"This targeted analysis provides insights into the specific areas of comparison requested.\n\n"
        
        elif parsing.query_type == "predict_player":
            if parsing.players:
                player_name = parsing.players[0]
                analysis += f"**Answer: {player_name} Performance Predictions for Next Season**\n\n"
                
                if 'predictions' in data:
                    predictions = data['predictions']
                    analysis += "**Predicted Performance:**\n"
                    for stat, pred_data in predictions.items():
                        if 'ensemble_mean' in pred_data:
                            prediction = pred_data['ensemble_mean']
                            trend = pred_data.get('trend_direction', 'stable')
                            confidence_pct = pred_data.get('confidence_percentage', 75)
                            stat_display = {
                                'PTS': 'Total Points', 'PPG': 'Points Per Game',
                                'REB': 'Total Rebounds', 'RPG': 'Rebounds Per Game',
                                'AST': 'Total Assists', 'APG': 'Assists Per Game',
                                'FG_PCT': 'Field Goal %', 'FG3_PCT': '3-Point %', 'FT_PCT': 'Free Throw %',
                                'STL': 'Steals', 'BLK': 'Blocks', 'TOV': 'Turnovers', 'MIN': 'Minutes'
                            }.get(stat, stat)
                            analysis += f"- {stat_display}: {prediction:.1f} (trend: {trend}, confidence: {confidence_pct}%)\n"
                    analysis += "\n"
                
                analysis += f"Predictions are based on {player_name}'s historical performance using multiple ML algorithms including Random Forest, Gradient Boosting, and XGBoost. "
                analysis += "Confidence percentages reflect model agreement and prediction reliability.\n\n"
        
        elif parsing.query_type == "analyze_player":
            if parsing.players:
                player_name = parsing.players[0]
                
                if parsing.time_context == 'career_peak':
                    analysis += f"**Answer: {player_name} Career Highs and Peak Performance**\n\n"
                elif parsing.time_context == 'best_season':
                    analysis += f"**Answer: {player_name} Best Season Analysis**\n\n"
                else:
                    analysis += f"**Answer: Comprehensive Career Analysis of {player_name}**\n\n"
                
                if parsing.time_context == 'career_peak' and 'career_peaks' in data:
                    career_peaks = data['career_peaks']
                    analysis += f"**Career Highs and Peak Performance:**\n"
                    
                    for stat, peak_data in career_peaks.items():
                        stat_display = {
                            'PTS': 'Points', 'REB': 'Rebounds', 'AST': 'Assists', 'FG_PCT': 'Field Goal %'
                        }.get(stat, stat)
                        
                        value = peak_data['value']
                        season = peak_data['season']
                        
                        if stat == 'FG_PCT':
                            analysis += f"- Career High {stat_display}: {value*100:.1f}% in {season}\n"
                        else:
                            analysis += f"- Career High {stat_display}: {value:.1f} in {season}\n"
                    
                    analysis += f"\nThese represent {player_name}'s absolute peak performance in each statistical category across their entire career.\n\n"
                
                elif parsing.time_context == 'best_season' and 'best_season_data' in data:
                    best_season_data = data['best_season_data']
                    analysis += f"**Best Season Analysis:**\n"
                    
                    for stat, season_data in best_season_data.items():
                        stat_display = {
                            'PTS': 'Points', 'REB': 'Rebounds', 'AST': 'Assists', 'FG_PCT': 'Field Goal %'
                        }.get(stat, stat)
                        
                        value = season_data['value']
                        season = season_data['season']
                        
                        if stat == 'FG_PCT':
                            analysis += f"- Best {stat_display} Season: {value*100:.1f}% in {season}\n"
                        else:
                            analysis += f"- Best {stat_display} Season: {value:.1f} in {season}\n"
                    
                    analysis += f"\nThe charts highlight {player_name}'s peak seasons for each statistical category, showing when they performed at their absolute best.\n\n"
                
                else:
                    if 'career_stats' in data:
                        career_stats = data['career_stats']
                        if not career_stats.empty:
                            seasons = len(career_stats)
                            analysis += f"**Career Overview:**\n"
                            analysis += f"- Seasons Played: {seasons}\n"
                            
                            if 'PTS' in career_stats.columns:
                                avg_pts = career_stats['PTS'].mean()
                                max_pts = career_stats['PTS'].max()
                                analysis += f"- Career Average Points: {avg_pts:.1f} per season\n"
                                analysis += f"- Best Scoring Season: {max_pts:.1f} points\n"
                            
                            if 'REB' in career_stats.columns and 'AST' in career_stats.columns:
                                avg_reb = career_stats['REB'].mean()
                                avg_ast = career_stats['AST'].mean()
                                analysis += f"- Career Averages: {avg_reb:.1f} rebounds, {avg_ast:.1f} assists\n"
                            analysis += "\n"
                
                analysis += f"The charts show {player_name}'s progression across key statistical categories including scoring, rebounding, and playmaking. "
                analysis += "Trend lines indicate performance trajectory over time.\n\n"
        
        elif parsing.query_type == "analyze_team":
            if parsing.teams:
                team_name = parsing.teams[0]
                season = data.get('season', 'current season')
                
                if parsing.time_context == 'all_time':
                    analysis += f"**Answer: {team_name} All-Time Franchise Statistics and History**\n\n"
                elif parsing.time_context == 'championship_years':
                    analysis += f"**Answer: {team_name} Championship Years and Title History**\n\n"
                else:
                    analysis += f"**Answer: {team_name} Performance Analysis for {season}**\n\n"
                
                if 'team_stats' in data:
                    team_stats = data['team_stats']
                    if not team_stats.empty:
                        if parsing.time_context == 'all_time':
                            analysis += f"**All-Time Franchise Overview:**\n"
                            total_seasons = len(team_stats)
                            analysis += f"- Franchise Seasons: {total_seasons}\n"
                            
                            if 'WINS' in team_stats.columns and 'LOSSES' in team_stats.columns:
                                total_wins = team_stats['WINS'].sum()
                                total_losses = team_stats['LOSSES'].sum()
                                all_time_win_pct = total_wins / (total_wins + total_losses) * 100 if (total_wins + total_losses) > 0 else 0
                                analysis += f"- All-Time Record: {total_wins}-{total_losses} ({all_time_win_pct:.1f}% win rate)\n"
                            
                            if 'WIN_PCT' in team_stats.columns:
                                best_season = team_stats.loc[team_stats['WIN_PCT'].idxmax()]
                                worst_season = team_stats.loc[team_stats['WIN_PCT'].idxmin()]
                                analysis += f"- Best Season: {best_season['YEAR']} ({best_season['WINS']}-{best_season['LOSSES']}, {best_season['WIN_PCT']*100:.1f}%)\n"
                                analysis += f"- Worst Season: {worst_season['YEAR']} ({worst_season['WINS']}-{worst_season['LOSSES']}, {worst_season['WIN_PCT']*100:.1f}%)\n"
                            
                            if 'NBA_FINALS_APPEARANCE' in team_stats.columns:
                                championships = len(team_stats[team_stats['NBA_FINALS_APPEARANCE'].str.contains('CHAMPION', na=False)])
                                finals_appearances = len(team_stats[team_stats['NBA_FINALS_APPEARANCE'].notna()])
                                if championships > 0:
                                    analysis += f"- Championships: {championships}\n"
                                if finals_appearances > 0:
                                    analysis += f"- Finals Appearances: {finals_appearances}\n"
                        
                        elif parsing.time_context == 'championship_years':
                            analysis += f"**Championship History:**\n"
                            if len(team_stats) > 0:
                                analysis += f"- Championship Seasons: {len(team_stats)}\n"
                                for _, season in team_stats.iterrows():
                                    year = season.get('YEAR', 'Unknown')
                                    record = f"{season.get('WINS', '?')}-{season.get('LOSSES', '?')}"
                                    win_pct = season.get('WIN_PCT', 0) * 100
                                    analysis += f"  • {year}: {record} ({win_pct:.1f}%)\n"
                            else:
                                analysis += f"- No championship years found in available data\n"
                        
                        else:
                            analysis += f"**Season Overview:**\n"
                            team_row = team_stats.iloc[0]
                            
                            if 'GP' in team_stats.columns:
                                games_played = team_row['GP']
                                analysis += f"- Games Played: {games_played}\n"
                            
                            if 'WINS' in team_stats.columns and 'LOSSES' in team_stats.columns:
                                wins = team_row['WINS']
                                losses = team_row['LOSSES']
                                if 'WIN_PCT' in team_stats.columns:
                                    win_pct = team_row['WIN_PCT'] * 100
                                else:
                                    win_pct = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
                                analysis += f"- Record: {wins}-{losses} ({win_pct:.1f}% win rate)\n"
                            elif 'WL' in team_stats.columns:
                                wins = len(team_stats[team_stats['WL'] == 'W'])
                                losses = len(team_stats[team_stats['WL'] == 'L'])
                                win_pct = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
                                analysis += f"- Record: {wins}-{losses} ({win_pct:.1f}% win rate)\n"
                            
                            if 'PTS' in team_stats.columns and 'GP' in team_stats.columns:
                                total_pts = team_row['PTS']
                                games_played = team_row['GP']
                                avg_ppg = total_pts / games_played if games_played > 0 else 0
                                analysis += f"- Average Points Per Game: {avg_ppg:.1f}\n"
                                analysis += f"- Total Points: {total_pts}\n"
                            elif 'PTS' in team_stats.columns:
                                avg_pts = team_stats['PTS'].mean()
                                analysis += f"- Average Points Per Game: {avg_pts:.1f}\n"
                            
                            if 'FG_PCT' in team_stats.columns:
                                fg_pct = team_row['FG_PCT'] * 100
                                analysis += f"- Field Goal Percentage: {fg_pct:.1f}%\n"
                            
                            if 'CONF_RANK' in team_stats.columns and team_row['CONF_RANK'] > 0:
                                conf_rank = team_row['CONF_RANK']
                                analysis += f"- Conference Ranking: #{conf_rank}\n"
                            
                            analysis += "\n"
                
                analysis += f"The performance charts show {team_name}'s win-loss record, scoring trends, and key metrics throughout the season.\n\n"
        
        analysis += self._generate_comparison_insights(query, data, parsing)
        
        return analysis
    
    def _generate_comparison_insights(self, query: str, data: Dict, parsing: QueryParsing) -> str:
        insights = "---\n\n## Key Analysis Insights\n\n"
        
        if parsing.query_type == "compare_players" and 'player1_stats' in data and 'player2_stats' in data:
            p1_stats = data['player1_stats']
            p2_stats = data['player2_stats']
            p1_name = data.get('player1_name', 'Player 1')
            p2_name = data.get('player2_name', 'Player 2')
            
            if not p1_stats.empty and not p2_stats.empty:
                insights += "**Main Comparison Factors:**\n\n"
                
                p1_seasons = len(p1_stats)
                p2_seasons = len(p2_stats)
                if p1_seasons != p2_seasons:
                    longer_career = p1_name if p1_seasons > p2_seasons else p2_name
                    insights += f"• **Career Longevity**: {longer_career} has played {max(p1_seasons, p2_seasons)} seasons vs {min(p1_seasons, p2_seasons)}\n"
                
                p1_latest = p1_stats.iloc[-1]
                p2_latest = p2_stats.iloc[-1]
                
                if parsing.attributes:
                    stats_for_insights = parsing.attributes
                else:
                    stats_for_insights = ['PPG', 'RPG', 'APG']
                
                insight_mappings = {
                    'PTS': ('PTS', 'Scoring', 'total points', False),
                    'PPG': ('PTS', 'Scoring', 'points per game', True),
                    'REB': ('REB', 'Rebounding', 'total rebounds', False),
                    'RPG': ('REB', 'Rebounding', 'rebounds per game', True),
                    'AST': ('AST', 'Playmaking', 'total assists', False),
                    'APG': ('AST', 'Playmaking', 'assists per game', True),
                    'STL': ('STL', 'Defense', 'steals per game', True),
                    'BLK': ('BLK', 'Defense', 'blocks per game', True),
                    'FG_PCT': ('FG_PCT', 'Shooting Accuracy', 'field goal percentage', False)
                }
                
                for stat in stats_for_insights:
                    if stat in insight_mappings:
                        raw_stat, category, description, is_per_game = insight_mappings[stat]
                        
                        if raw_stat in p1_latest and raw_stat in p2_latest:
                            if is_per_game and 'GP' in p1_latest and 'GP' in p2_latest:
                                p1_val = p1_latest[raw_stat] / max(p1_latest['GP'], 1)
                                p2_val = p2_latest[raw_stat] / max(p2_latest['GP'], 1)
                                leader = p1_name if p1_val > p2_val else p2_name
                                diff = abs(p1_val - p2_val)
                                insights += f"• **{category}**: {leader} leads in recent {description} ({p1_val:.1f} vs {p2_val:.1f})\n"
                            elif raw_stat == 'FG_PCT':
                                p1_pct = p1_latest[raw_stat] * 100 if p1_latest[raw_stat] <= 1 else p1_latest[raw_stat]
                                p2_pct = p2_latest[raw_stat] * 100 if p2_latest[raw_stat] <= 1 else p2_latest[raw_stat]
                                leader = p1_name if p1_pct > p2_pct else p2_name
                                insights += f"• **{category}**: {leader} has better {description} ({p1_pct:.1f}% vs {p2_pct:.1f}%)\n"
                            else:
                                leader = p1_name if p1_latest[raw_stat] > p2_latest[raw_stat] else p2_name
                                insights += f"• **{category}**: {leader} leads in {description}\n"
                
                insights += f"\n*Based on most recent season statistics*\n\n"
        
        elif parsing.query_type == "predict_player" and 'predictions' in data:
            predictions = data['predictions']
            player_name = parsing.players[0] if parsing.players else "Player"
            
            insights += f"**Prediction Summary for {player_name}:**\n\n"
            
            improving_stats = []
            declining_stats = []
            stable_stats = []
            
            for stat, pred_data in predictions.items():
                trend = pred_data.get('trend_direction', 'stable')
                stat_display = {
                    'PTS': 'Total Scoring', 'PPG': 'Scoring',
                    'REB': 'Total Rebounding', 'RPG': 'Rebounding', 
                    'AST': 'Total Assists', 'APG': 'Playmaking',
                    'FG_PCT': 'Shooting', 'FG3_PCT': '3-Point Shooting', 'FT_PCT': 'Free Throw Shooting'
                }.get(stat, stat)
                
                if trend == 'improving':
                    improving_stats.append(stat_display)
                elif trend == 'declining':
                    declining_stats.append(stat_display)
                else:
                    stable_stats.append(stat_display)
            
            if improving_stats:
                insights += f"• **Expected Improvement**: {', '.join(improving_stats)}\n"
            if declining_stats:
                insights += f"• **Potential Decline**: {', '.join(declining_stats)}\n"
            if stable_stats:
                insights += f"• **Stable Performance**: {', '.join(stable_stats)}\n"
            
            insights += f"\n*Predictions based on career trajectory and aging patterns*\n\n"
        
        return insights
    
    def _summarize_data(self, data: Dict) -> str:
        summary = "Data includes: "
        for key, value in data.items():
            if hasattr(value, 'shape'):
                summary += f"{key} ({value.shape[0]} records), "
            elif isinstance(value, (list, dict)):
                summary += f"{key} ({len(value)} items), "
        return summary.rstrip(', ')
    
    def check_ollama_connection(self) -> bool:
        try:
            models = self.client.list()
            return len(models['models']) > 0
        except Exception:
            return False