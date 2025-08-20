import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from plotly.subplots import make_subplots

from .constants import *

class NBAVisualizer:
    def __init__(self):
        self.primary_colors = [NBA_BLUE, NBA_ORANGE, NBA_GREEN, NBA_RED, NBA_PURPLE]
        self.team_colors = {
            'Lakers': LAKERS_PURPLE,
            'Warriors': WARRIORS_BLUE,
            'Bulls': BULLS_RED,
            'Celtics': CELTICS_GREEN,
            'Heat': HEAT_RED
        }
    
    def _prepare_historical_data(self, historical_data: pd.DataFrame, stat: str):
        if historical_data.empty:
            return []
        
        games_played_column = 'GP'
        
        if stat in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'MIN'] and games_played_column in historical_data.columns:
            if stat in historical_data.columns:
                return historical_data[stat] / np.maximum(historical_data[games_played_column], MIN_GAMES_THRESHOLD)
        
        if stat in historical_data.columns:
            return historical_data[stat]
        
        if games_played_column not in historical_data.columns:
            return []

        per_game_stat_mappings = {
            'PPG': 'PTS',
            'RPG': 'REB', 
            'APG': 'AST',
            'MIN_PER_GAME': 'MIN'
        }
        
        total_stat_column = per_game_stat_mappings.get(stat)
        if total_stat_column and total_stat_column in historical_data.columns:
            return historical_data[total_stat_column] / np.maximum(historical_data[games_played_column], MIN_GAMES_THRESHOLD)
        
        return []
        
    def create_player_comparison_chart(self, player1_data: pd.DataFrame, player2_data: pd.DataFrame, 
                                     player1_name: str, player2_name: str, 
                                     stats_to_compare: List[str] = None, alignment: str = "latest") -> go.Figure:
        
        default_comparison_stats = ['PPG', 'RPG', 'APG', 'FG_PCT', 'STL', 'BLK']
        stats_to_compare = stats_to_compare or default_comparison_stats
        
        if alignment == "latest":
            p1_latest = player1_data.iloc[-1] if not player1_data.empty else None
            p2_latest = player2_data.iloc[-1] if not player2_data.empty else None
        elif alignment == "season1":
            p1_latest = player1_data.iloc[0] if not player1_data.empty else None
            p2_latest = player2_data.iloc[0] if not player2_data.empty else None
        else:
            p1_latest = player1_data.mean() if not player1_data.empty else None
            p2_latest = player2_data.mean() if not player2_data.empty else None
        
        if p1_latest is None or p2_latest is None:
            return self._create_empty_chart("Insufficient data for comparison")
        
        categories = []
        player1_values = []
        player2_values = []
        
        for stat in stats_to_compare:
            val1 = None
            val2 = None
            
            if stat == 'PPG' and 'PTS' in p1_latest and 'PTS' in p2_latest and 'GP' in p1_latest and 'GP' in p2_latest:
                val1 = p1_latest['PTS'] / max(p1_latest['GP'], 1)
                val2 = p2_latest['PTS'] / max(p2_latest['GP'], 1)
            elif stat == 'RPG' and 'REB' in p1_latest and 'REB' in p2_latest and 'GP' in p1_latest and 'GP' in p2_latest:
                val1 = p1_latest['REB'] / max(p1_latest['GP'], 1)
                val2 = p2_latest['REB'] / max(p2_latest['GP'], 1)
            elif stat == 'APG' and 'AST' in p1_latest and 'AST' in p2_latest and 'GP' in p1_latest and 'GP' in p2_latest:
                val1 = p1_latest['AST'] / max(p1_latest['GP'], 1)
                val2 = p2_latest['AST'] / max(p2_latest['GP'], 1)
            elif stat in p1_latest and stat in p2_latest:
                val1 = float(p1_latest[stat]) if pd.notna(p1_latest[stat]) else 0
                val2 = float(p2_latest[stat]) if pd.notna(p2_latest[stat]) else 0
            
            if val1 is not None and val2 is not None:
                if stat in ['FG_PCT', 'FG3_PCT', 'FT_PCT'] and val1 <= 1 and val2 <= 1:
                    val1 *= 100
                    val2 *= 100
                
                categories.append(stat)
                player1_values.append(val1)
                player2_values.append(val2)
        
        if not categories:
            return self._create_empty_chart("No comparable statistics found")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=player1_values + [player1_values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=player1_name,
            fillcolor='rgba(31, 119, 180, 0.3)',
            line_color='rgba(31, 119, 180, 1)'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=player2_values + [player2_values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=player2_name,
            fillcolor='rgba(255, 127, 14, 0.3)',
            line_color='rgba(255, 127, 14, 1)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(max(player1_values), max(player2_values)) * 1.1]
                )),
            showlegend=True,
            title=f"{player1_name} vs {player2_name} - Statistical Comparison"
        )
        
        return fig
    
    def create_season_aligned_comparison(self, player1_data: pd.DataFrame, player2_data: pd.DataFrame, 
                                       player1_name: str, player2_name: str, stat: str = 'PTS') -> go.Figure:
        """Create a chart comparing players season by season (e.g., both players' season 1, season 2, etc.)"""
        
        if player1_data.empty or player2_data.empty or stat not in player1_data.columns or stat not in player2_data.columns:
            return self._create_empty_chart(f"No data available for {stat} comparison")
        
        max_seasons = min(len(player1_data), len(player2_data))
        
        if max_seasons == 0:
            return self._create_empty_chart("No overlapping seasons to compare")
        
        seasons = [f"Season {i+1}" for i in range(max_seasons)]
        
        fig = go.Figure()
        
        if stat in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'MIN'] and 'GP' in player1_data.columns:
            player1_values = player1_data[stat][:max_seasons] / np.maximum(player1_data['GP'][:max_seasons], 1)
        else:
            player1_values = player1_data[stat][:max_seasons]
        
        if stat in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'MIN'] and 'GP' in player2_data.columns:
            player2_values = player2_data[stat][:max_seasons] / np.maximum(player2_data['GP'][:max_seasons], 1)
        else:
            player2_values = player2_data[stat][:max_seasons]
        
        fig.add_trace(go.Scatter(
            x=seasons,
            y=player1_values,
            mode='lines+markers',
            name=f"{player1_name}",
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=seasons,
            y=player2_values,
            mode='lines+markers',
            name=f"{player2_name}",
            line=dict(color='#ff7f0e', width=3),
            marker=dict(size=8)
        ))
        
        y_title = stat
        if stat == 'FG_PCT':
            y_title = "Field Goal %"
        elif stat == 'PTS':
            y_title = "Points Per Game"
        elif stat == 'REB':
            y_title = "Rebounds Per Game"
        elif stat == 'AST':
            y_title = "Assists Per Game"
        elif stat == 'STL':
            y_title = "Steals Per Game"
        elif stat == 'BLK':
            y_title = "Blocks Per Game"
        
        fig.update_layout(
            title=f"{player1_name} vs {player2_name} - {y_title} by Career Season",
            xaxis_title="Career Season",
            yaxis_title=y_title,
            hovermode='x unified'
        )
        
        return fig
    
    def create_career_progression_chart(self, player_data: pd.DataFrame, 
                                      player_name: str, stat: str = 'PTS') -> go.Figure:
        if player_data.empty or stat not in player_data.columns:
            return self._create_empty_chart(f"No data available for {stat}")
        
        fig = go.Figure()
        
        seasons = [f"Season {i+1}" for i in range(len(player_data))]
        
        if stat in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'MIN'] and 'GP' in player_data.columns:
            y_values = player_data[stat] / np.maximum(player_data['GP'], 1)
        else:
            y_values = player_data[stat]
        
        fig.add_trace(go.Scatter(
            x=seasons,
            y=y_values,
            mode='lines+markers',
            name=f"{player_name} {stat}",
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8)
        ))
        
        if len(player_data) > 2:
            x_numeric = list(range(len(player_data)))
            z = np.polyfit(x_numeric, y_values.fillna(y_values.mean()), 1)
            p = np.poly1d(z)
            
            fig.add_trace(go.Scatter(
                x=seasons,
                y=p(x_numeric),
                mode='lines',
                name='Trend',
                line=dict(color='red', width=2, dash='dash'),
                opacity=0.7
            ))
        
        fig.update_layout(
            title=f"{player_name} - {stat} Career Progression",
            xaxis_title="Season",
            yaxis_title=stat,
            hovermode='x unified'
        )
        
        return fig
    
    def create_team_performance_chart(self, team_data: pd.DataFrame, team_name: str) -> go.Figure:
        if team_data.empty:
            return self._create_empty_chart("No team data available")
        
        if len(team_data) > 5 and 'YEAR' in team_data.columns:
            return self._create_franchise_history_chart(team_data, team_name)
        
        if len(team_data) == 1 and 'YEAR' in team_data.columns:
            return self._create_season_vs_league_chart(team_data, team_name)
        
        if 'WL' in team_data.columns:
            wins = len(team_data[team_data['WL'] == 'W'])
            losses = len(team_data[team_data['WL'] == 'L'])
        else:
            wins = losses = len(team_data) // 2
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Win/Loss Record', 'Points Per Game', 'Field Goal %', 'Rebounds Per Game'),
            specs=[[{"type": "pie"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        fig.add_trace(go.Pie(
            labels=['Wins', 'Losses'],
            values=[wins, losses],
            marker_colors=['green', 'red']
        ), row=1, col=1)
        
        if 'PTS' in team_data.columns:
            fig.add_trace(go.Scatter(
                x=list(range(len(team_data))),
                y=team_data['PTS'],
                mode='lines+markers',
                name='Points',
                line_color='blue'
            ), row=1, col=2)
        
        if 'FG_PCT' in team_data.columns:
            fig.add_trace(go.Scatter(
                x=list(range(len(team_data))),
                y=team_data['FG_PCT'] * 100,
                mode='lines+markers',
                name='FG%',
                line_color='orange'
            ), row=2, col=1)
        
        if 'REB' in team_data.columns:
            fig.add_trace(go.Scatter(
                x=list(range(len(team_data))),
                y=team_data['REB'],
                mode='lines+markers',
                name='Rebounds',
                line_color='purple'
            ), row=2, col=2)
        
        fig.update_layout(
            title=f"{team_name} - Season Performance Analysis",
            showlegend=False,
            height=600
        )
        
        return fig
    
    def create_prediction_chart(self, historical_data: pd.DataFrame, predictions: Dict, 
                              player_name: str, stat: str) -> go.Figure:
        if stat not in predictions:
            return self._create_empty_chart(f"No predictions available for {stat}")
        
        pred_data = predictions[stat]
        
        fig = go.Figure()
        
        seasons = [f"Season {i+1}" for i in range(len(historical_data))]
        historical_values = self._prepare_historical_data(historical_data, stat)
        
        if len(historical_values) > 0:
            fig.add_trace(go.Scatter(
                x=seasons,
                y=historical_values,
                mode='lines+markers',
                name='Historical Performance',
                line=dict(color='blue', width=3),
                marker=dict(size=8),
                hovertemplate=f'{stat}: %{{y:.1f}}<br>Season: %{{x}}<extra></extra>'
            ))
        
        next_season = f"Season {len(seasons) + 1}"
        prediction = pred_data['ensemble_mean']
        confidence_interval = pred_data['confidence_interval']
        confidence_percentage = pred_data.get('confidence_percentage', 75)
        
        fig.add_trace(go.Scatter(
            x=[seasons[-1] if seasons else "Season 1", next_season],
            y=[historical_values.iloc[-1] if len(historical_values) > 0 else prediction, prediction],
            mode='lines+markers',
            name='Prediction',
            line=dict(color='red', width=3, dash='dash'),
            marker=dict(size=10, symbol='diamond'),
            hovertemplate=f'Predicted {stat}: %{{y:.1f}}<br>Confidence: {confidence_percentage:.1f}%<br>Season: %{{x}}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=[next_season, next_season, next_season],
            y=[confidence_interval[0], prediction, confidence_interval[1]],
            mode='markers',
            name='Confidence Interval',
            marker=dict(color='red', size=[6, 10, 6], opacity=0.5),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=[next_season],
            y=[prediction],
            error_y=dict(
                type='data',
                symmetric=False,
                array=[confidence_interval[1] - prediction],
                arrayminus=[prediction - confidence_interval[0]],
                color='red'
            ),
            mode='markers',
            marker=dict(color='red', size=10),
            showlegend=False
        ))
        
        fig.update_layout(
            title=f"{player_name} - {stat} Prediction for Next Season",
            xaxis_title="Season",
            yaxis_title=stat,
            hovermode='x unified'
        )
        
        return fig
    
    def create_multi_season_prediction_chart(self, historical_data: pd.DataFrame, predictions: Dict, 
                                           player_name: str, stat: str, num_seasons: int) -> go.Figure:
        """Create a chart showing multi-season projections"""
        if stat not in predictions:
            return self._create_empty_chart(f"No multi-season predictions available for {stat}")
        
        pred_data = predictions[stat]
        
        if 'multi_season_projections' not in pred_data:
            return self.create_prediction_chart(historical_data, {stat: pred_data}, player_name, stat)
        
        fig = go.Figure()
        
        seasons = [f"Season {i+1}" for i in range(len(historical_data))]
        historical_values = self._prepare_historical_data(historical_data, stat)
        
        if len(historical_values) > 0:
            if stat in ['FG_PCT', 'FG3_PCT', 'FT_PCT']:
                historical_display = historical_values * 100
                hover_template = f'{stat}: %{{y:.1f}}%<br>Season: %{{x}}<extra></extra>'
            else:
                historical_display = historical_values
                hover_template = f'{stat}: %{{y:.1f}}<br>Season: %{{x}}<extra></extra>'
            
            fig.add_trace(go.Scatter(
                x=seasons,
                y=historical_display,
                mode='lines+markers',
                name='Historical Performance',
                line=dict(color='blue', width=3),
                marker=dict(size=8),
                hovertemplate=hover_template
            ))
        
        projections = pred_data['multi_season_projections']
        future_seasons = [f"Season {len(seasons) + proj['season']}" for proj in projections]
        projected_values = [proj['projected_value'] for proj in projections]
        
        confidence_percentages = [proj.get('confidence_percentage', 75) for proj in projections]
        
        if stat in ['FG_PCT', 'FG3_PCT', 'FT_PCT']:
            display_values = [val * 100 for val in projected_values]
            hover_texts = [f'Predicted {stat}: {val:.1f}%<br>Confidence: {conf:.1f}%<br>Season: {season}' 
                          for val, conf, season in zip(display_values, confidence_percentages, future_seasons)]
            chart_projected_values = display_values
        else:
            hover_texts = [f'Predicted {stat}: {val:.1f}<br>Confidence: {conf:.1f}%<br>Season: {season}' 
                          for val, conf, season in zip(projected_values, confidence_percentages, future_seasons)]
            chart_projected_values = projected_values
        
        if len(historical_values) > 0:
            connection_x = [seasons[-1], future_seasons[0]]
            if hasattr(historical_values, 'iloc'):
                last_value = historical_values.iloc[-1]
            else:
                last_value = historical_values[-1] if len(historical_values) > 0 else 0
            
            if stat in ['FG_PCT', 'FG3_PCT', 'FT_PCT']:
                last_value = last_value * 100
            
            connection_y = [last_value, chart_projected_values[0]]
            
            fig.add_trace(go.Scatter(
                x=connection_x,
                y=connection_y,
                mode='lines',
                name='Transition',
                line=dict(color='orange', width=2, dash='dot'),
                showlegend=False
            ))
        
        fig.add_trace(go.Scatter(
            x=future_seasons,
            y=chart_projected_values,
            mode='lines+markers',
            name=f'Projected Performance',
            line=dict(color='red', width=3, dash='dash'),
            marker=dict(size=10, symbol='diamond'),
            hovertemplate='%{text}<extra></extra>',
            text=hover_texts
        ))
        
        if stat in ['FG_PCT', 'FG3_PCT', 'FT_PCT']:
            upper_bounds = [proj['confidence_interval'][1] * 100 for proj in projections]
            lower_bounds = [proj['confidence_interval'][0] * 100 for proj in projections]
        else:
            upper_bounds = [proj['confidence_interval'][1] for proj in projections]
            lower_bounds = [proj['confidence_interval'][0] for proj in projections]
        
        fig.add_trace(go.Scatter(
            x=future_seasons + future_seasons[::-1],
            y=upper_bounds + lower_bounds[::-1],
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Range',
            showlegend=True
        ))
        
        for i, proj in enumerate(projections):
            if stat in ['FG_PCT', 'FG3_PCT', 'FT_PCT']:
                proj_val = chart_projected_values[i]
                error_upper = proj['confidence_interval'][1] * 100 - proj_val
                error_lower = proj_val - proj['confidence_interval'][0] * 100
            else:
                proj_val = chart_projected_values[i]
                error_upper = proj['confidence_interval'][1] - proj_val
                error_lower = proj_val - proj['confidence_interval'][0]
            
            fig.add_trace(go.Scatter(
                x=[future_seasons[i]],
                y=[proj_val],
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=[error_upper],
                    arrayminus=[error_lower],
                    color='red',
                    width=3
                ),
                mode='markers',
                marker=dict(color='red', size=12),
                showlegend=False
            ))
        
        stat_display = stat
        if stat == 'PTS':
            stat_display = 'Points Per Game'
        elif stat == 'REB':
            stat_display = 'Rebounds Per Game'
        elif stat == 'AST':
            stat_display = 'Assists Per Game'
        elif stat == 'FG_PCT':
            stat_display = 'Field Goal Percentage'
        
        fig.update_layout(
            title=f"{player_name} - {stat_display} Multi-Season Projection ({num_seasons} Seasons)",
            xaxis_title="Career Season",
            yaxis_title=stat_display,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_stat_distribution_chart(self, data: pd.DataFrame, stat: str, player_name: str) -> go.Figure:
        if stat not in data.columns or data.empty:
            return self._create_empty_chart(f"No data available for {stat}")
        
        values = data[stat].dropna()
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=values,
            nbinsx=20,
            name=f"{stat} Distribution",
            opacity=0.7,
            marker_color='lightblue'
        ))
        
        mean_val = values.mean()
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_val:.1f}"
        )
        
        fig.update_layout(
            title=f"{player_name} - {stat} Distribution",
            xaxis_title=stat,
            yaxis_title="Frequency",
            bargap=0.1
        )
        
        return fig
    
    def _create_franchise_history_chart(self, team_data: pd.DataFrame, team_name: str) -> go.Figure:
        """Create a meaningful franchise history visualization showing eras, championships, and key achievements"""
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=(
                'Win Percentage by Era', 'Championships & Finals', 
                'Team Performance Timeline', 'Conference Rankings',
                'Best & Worst Seasons', 'Franchise Milestones'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter", "colspan": 2}, None],
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "table", "colspan": 2}, None]
            ],
            vertical_spacing=0.06,
            horizontal_spacing=0.1
        )
        
        years = []
        for year in team_data['YEAR']:
            if isinstance(year, str):
                year_str = str(year).split('-')[0]
                if year_str.isdigit():
                    years.append(int(year_str))
                else:
                    years.append(2000)
            else:
                years.append(int(year))
        
        team_data = team_data.copy()
        team_data['YEAR_INT'] = years
        
        if 'WIN_PCT' in team_data.columns:
            fig.add_trace(go.Scatter(
                x=team_data['YEAR_INT'],
                y=team_data['WIN_PCT'] * 100,
                mode='lines+markers',
                name='Win %',
                line=dict(color='blue', width=3),
                marker=dict(size=6),
                hovertemplate='%{x}: %{y:.1f}%<extra></extra>'
            ), row=1, col=1)
            
            if 'NBA_FINALS_APPEARANCE' in team_data.columns:
                champ_data = team_data[team_data['NBA_FINALS_APPEARANCE'].str.contains('CHAMPION', na=False)]
                if not champ_data.empty:
                    fig.add_trace(go.Scatter(
                        x=champ_data['YEAR_INT'],
                        y=champ_data['WIN_PCT'] * 100,
                        mode='markers',
                        name='Championships',
                        marker=dict(color='gold', size=12, symbol='star'),
                        hovertemplate='%{x}: Championship Season (%{y:.1f}%)<extra></extra>'
                    ), row=1, col=1)
        
        if 'NBA_FINALS_APPEARANCE' in team_data.columns:
            championship_years = team_data[team_data['NBA_FINALS_APPEARANCE'].str.contains('CHAMPION', na=False)]['YEAR_INT'].tolist()
            finals_years = team_data[team_data['NBA_FINALS_APPEARANCE'].notna()]['YEAR_INT'].tolist()
            
            decades = {}
            for year in championship_years:
                decade = (year // 10) * 10
                decades[decade] = decades.get(decade, 0) + 1
            
            if decades:
                fig.add_trace(go.Bar(
                    x=[f"{d}s" for d in sorted(decades.keys())],
                    y=list(decades.values()),
                    name='Championships',
                    marker_color='gold',
                    hovertemplate='%{x}: %{y} championships<extra></extra>'
                ), row=1, col=2)
        
        if 'WINS' in team_data.columns and 'LOSSES' in team_data.columns:
            fig.add_trace(go.Scatter(
                x=team_data['YEAR_INT'],
                y=team_data['WINS'],
                mode='lines+markers',
                name='Wins',
                line=dict(color='green', width=2),
                hovertemplate='%{x}: %{y} wins<extra></extra>'
            ), row=2, col=1)
            
            fig.add_trace(go.Scatter(
                x=team_data['YEAR_INT'],
                y=team_data['LOSSES'],
                mode='lines+markers',
                name='Losses',
                line=dict(color='red', width=2),
                hovertemplate='%{x}: %{y} losses<extra></extra>'
            ), row=2, col=1)
        
        if 'CONF_RANK' in team_data.columns:
            valid_ranks = team_data[team_data['CONF_RANK'] > 0]
            if not valid_ranks.empty:
                fig.add_trace(go.Scatter(
                    x=valid_ranks['YEAR_INT'],
                    y=valid_ranks['CONF_RANK'],
                    mode='lines+markers',
                    name='Conference Rank',
                    line=dict(color='purple', width=2),
                    hovertemplate='%{x}: #%{y} in conference<extra></extra>'
                ), row=3, col=1)
        
        if 'WIN_PCT' in team_data.columns:
            best_season = team_data.loc[team_data['WIN_PCT'].idxmax()]
            worst_season = team_data.loc[team_data['WIN_PCT'].idxmin()]
            
            best_year = best_season['YEAR'] if 'YEAR' in best_season else 'Unknown'
            worst_year = worst_season['YEAR'] if 'YEAR' in worst_season else 'Unknown'
            best_record = f"{best_season.get('WINS', '?')}-{best_season.get('LOSSES', '?')}"
            worst_record = f"{worst_season.get('WINS', '?')}-{worst_season.get('LOSSES', '?')}"
            
            seasons = [f'Best Season\n({best_year})', f'Worst Season\n({worst_year})']
            win_pcts = [best_season['WIN_PCT'] * 100, worst_season['WIN_PCT'] * 100]
            colors = ['gold', 'darkred']
            
            hover_texts = [
                f'Best Season: {best_year}<br>Record: {best_record}<br>Win %: {win_pcts[0]:.1f}%',
                f'Worst Season: {worst_year}<br>Record: {worst_record}<br>Win %: {win_pcts[1]:.1f}%'
            ]
            
            fig.add_trace(go.Bar(
                x=seasons,
                y=win_pcts,
                marker_color=colors,
                name='Season Records',
                hovertemplate='%{text}<extra></extra>',
                text=hover_texts
            ), row=3, col=2)
        
        milestones = []
        if 'NBA_FINALS_APPEARANCE' in team_data.columns:
            championships = len(team_data[team_data['NBA_FINALS_APPEARANCE'].str.contains('CHAMPION', na=False)])
            
            finals_appearances = len(team_data[
                (team_data['NBA_FINALS_APPEARANCE'].notna()) & 
                (team_data['NBA_FINALS_APPEARANCE'] != 'N/A')
            ])
            
            milestones.extend([
                ['Total Championships', str(championships)],
                ['Finals Appearances', str(finals_appearances)]
            ])
        
        if 'WIN_PCT' in team_data.columns:
            total_wins = team_data['WINS'].sum() if 'WINS' in team_data.columns else 0
            total_losses = team_data['LOSSES'].sum() if 'LOSSES' in team_data.columns else 0
            franchise_seasons = len(team_data)
            all_time_win_pct = (total_wins / (total_wins + total_losses)) * 100 if (total_wins + total_losses) > 0 else 0
            
            milestones.extend([
                ['Franchise Seasons', str(franchise_seasons)],
                ['All-Time Record', f"{total_wins}-{total_losses}"],
                ['All-Time Win %', f"{all_time_win_pct:.1f}%"],
                ['Best Season Record', f"{best_season['WINS']}-{best_season['LOSSES']} ({best_season['YEAR']})"],
                ['Worst Season Record', f"{worst_season['WINS']}-{worst_season['LOSSES']} ({worst_season['YEAR']})"]
            ])
            
            if 'PO_WINS' in team_data.columns and 'PO_LOSSES' in team_data.columns:
                total_po_wins = team_data['PO_WINS'].sum()
                total_po_losses = team_data['PO_LOSSES'].sum()
                if total_po_wins > 0 or total_po_losses > 0:
                    po_win_pct = (total_po_wins / (total_po_wins + total_po_losses)) * 100
                    milestones.extend([
                        ['Playoff Record', f"{total_po_wins}-{total_po_losses}"],
                        ['Playoff Win %', f"{po_win_pct:.1f}%"]
                    ])
            
            if 'CONF_COUNT' in team_data.columns:
                conf_titles = team_data['CONF_COUNT'].sum()
                if conf_titles > 0:
                    milestones.append(['Conference Titles', str(conf_titles)])
            
            if 'DIV_COUNT' in team_data.columns:
                div_titles = team_data['DIV_COUNT'].sum()
                if div_titles > 0:
                    milestones.append(['Division Titles', str(div_titles)])
        
        if milestones:
            fig.add_trace(go.Table(
                header=dict(
                    values=['Milestone', 'Value'], 
                    fill_color='darkblue',
                    font=dict(color='white', size=12),
                    align='center'
                ),
                cells=dict(
                    values=list(zip(*milestones)), 
                    fill_color='white',
                    font=dict(color='black', size=11),
                    align='left',
                    line=dict(color='darkblue', width=1)
                )
            ), row=4, col=1)
        
        fig.update_layout(
            title=f"{team_name} - Complete Franchise History",
            height=1200,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig.update_yaxes(title_text="Win Percentage (%)", row=1, col=1)
        fig.update_yaxes(title_text="Championships", row=1, col=2)
        fig.update_yaxes(title_text="Games", row=2, col=1)
        fig.update_yaxes(title_text="Conference Rank", autorange="reversed", row=3, col=1)
        fig.update_yaxes(title_text="Win Percentage (%)", row=3, col=2)
        
        fig.update_xaxes(title_text="Year", row=1, col=1)
        fig.update_xaxes(title_text="Year", row=2, col=1)
        fig.update_xaxes(title_text="Year", row=3, col=1)
        
        return fig
    
    def _create_season_vs_league_chart(self, team_data: pd.DataFrame, team_name: str) -> go.Figure:
        """Create a chart comparing team's specific season performance to league averages"""
        season_row = team_data.iloc[0]
        season_year = season_row.get('YEAR', 'Unknown Season')
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'{team_name} vs League Average', 'Offensive Performance',
                'Team Efficiency Metrics', 'Season Context'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "table"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.15
        )
        
        league_averages = {
            'WIN_PCT': 0.500,
            'PTS': 110.0,
            'FG_PCT': 0.460,
            'FG3_PCT': 0.350,
            'FT_PCT': 0.770,
            'REB': 43.0,
            'AST': 24.0,
            'STL': 8.0,
            'BLK': 5.0,
            'TOV': 14.0
        }
        
        if 'WIN_PCT' in season_row:
            team_win_pct = season_row['WIN_PCT'] * 100
            league_win_pct = league_averages['WIN_PCT'] * 100
            
            fig.add_trace(go.Bar(
                x=['Team', 'League Avg'],
                y=[team_win_pct, league_win_pct],
                marker_color=['blue' if team_win_pct >= league_win_pct else 'red', 'gray'],
                name='Win Percentage',
                hovertemplate='%{x}: %{y:.1f}%<extra></extra>'
            ), row=1, col=1)
        
        offensive_stats = ['PTS', 'FG_PCT', 'FG3_PCT']
        offensive_labels = ['Points/Game', 'FG%', '3P%']
        team_offensive = []
        league_offensive = []
        
        for stat in offensive_stats:
            if stat in season_row and stat in league_averages:
                if stat in ['FG_PCT', 'FG3_PCT']:
                    team_val = season_row[stat] * 100
                    league_val = league_averages[stat] * 100
                elif stat == 'PTS' and 'GP' in season_row:
                    team_val = season_row[stat] / season_row['GP'] if season_row['GP'] > 0 else 0
                    league_val = league_averages[stat]
                else:
                    team_val = season_row[stat]
                    league_val = league_averages[stat]
                
                team_offensive.append(team_val)
                league_offensive.append(league_val)
        
        if team_offensive:
            fig.add_trace(go.Bar(
                x=offensive_labels[:len(team_offensive)],
                y=team_offensive,
                name=f'{team_name}',
                marker_color='blue',
                hovertemplate='%{x}: %{y:.1f}<extra></extra>'
            ), row=1, col=2)
            
            fig.add_trace(go.Bar(
                x=offensive_labels[:len(league_offensive)],
                y=league_offensive,
                name='League Avg',
                marker_color='gray',
                hovertemplate='%{x}: %{y:.1f}<extra></extra>'
            ), row=1, col=2)
        
        efficiency_stats = ['REB', 'AST', 'TOV']
        efficiency_labels = ['Rebounds/Game', 'Assists/Game', 'Turnovers/Game']
        team_efficiency = []
        league_efficiency = []
        
        for stat in efficiency_stats:
            if stat in season_row and stat in league_averages and 'GP' in season_row:
                team_val = season_row[stat] / season_row['GP'] if season_row['GP'] > 0 else 0
                league_val = league_averages[stat]
                team_efficiency.append(team_val)
                league_efficiency.append(league_val)
        
        if team_efficiency:
            fig.add_trace(go.Bar(
                x=efficiency_labels[:len(team_efficiency)],
                y=team_efficiency,
                name=f'{team_name}',
                marker_color='green',
                hovertemplate='%{x}: %{y:.1f}<extra></extra>',
                showlegend=False
            ), row=2, col=1)
            
            fig.add_trace(go.Bar(
                x=efficiency_labels[:len(league_efficiency)],
                y=league_efficiency,
                name='League Avg',
                marker_color='gray',
                hovertemplate='%{x}: %{y:.1f}<extra></extra>',
                showlegend=False
            ), row=2, col=1)
        
        context_data = []
        if 'WIN_PCT' in season_row and 'WINS' in season_row and 'LOSSES' in season_row:
            record = f"{season_row['WINS']}-{season_row['LOSSES']}"
            win_pct = f"{season_row['WIN_PCT']*100:.1f}%"
            context_data.extend([
                ['Season', str(season_year)],
                ['Record', record],
                ['Win Percentage', win_pct]
            ])
        
        if 'CONF_RANK' in season_row and season_row['CONF_RANK'] > 0:
            context_data.append(['Conference Rank', f"#{season_row['CONF_RANK']}"])
        
        if 'DIV_RANK' in season_row and season_row['DIV_RANK'] > 0:
            context_data.append(['Division Rank', f"#{season_row['DIV_RANK']}"])
        
        if 'NBA_FINALS_APPEARANCE' in season_row and season_row['NBA_FINALS_APPEARANCE'] != 'N/A':
            context_data.append(['Playoff Result', season_row['NBA_FINALS_APPEARANCE']])
        
        if 'WIN_PCT' in season_row:
            team_win_pct = season_row['WIN_PCT']
            if team_win_pct > 0.600:
                performance = "Elite (Top Tier)"
            elif team_win_pct > 0.500:
                performance = "Above Average"
            elif team_win_pct > 0.400:
                performance = "Below Average"
            else:
                performance = "Poor"
            context_data.append(['Season Rating', performance])
        
        if context_data:
            fig.add_trace(go.Table(
                header=dict(
                    values=['Metric', 'Value'], 
                    fill_color='darkblue',
                    font=dict(color='white', size=12),
                    align='center'
                ),
                cells=dict(
                    values=list(zip(*context_data)), 
                    fill_color='white',
                    font=dict(color='black', size=11),
                    align='left',
                    line=dict(color='darkblue', width=1)
                )
            ), row=2, col=2)
        
        fig.update_layout(
            title=f"{team_name} {season_year} Season - League Comparison Analysis",
            height=800,
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Win Percentage (%)", row=1, col=1)
        fig.update_yaxes(title_text="Performance", row=1, col=2)
        fig.update_yaxes(title_text="Per Game Stats", row=2, col=1)
        
        return fig
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            title="No Data Available"
        )
        return fig