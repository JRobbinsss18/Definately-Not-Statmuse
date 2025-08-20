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
        fig.add_trace(go.Bar(
            x=categories,
            y=player1_values,
            name=player1_name,
            marker_color='rgba(31, 119, 180, 0.8)',
            text=[f"{val:.1f}" for val in player1_values],
            textposition='outside'
        ))
        fig.add_trace(go.Bar(
            x=categories,
            y=player2_values,
            name=player2_name,
            marker_color='rgba(255, 127, 14, 0.8)',
            text=[f"{val:.1f}" for val in player2_values],
            textposition='outside'
        ))
        fig.update_layout(
            title=dict(
                text=f"{player1_name} vs {player2_name} Statistical Comparison",
                font=dict(size=20, color='#1f2937'),
                x=0.5
            ),
            xaxis=dict(
                title="Statistics",
                title_font=dict(size=14),
                tickfont=dict(size=12)
            ),
            yaxis=dict(
                title="Performance Value",
                title_font=dict(size=14),
                tickfont=dict(size=12)
            ),
            barmode='group',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif"),
            height=500,
            margin=dict(l=60, r=60, t=80, b=60)
        )
        return fig
    def create_season_aligned_comparison(self, player1_data: pd.DataFrame, player2_data: pd.DataFrame,
                                       player1_name: str, player2_name: str, stat: str = 'PTS') -> go.Figure:
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
        stat_labels = {
            'PTS': 'Points Per Game', 'PPG': 'Points Per Game',
            'REB': 'Rebounds Per Game', 'RPG': 'Rebounds Per Game',
            'AST': 'Assists Per Game', 'APG': 'Assists Per Game',
            'STL': 'Steals Per Game', 'BLK': 'Blocks Per Game',
            'FG_PCT': 'Field Goal %', 'FT_PCT': 'Free Throw %',
            'MIN': 'Minutes Per Game'
        }
        display_stat = stat_labels.get(stat, stat)
        seasons = [f"Year {i+1}" for i in range(len(player_data))]
        y_values = self._prepare_historical_data(player_data, stat)
        if len(y_values) == 0:
            return self._create_empty_chart(f"No data available for {display_stat}")
        max_val = max(y_values)
        min_val = min(y_values)
        avg_val = np.mean(y_values)
        colors = []
        for val in y_values:
            if val >= avg_val * 1.1:
                colors.append('#2E8B57')
            elif val <= avg_val * 0.9:
                colors.append('#CD5C5C')
            else:
                colors.append('#4682B4')
        fig.add_trace(go.Scatter(
            x=seasons,
            y=y_values,
            mode='lines+markers',
            name=f"{display_stat}",
            line=dict(color='#4682B4', width=4),
            marker=dict(size=12, color=colors, line=dict(width=2, color='white')),
            hovertemplate=f"<b>%{{x}}</b><br>{display_stat}: %{{y:.1f}}<br><extra></extra>",
            text=[f"{val:.1f}" for val in y_values],
            textposition="top center",
            textfont=dict(size=10, color='black')
        ))
        fig.add_hline(
            y=avg_val,
            line_dash="dot",
            line_color="orange",
            line_width=2,
            annotation_text=f"Career Average: {avg_val:.1f}",
            annotation_position="top left",
            annotation_font_size=12
        )
        if len(player_data) > 3:
            x_numeric = list(range(len(player_data)))
            z = np.polyfit(x_numeric, y_values, 1)
            p = np.poly1d(z)
            trend_color = '#32CD32' if z[0] > 0 else '#FF6347'
            trend_text = "Improving" if z[0] > 0 else "Declining"
            fig.add_trace(go.Scatter(
                x=seasons,
                y=p(x_numeric),
                mode='lines',
                name=trend_text,
                line=dict(color=trend_color, width=3, dash='dash'),
                opacity=0.8,
                hovertemplate=f"<b>Trend Line</b><br>%{{y:.1f}}<extra></extra>"
            ))
        fig.update_layout(
            title=dict(
                text=f"{player_name} {display_stat} Career Progression",
                font=dict(size=20, color='#1f2937'),
                x=0.5
            ),
            xaxis=dict(
                title="Career Timeline",
                title_font=dict(size=14),
                tickfont=dict(size=12),
                gridcolor='lightgray',
                gridwidth=1
            ),
            yaxis=dict(
                title=display_stat,
                title_font=dict(size=14),
                tickfont=dict(size=12),
                gridcolor='lightgray',
                gridwidth=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif"),
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=12)
            ),
            height=500,
            margin=dict(l=60, r=60, t=80, b=60)
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
        stat_labels = {
            'PTS': 'Points Per Game', 'PPG': 'Points Per Game',
            'REB': 'Rebounds Per Game', 'RPG': 'Rebounds Per Game',
            'AST': 'Assists Per Game', 'APG': 'Assists Per Game',
            'STL': 'Steals Per Game', 'BLK': 'Blocks Per Game',
            'FG_PCT': 'Field Goal %', 'FT_PCT': 'Free Throw %',
            'MIN': 'Minutes Per Game'
        }
        display_stat = stat_labels.get(stat, stat)
        values = self._prepare_historical_data(data, stat)
        if len(values) == 0:
            return self._create_empty_chart(f"No data available for {display_stat}")
        values = pd.Series(values).dropna()
        mean_val = values.mean()
        median_val = values.median()
        max_val = values.max()
        min_val = values.min()
        std_val = values.std()
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=[
                f"{player_name} {display_stat} Distribution",
                "Performance Summary"
            ],
            specs=[
                [{"type": "xy"}],
                [{"type": "xy"}]
            ],
            vertical_spacing=0.15
        )
        colors = ['#FF6B6B' if x < mean_val * 0.9 else
                 '#4ECDC4' if x > mean_val * 1.1 else
                 '#45B7D1' for x in values]
        fig.add_trace(
            go.Histogram(
                x=values,
                nbinsx=min(15, len(values)),
                name="Games",
                opacity=0.8,
                marker=dict(
                    color=colors[0] if len(set(colors)) == 1 else '#45B7D1',
                    line=dict(width=1, color='white')
                ),
                hovertemplate=f"<b>{display_stat} Range</b><br>%{{x:.1f}}<br>Games: %{{y}}<extra></extra>"
            ),
            row=1, col=1
        )
        fig.add_vline(
            x=mean_val,
            line_dash="dot",
            line_color="#FF9500",
            line_width=3,
            annotation_text=f"Average: {mean_val:.1f}",
            annotation_position="top right",
            annotation_font=dict(size=12, color="#FF9500"),
            row=1, col=1
        )
        fig.add_vline(
            x=max_val,
            line_dash="dash",
            line_color="#32CD32",
            line_width=2,
            annotation_text=f"🔥 Best: {max_val:.1f}",
            annotation_position="top left",
            annotation_font=dict(size=10, color="#32CD32"),
            row=1, col=1
        )
        summary_stats = ['Career High', 'Career Low', 'Average']
        summary_values = [max_val, min_val, mean_val]
        summary_colors = ['#32CD32', '#FF6B6B', '#45B7D1']
        fig.add_trace(
            go.Bar(
                x=summary_stats,
                y=summary_values,
                marker_color=summary_colors,
                text=[f"{val:.1f}" for val in summary_values],
                textposition='outside',
                name="Career Summary"
            ),
            row=2, col=1
        )
        fig.update_layout(
            title=dict(
                text=f"{player_name} {display_stat} Analysis",
                font=dict(size=20, color='#1f2937'),
                x=0.5
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif"),
            height=600,
            showlegend=False,
            margin=dict(l=60, r=60, t=100, b=60)
        )
        fig.update_xaxes(
            title=f"{display_stat}",
            title_font=dict(size=14),
            tickfont=dict(size=12),
            gridcolor='lightgray',
            gridwidth=1,
            row=1, col=1
        )
        fig.update_yaxes(
            title="Number of Games",
            title_font=dict(size=14),
            tickfont=dict(size=12),
            gridcolor='lightgray',
            gridwidth=1,
            row=1, col=1
        )
        annotations = [
            dict(
                text=f"Career High: {max_val:.1f}<br>Career Low: {min_val:.1f}<br>Average: {mean_val:.1f}",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                xanchor="left", yanchor="top",
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="gray",
                borderwidth=1,
                font=dict(size=11),
                showarrow=False
            )
        ]
        fig.update_layout(annotations=annotations)
        return fig
    def create_player_performance_summary(self, player_data: pd.DataFrame, player_name: str) -> go.Figure:
        if player_data.empty:
            return self._create_empty_chart(f"No data available for {player_name}")
        key_stats = ['PPG', 'RPG', 'APG']
        actual_stats = []
        values = []
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        for stat in key_stats:
            stat_data = self._prepare_historical_data(player_data, stat)
            if len(stat_data) > 0:
                actual_stats.append(stat)
                values.append(np.mean(stat_data))
        if not actual_stats:
            return self._create_empty_chart(f"No key statistics available for {player_name}")
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Career Averages",
                "Games Played",
                "Statistical Breakdown",
                "Performance Metrics"
            ],
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        fig.add_trace(
            go.Bar(
                x=actual_stats,
                y=values,
                marker_color=colors[:len(actual_stats)],
                text=[f"{val:.1f}" for val in values],
                textposition='outside',
                name="Career Averages",
                hovertemplate="<b>%{x}</b><br>Average: %{y:.1f}<extra></extra>"
            ),
            row=1, col=1
        )
        seasons = list(range(1, len(player_data) + 1)) if len(player_data) > 0 else [1]
        games_played = player_data['GP'].tolist() if 'GP' in player_data.columns else [0] * len(seasons)
        fig.add_trace(
            go.Bar(
                x=seasons,
                y=games_played,
                name="Games Played",
                marker_color='#4ECDC4',
                text=games_played,
                textposition='outside'
            ),
            row=1, col=2
        )
        breakdown_stats = actual_stats.copy()
        breakdown_values = values.copy()
        breakdown_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(breakdown_stats)]
        if len(values) >= 2:
            consistency = max(0, 100 - (np.std(values) / np.mean(values) * 50))
            breakdown_stats.append('Consistency')
            breakdown_values.append(consistency)
            breakdown_colors.append('#FFD700')
        fig.add_trace(
            go.Bar(
                y=breakdown_stats,
                x=breakdown_values,
                orientation='h',
                marker_color=breakdown_colors,
                text=[f"{val:.1f}" for val in breakdown_values],
                textposition='outside',
                name="Statistical Profile",
                hovertemplate="<b>%{y}</b><br>Value: %{x:.1f}<extra></extra>"
            ),
            row=2, col=1
        )
        metrics = ['Average', 'Best Season', 'Consistency']
        avg_val = np.mean(values) if values else 0
        best_val = max(values) if values else 0
        consistency_val = max(0, 100 - (np.std(values) / np.mean(values) * 50)) if len(values) > 1 and np.mean(values) > 0 else 70
        metric_values = [avg_val, best_val, consistency_val]
        colors = ['#45B7D1', '#32CD32', '#FFD700']
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=metric_values,
                marker_color=colors,
                text=[f"{val:.1f}" for val in metric_values],
                textposition='outside',
                name="Performance Metrics"
            ),
            row=2, col=2
        )
        fig.update_layout(
            title=dict(
                text=f"{player_name} Performance Overview",
                font=dict(size=22, color='#1f2937'),
                x=0.5
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif"),
            height=700,
            showlegend=False,
            margin=dict(l=60, r=60, t=100, b=60)
        )
        fig.update_xaxes(title_font=dict(size=12), tickfont=dict(size=10), row=1, col=1)
        fig.update_yaxes(title_font=dict(size=12), tickfont=dict(size=10), row=1, col=1)
        fig.update_xaxes(title="Performance Score", title_font=dict(size=12), tickfont=dict(size=10), row=2, col=1)
        fig.update_yaxes(title="Statistics", title_font=dict(size=12), tickfont=dict(size=10), row=2, col=1)
        return fig
    def _create_franchise_history_chart(self, team_data: pd.DataFrame, team_name: str) -> go.Figure:
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