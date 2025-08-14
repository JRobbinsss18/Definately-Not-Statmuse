import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class MLPredictor:
    def __init__(self):
        self.models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'lightgbm': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        }
        self.scalers = {}
        self.trained_models = {}
        
    def prepare_player_data(self, career_stats: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        if career_stats.empty:
            return pd.DataFrame(), []
            
        # Select key features for prediction
        feature_columns = [
            'GP', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
            'FTM', 'FTA', 'FT_PCT', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'
        ]
        
        # Filter available columns
        available_features = [col for col in feature_columns if col in career_stats.columns]
        
        if not available_features:
            return pd.DataFrame(), []
            
        # Create feature matrix
        feature_data = career_stats[available_features].copy()
        
        for col in feature_data.columns:
            feature_data[col] = feature_data[col].fillna(feature_data[col].mean())
        
        # Add derived features
        if 'GP' in feature_data.columns and 'MIN' in feature_data.columns:
            feature_data['MIN_PER_GAME'] = feature_data['MIN'] / np.maximum(feature_data['GP'], 1)
        
        if 'PTS' in feature_data.columns and 'GP' in feature_data.columns:
            feature_data['PPG'] = feature_data['PTS'] / np.maximum(feature_data['GP'], 1)
        
        if 'REB' in feature_data.columns and 'GP' in feature_data.columns:
            feature_data['RPG'] = feature_data['REB'] / np.maximum(feature_data['GP'], 1)
            
        if 'AST' in feature_data.columns and 'GP' in feature_data.columns:
            feature_data['APG'] = feature_data['AST'] / np.maximum(feature_data['GP'], 1)
        
        # Add time-based features
        feature_data['SEASON_NUMBER'] = range(len(feature_data))
        feature_data['CAREER_YEAR'] = feature_data['SEASON_NUMBER'] + 1
        
        return feature_data, list(feature_data.columns)
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, target_stat: str) -> Dict[str, float]:
        if len(X) < 3:
            return {}
        
        if target_stat in ['FG_PCT', 'FG3_PCT', 'FT_PCT']:
            valid_mask = (y >= 0.0) & (y <= 1.0) & (~y.isna())
            if valid_mask.sum() < 3:
                return {}
            X = X[valid_mask]
            y = y[valid_mask]
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, shuffle=False
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers[target_stat] = scaler
        
        model_scores = {}
        trained_models = {}
        
        for model_name, model in self.models.items():
            try:
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                
                # Calculate scores
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                model_scores[model_name] = {
                    'mae': mae,
                    'r2': r2,
                    'score': r2 - (mae / (y_test.mean() + 1e-8))
                }
                
                trained_models[model_name] = model
                
            except Exception:
                continue
        
        self.trained_models[target_stat] = trained_models
        return model_scores
    
    def predict_next_season(self, player_data: pd.DataFrame, target_stats: List[str]) -> Dict[str, Dict]:
        predictions = {}
        
        # Prepare data once outside the loop
        feature_data, feature_names = self.prepare_player_data(player_data)
        if feature_data.empty:
            return predictions
        
        for stat in target_stats:
            # Check if stat exists in either original data or prepared feature data
            if stat not in player_data.columns and stat not in feature_data.columns:
                continue
                
            # Use target values from feature_data if available (for derived stats like PPG),
            # otherwise use original player_data
            if stat in feature_data.columns:
                target_values = feature_data[stat].fillna(feature_data[stat].mean())
            else:
                target_values = player_data[stat].fillna(player_data[stat].mean())
            
            # Train models
            model_scores = self.train_models(feature_data, target_values, stat)
            if not model_scores:
                prediction = self._simple_trend_prediction(target_values, stat)
                predictions[stat] = {
                    'prediction': prediction,
                    'ensemble_mean': prediction,
                    'confidence_interval': (prediction * 0.9, prediction * 1.1),
                    'confidence_percentage': 60.0,
                    'best_model': 'trend_fallback',
                    'model_scores': {},
                    'trend_direction': self._calculate_trend(target_values),
                    'historical_avg': target_values.mean()
                }
                continue
            
            # Select best model
            best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x]['score'])
            best_model = self.trained_models[stat][best_model_name]
            
            # Create next season features (extrapolate based on trend)
            next_season_features = self._extrapolate_features(feature_data)
            
            # Scale and predict
            if stat in self.scalers:
                next_season_scaled = self.scalers[stat].transform(next_season_features.values.reshape(1, -1))
                prediction = best_model.predict(next_season_scaled)[0]
                if stat in ['FG_PCT', 'FG3_PCT', 'FT_PCT']:
                    prediction = np.clip(prediction, 0.0, 1.0)
            else:
                prediction = target_values.iloc[-1]  # Fallback to last season
            
            # Get ensemble prediction from all models
            ensemble_predictions = []
            for model_name, model in self.trained_models[stat].items():
                try:
                    next_season_scaled = self.scalers[stat].transform(next_season_features.values.reshape(1, -1))
                    pred = model.predict(next_season_scaled)[0]
                    if stat in ['FG_PCT', 'FG3_PCT', 'FT_PCT']:
                        pred = np.clip(pred, 0.0, 1.0)
                    ensemble_predictions.append(pred)
                except:
                    continue
            
            ensemble_mean = np.mean(ensemble_predictions) if ensemble_predictions else prediction
            ensemble_std = np.std(ensemble_predictions) if len(ensemble_predictions) > 1 else abs(prediction * 0.1)
            
            # Calculate confidence percentage based on model agreement
            # Higher agreement between models = higher confidence
            if ensemble_predictions:
                prediction_std = np.std(ensemble_predictions)
                prediction_mean = np.mean(ensemble_predictions)
            else:
                prediction_std = abs(prediction * 0.1)  # 10% of prediction as fallback
                prediction_mean = prediction
            
            # Calculate confidence as inverse of relative standard deviation
            if prediction_mean != 0:
                coefficient_of_variation = prediction_std / abs(prediction_mean)
                confidence_percentage = max(60, min(95, 90 - (coefficient_of_variation * 100)))
            else:
                confidence_percentage = 75  # Default confidence for zero predictions
            
            predictions[stat] = {
                'prediction': prediction,
                'ensemble_mean': ensemble_mean,
                'confidence_interval': (ensemble_mean - ensemble_std, ensemble_mean + ensemble_std),
                'confidence_percentage': round(confidence_percentage, 1),
                'best_model': best_model_name,
                'model_scores': model_scores,
                'trend_direction': self._calculate_trend(target_values),
                'historical_avg': target_values.mean()
            }
        
        return predictions
    
    def predict_multiple_seasons(self, career_stats: pd.DataFrame, target_stats: List[str], num_seasons: int = 3) -> Dict:
        if career_stats.empty or num_seasons < 1 or num_seasons > 10:
            return {}
            
        predictions = {}
        
        # Prepare data once to check available stats
        feature_data, feature_names = self.prepare_player_data(career_stats)
        if feature_data.empty:
            return {}
        
        for stat in target_stats:
            # Check if stat exists in either original data or prepared feature data
            if stat not in career_stats.columns and stat not in feature_data.columns:
                continue
                
            # Use single season prediction as base
            base_prediction = self.predict_next_season(career_stats, [stat])
            
            if stat in base_prediction and isinstance(base_prediction[stat], dict):
                try:
                    stat_predictions = []
                    base_value = base_prediction[stat]['ensemble_mean']
                    base_confidence = base_prediction[stat]['confidence_interval']
                    
                    # Ensure base_confidence is a tuple/list
                    if not isinstance(base_confidence, (tuple, list)):
                        base_confidence = (base_value * 0.9, base_value * 1.1)  # 10% range fallback
                    
                    # Create a safe copy for the loop
                    conf_tuple = tuple(base_confidence)
                except (KeyError, TypeError):
                    continue
                
                # Calculate aging curve and trend
                recent_seasons = min(5, len(career_stats))
                if recent_seasons >= 3:
                    # Use feature_data for derived stats (PPG, RPG, APG), otherwise use career_stats
                    if stat in feature_data.columns:
                        recent_values = feature_data[stat].tail(recent_seasons)
                    else:
                        recent_values = career_stats[stat].tail(recent_seasons)
                    if hasattr(recent_values, 'values'):
                        # It's a pandas Series
                        recent_values_array = recent_values.values
                    else:
                        # It's already a numpy array or scalar
                        recent_values_array = np.array([recent_values]) if np.isscalar(recent_values) else recent_values
                    
                    if len(recent_values_array) >= 3:
                        trend = np.polyfit(range(len(recent_values_array)), recent_values_array, 1)[0]
                    else:
                        trend = 0
                else:
                    trend = 0
                
                # Project multiple seasons with aging considerations
                for season_num in range(1, num_seasons + 1):
                    # Apply aging curve (slight decline after age/experience)
                    aging_factor = max(0.95, 1.0 - (season_num - 1) * 0.025)  # 2.5% decline per year
                    trend_factor = trend * 0.5 * season_num  # Dampen trend over time
                    
                    projected_value = (base_value + trend_factor) * aging_factor
                    
                    # Adjust confidence interval (wider for future seasons)
                    confidence_width = (conf_tuple[1] - conf_tuple[0]) * (1 + season_num * 0.2)
                    confidence_interval = (
                        max(0, projected_value - confidence_width/2),
                        projected_value + confidence_width/2
                    )
                    
                    # Calculate confidence percentage for multi-season (decreases with time)
                    base_confidence = base_prediction[stat].get('confidence_percentage', 75)
                    season_confidence = base_confidence * (0.9 ** (season_num - 1))  # 10% reduction per season
                    
                    stat_predictions.append({
                        'season': season_num,
                        'projected_value': round(projected_value, 1),
                        'confidence_interval': (round(confidence_interval[0], 1), round(confidence_interval[1], 1)),
                        'confidence_percentage': round(season_confidence, 1),
                        'aging_factor': round(aging_factor, 3),
                        'trend_influence': round(trend_factor, 1)
                    })
                
                predictions[stat] = {
                    'multi_season_projections': stat_predictions,
                    'base_prediction': base_prediction[stat],
                    'projection_type': f'{num_seasons}_season_outlook',
                    'trend_direction': 'improving' if trend > 0 else 'declining' if trend < -0.1 else 'stable'
                }
        
        return predictions
    
    def _extrapolate_features(self, feature_data: pd.DataFrame) -> pd.Series:
        next_features = feature_data.iloc[-1].copy()
        
        if 'CAREER_YEAR' in next_features:
            next_features['CAREER_YEAR'] += 1
        if 'SEASON_NUMBER' in next_features:
            next_features['SEASON_NUMBER'] += 1
        
        percentage_stats = ['FG_PCT', 'FG3_PCT', 'FT_PCT']
        
        for col in feature_data.columns:
            if col not in ['CAREER_YEAR', 'SEASON_NUMBER'] and len(feature_data) >= 3:
                recent_values = feature_data[col].tail(3).values
                if len(recent_values) >= 2:
                    trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
                    projected_value = recent_values[-1] + trend
                    if col in percentage_stats:
                        projected_value = np.clip(projected_value, 0.0, 1.0)
                    next_features[col] = projected_value
        
        return next_features
    
    def _simple_trend_prediction(self, values: pd.Series, stat: str) -> float:
        if len(values) < 1:
            return 0.0
        if len(values) < 2:
            return values.iloc[-1]
        
        recent_values = values.tail(3).values
        if len(recent_values) >= 2 and not np.any(np.isnan(recent_values)):
            try:
                if stat in ['FG_PCT', 'FG3_PCT', 'FT_PCT']:
                    trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0] * 0.5
                else:
                    trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
                
                prediction = recent_values[-1] + trend
                
                if stat in ['FG_PCT', 'FG3_PCT', 'FT_PCT']:
                    prediction = np.clip(prediction, 0.25, 0.65)
                elif prediction < 0:
                    prediction = max(0, values.iloc[-1])
                
                return prediction
            except:
                pass
        
        return values.iloc[-1]
    
    def _calculate_trend(self, values: pd.Series) -> str:
        if len(values) < 2:
            return "insufficient_data"
        recent_trend = np.polyfit(range(len(values.tail(3))), values.tail(3).values, 1)[0]
        
        if values.max() <= 1.0 and values.min() >= 0.0:
            threshold = 0.01
        else:
            threshold = 0.05
        
        if recent_trend > threshold:
            return "improving"
        elif recent_trend < -threshold:
            return "declining"
        return "stable"
    
    def get_prediction_summary(self, predictions: Dict[str, Dict]) -> str:
        if not predictions:
            return "Unable to generate predictions due to insufficient data."
        
        summary = "PREDICTION ANALYSIS:\n\n"
        
        for stat, pred_data in predictions.items():
            # Handle both single season and multi-season predictions
            if 'multi_season_projections' in pred_data:
                # Multi-season prediction
                projections = pred_data['multi_season_projections']
                trend = pred_data.get('trend_direction', 'stable')
                
                summary += f"{stat} Multi-Season Projection (trend: {trend}):\n"
                for proj in projections:
                    season = proj['season']
                    value = proj['projected_value']
                    conf_pct = proj.get('confidence_percentage', 75)
                    summary += f"  Season {season}: {value:.1f} ({conf_pct}% confidence)\n"
                summary += "\n"
            else:
                # Single season prediction
                prediction = pred_data.get('prediction', pred_data.get('ensemble_mean', 0))
                trend = pred_data.get('trend_direction', 'stable')
                confidence_pct = pred_data.get('confidence_percentage', 75)
                best_model = pred_data.get('best_model', 'ensemble')
                
                summary += f"{stat}: {prediction:.1f} (trend: {trend})\n"
                summary += f"  Confidence: {confidence_pct}%\n"
                summary += f"  Best model: {best_model}\n\n"
        
        return summary