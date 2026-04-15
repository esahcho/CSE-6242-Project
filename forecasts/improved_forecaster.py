"""
IMPROVED GRADIENT BOOSTING FORECAST WITH SEASONAL COMPONENTS
Implementation guide for enhancing annual pattern representation

This module provides enhanced feature engineering and a hybrid ensemble approach
that combines Gradient Boosting with explicit seasonal components.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor


class SeasonalGradientBoostingForecaster:
    """
    Hybrid forecasting model that combines:
    1. Seasonal baseline (historical daily averages)
    2. Gradient Boosting model (captures anomalies/variability)
    3. Explicit seasonal features (sun altitude, solar declination)
    """
    
    def __init__(self, n_components=45):
        self.n_components = n_components
        self.seasonal_baseline = None
        self.model_gb = None
        self.pca = None
        self.scaler_X = None
        self.residual_mean = None
        self.residual_std = None
        
    def calculate_seasonal_baseline(self, ghi_series):
        """
        Calculate the expected GHI for each day-of-year based on historical patterns.
        Returns a dictionary with seasonal statistics.
        """
        seasonal_stats = {}
        
        # Group by day-of-year across all years
        ghi_with_doy = pd.DataFrame({
            'ghi': ghi_series,
            'doy': ghi_series.index.dayofyear
        })
        
        grouped = ghi_with_doy.groupby('doy')['ghi'].agg([
            'mean', 'std', 'min', 'max', 'count'
        ])
        
        grouped['amplitude'] = grouped['max'] - grouped['min']
        grouped['median'] = ghi_with_doy.groupby('doy')['ghi'].median()
        
        # Convert to dictionary for easy lookup
        for doy in grouped.index:
            seasonal_stats[doy] = {
                'mean': grouped.loc[doy, 'mean'],
                'std': grouped.loc[doy, 'std'],
                'min': grouped.loc[doy, 'min'],
                'max': grouped.loc[doy, 'max'],
                'amplitude': grouped.loc[doy, 'amplitude'],
                'count': grouped.loc[doy, 'count']
            }
        
        self.seasonal_baseline = seasonal_stats
        return seasonal_stats
    
    def calculate_hourly_pattern(self, ghi_series):
        """
        Calculate the typical hourly profile (how GHI varies by hour of day).
        Returns normalized hour-of-day pattern.
        """
        hourly_avg = ghi_series.groupby(ghi_series.index.hour).mean()
        hourly_pattern = hourly_avg / hourly_avg.mean() if hourly_avg.mean() > 0 else hourly_avg
        return hourly_pattern.to_dict()
    
    def build_enhanced_features(self, features_df, ghi_series, seasonal_baseline):
        """
        Add enhanced seasonal features to the feature dataframe.
        
        Parameters:
        -----------
        features_df : DataFrame
            Original features (with lag and rolling stats)
        ghi_series : Series
            Historical GHI series
        seasonal_baseline : dict
            Output from calculate_seasonal_baseline()
        
        Returns:
        --------
        features_df_enhanced : DataFrame
            Features with new seasonal components
        """
        features_df = features_df.copy()
        idx = features_df.index
        
        # 1. SEASONAL BASELINE FEATURES
        # Extract expected values for each day-of-year
        doy_array = idx.dayofyear
        
        seasonal_means = np.array([
            seasonal_baseline.get(doy, {}).get('mean', 0) for doy in doy_array
        ])
        seasonal_amplitude = np.array([
            seasonal_baseline.get(doy, {}).get('amplitude', 0) for doy in doy_array
        ])
        seasonal_max = np.array([
            seasonal_baseline.get(doy, {}).get('max', 0) for doy in doy_array
        ])
        seasonal_min = np.array([
            seasonal_baseline.get(doy, {}).get('min', 0) for doy in doy_array
        ])
        
        features_df['seasonal_baseline_mean'] = seasonal_means
        features_df['seasonal_baseline_amplitude'] = seasonal_amplitude
        features_df['expected_daily_peak'] = seasonal_max
        features_df['expected_daily_min'] = seasonal_min
        features_df['expected_daily_range'] = seasonal_max - seasonal_min
        
        # 2. AMPLITUDE SCALING FACTOR (relative to annual average)
        mean_amplitude = np.mean([v['amplitude'] for v in seasonal_baseline.values()])
        features_df['amplitude_scaling_factor'] = (
            seasonal_amplitude / mean_amplitude if mean_amplitude > 0 else 1.0
        )
        
        # 3. PHYSICAL BASIS FEATURES (Sun position approximation)
        doy_norm = (doy_array - 1) / 365.0  # Normalized day of year
        
        # Solar declination (sun's latitude): varies from -23.5° to +23.5°
        # Formula: δ = 23.45 * sin(2π * (day_of_year - 81) / 365)
        features_df['solar_declination_approx'] = (
            23.5 * np.sin(2 * np.pi * (doy_array - 81) / 365)
        )
        
        # Sun altitude proxy: peaks around day 172 (summer solstice for Northern hemisphere)
        # Higher values = higher sun = more potential irradiance
        features_df['sun_altitude_proxy'] = (
            np.cos(2 * np.pi * (doy_array - 172) / 365)
        )
        
        # Relative to climatological maximum (day length effect)
        features_df['season_length_factor'] = (
            np.sin(2 * np.pi * doy_norm) ** 2  # Smoother than hard boundaries
        )
        
        # 4. SEASONAL LAG INTERACTIONS (scaled by seasonal amplitude)
        # This helps the model understand that lags should be scaled differently in different seasons
        short_lags = [24, 48, 168]  # 1 day, 2 days, 1 week
        
        for lag in short_lags:
            lag_col = f'GHI_lag_{lag}'
            if lag_col in features_df.columns:
                # Interaction: how does this lag relate to seasonal amplitude?
                features_df[f'{lag_col}_x_amplitude_scale'] = (
                    features_df[lag_col] * features_df['amplitude_scaling_factor']
                )
                # Interaction with time of year
                features_df[f'{lag_col}_x_solar_declination'] = (
                    features_df[lag_col] * features_df['solar_declination_approx'] / 23.5
                )
        
        # 5. MONTH INTERACTION FEATURES
        # Help model learn season-specific relationships
        month_sin = np.sin(2 * np.pi * (idx.month - 1) / 12)
        month_cos = np.cos(2 * np.pi * (idx.month - 1) / 12)
        
        if 'GHI_lag_24' in features_df.columns:
            features_df['GHI_lag_24_x_month_sin'] = features_df['GHI_lag_24'] * month_sin
            features_df['GHI_lag_24_x_month_cos'] = features_df['GHI_lag_24'] * month_cos
        
        return features_df
    
    def build_future_features_enhanced(self, target_date, history_ghi, seasonal_baseline, 
                                      short_lags, seasonal_lags, rolling_windows):
        """
        Build feature vector for future prediction with enhanced seasonal components.
        
        Parameters:
        -----------
        target_date : Timestamp
            Date to forecast
        history_ghi : list
            Historical GHI values including recursive predictions
        seasonal_baseline : dict
            Seasonal statistics
        short_lags, seasonal_lags, rolling_windows : lists
            Lag and window configuration
        
        Returns:
        --------
        features : dict
            Feature vector for prediction
        """
        features = {}
        doy = target_date.dayofyear
        hour = target_date.hour
        
        # === SHORT-TERM FEATURES (existing) ===
        for lag in short_lags:
            if lag < len(history_ghi):
                features[f'GHI_lag_{lag}'] = history_ghi[-lag]
            else:
                features[f'GHI_lag_{lag}'] = np.mean(history_ghi) if history_ghi else 0
        
        # === SEASONAL LAG FEATURES (existing) ===
        for lag in seasonal_lags:
            if lag < len(history_ghi):
                features[f'GHI_seasonal_lag_{lag}'] = history_ghi[-lag]
            else:
                features[f'GHI_seasonal_lag_{lag}'] = np.mean(history_ghi) if history_ghi else 0
        
        # === ROLLING STATISTICS (existing) ===
        for window in rolling_windows:
            if window < len(history_ghi):
                window_values = np.array(history_ghi[-window:])
                features[f'GHI_rolling_mean_{window}'] = window_values.mean()
                features[f'GHI_rolling_std_{window}'] = window_values.std(ddof=0)
                features[f'GHI_rolling_min_{window}'] = window_values.min()
                features[f'GHI_rolling_max_{window}'] = window_values.max()
            else:
                window_values = np.array(history_ghi)
                features[f'GHI_rolling_mean_{window}'] = window_values.mean()
                features[f'GHI_rolling_std_{window}'] = window_values.std(ddof=0)
                features[f'GHI_rolling_min_{window}'] = window_values.min()
                features[f'GHI_rolling_max_{window}'] = window_values.max()
        
        # === CYCLICAL FEATURES (existing) ===
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        features['week_sin'] = np.sin(2 * np.pi * target_date.isocalendar().week / 52)
        features['week_cos'] = np.cos(2 * np.pi * target_date.isocalendar().week / 52)
        features['month_sin'] = np.sin(2 * np.pi * (target_date.month - 1) / 12)
        features['month_cos'] = np.cos(2 * np.pi * (target_date.month - 1) / 12)
        features['doy_sin'] = np.sin(2 * np.pi * doy / 365)
        features['doy_cos'] = np.cos(2 * np.pi * doy / 365)
        
        # Categorical indicators
        features['month'] = target_date.month
        features['day_of_week'] = target_date.dayofweek
        features['is_summer'] = 1.0 if target_date.month in [6, 7, 8] else 0.0
        features['is_winter'] = 1.0 if target_date.month in [12, 1, 2] else 0.0
        
        # Differencing features
        if len(history_ghi) > 25:
            features['GHI_diff_1h'] = history_ghi[-1] - history_ghi[-2]
            features['GHI_diff_24h'] = history_ghi[-1] - history_ghi[-25]
        else:
            features['GHI_diff_1h'] = 0
            features['GHI_diff_24h'] = 0
        
        # === NEW: SEASONAL BASELINE FEATURES ===
        seasonal_stats = seasonal_baseline.get(doy, {})
        features['seasonal_baseline_mean'] = seasonal_stats.get('mean', 0)
        features['seasonal_baseline_amplitude'] = seasonal_stats.get('amplitude', 0)
        features['expected_daily_peak'] = seasonal_stats.get('max', 0)
        features['expected_daily_min'] = seasonal_stats.get('min', 0)
        features['expected_daily_range'] = features['expected_daily_peak'] - features['expected_daily_min']
        
        # === NEW: AMPLITUDE SCALING ===
        all_amplitudes = [v.get('amplitude', 0) for v in seasonal_baseline.values()]
        mean_amplitude = np.mean(all_amplitudes) if all_amplitudes else 1.0
        amplitude_scaling = (seasonal_stats.get('amplitude', 0) / mean_amplitude) if mean_amplitude > 0 else 1.0
        features['amplitude_scaling_factor'] = amplitude_scaling
        
        # === NEW: PHYSICAL BASIS FEATURES ===
        features['solar_declination_approx'] = 23.5 * np.sin(2 * np.pi * (doy - 81) / 365)
        features['sun_altitude_proxy'] = np.cos(2 * np.pi * (doy - 172) / 365)
        features['season_length_factor'] = np.sin(2 * np.pi * (doy - 1) / 365) ** 2
        
        # === NEW: INTERACTION FEATURES ===
        for lag in [24, 48, 168]:
            if lag < len(history_ghi):
                lag_val = history_ghi[-lag]
                features[f'GHI_lag_{lag}_x_amplitude_scale'] = lag_val * amplitude_scaling
                features[f'GHI_lag_{lag}_x_solar_declination'] = (
                    lag_val * features['solar_declination_approx'] / 23.5
                )
        
        features['GHI_lag_24_x_month_sin'] = features.get('GHI_lag_24', 0) * features['month_sin']
        features['GHI_lag_24_x_month_cos'] = features.get('GHI_lag_24', 0) * features['month_cos']
        
        return features
    
    def fit(self, city_ghi_series, features_df):
        """
        Fit the forecaster on historical data.
        
        Parameters:
        -----------
        city_ghi_series : Series
            Historical GHI values with DatetimeIndex
        features_df : DataFrame
            Feature matrix (with added seasonal features)
        """
        # Calculate seasonal baseline
        self.calculate_seasonal_baseline(city_ghi_series)
        
        # Build enhanced features
        features_df_enhanced = self.build_enhanced_features(
            features_df, city_ghi_series, self.seasonal_baseline
        )
        
        # Calculate residuals from seasonal baseline
        seasonal_fitted = pd.Series([
            self.seasonal_baseline.get(idx.dayofyear, {}).get('mean', 0) 
            for idx in city_ghi_series.index
        ], index=city_ghi_series.index)
        
        residuals = city_ghi_series - seasonal_fitted
        self.residual_mean = residuals.mean()
        self.residual_std = residuals.std()
        
        # Prepare training data
        X_train = features_df_enhanced.drop(columns=['GHI'], errors='ignore')
        y_train = residuals.dropna()
        X_train = X_train.loc[y_train.index]
        
        # PCA
        self.pca = PCA(n_components=self.n_components, random_state=42)
        X_train_pca = self.pca.fit_transform(X_train)
        
        # Scale
        self.scaler_X = StandardScaler()
        X_train_scaled = self.scaler_X.fit_transform(X_train_pca)
        
        # Train Gradient Boosting (on residuals!)
        self.model_gb = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.12,
            subsample=0.85,
            max_features='sqrt',
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.model_gb.fit(X_train_scaled, y_train.values)
    
    def predict(self, target_date, history_ghi, feature_columns, 
                short_lags, seasonal_lags, rolling_windows):
        """
        Make prediction for a single timestamp using hybrid approach.
        
        Formula:
        --------
        Forecast = Seasonal_Baseline + Residual_Model_Prediction
        
        Returns:
        --------
        float : Predicted GHI (non-negative)
        """
        # Get enhanced feature vector
        future_features = self.build_future_features_enhanced(
            target_date, history_ghi, self.seasonal_baseline,
            short_lags, seasonal_lags, rolling_windows
        )
        
        # Convert to array matching training feature order
        feature_vector = np.array([
            future_features.get(col, 0) for col in feature_columns
        ]).reshape(1, -1)
        
        # PCA transform
        feature_pca = self.pca.transform(feature_vector)
        
        # Scale
        feature_scaled = self.scaler_X.transform(feature_pca)
        
        # Predict residual
        residual_pred = self.model_gb.predict(feature_scaled)[0]
        
        # Get seasonal baseline for this day-of-year
        doy = target_date.dayofyear
        seasonal_baseline_val = self.seasonal_baseline.get(doy, {}).get('mean', 0)
        
        # Combine: seasonal + residual
        forecast_combined = seasonal_baseline_val + residual_pred
        
        # Ensure non-negative
        forecast_final = max(0.0, forecast_combined)
        
        return forecast_final


# ============================================================================
# USAGE EXAMPLE IN YOUR NOTEBOOK
# ============================================================================
"""
# In your model training section (replacing the current GradientBoostingRegressor code):

for city in unique_cities:
    print(f"\\nProcessing city: {city}...")
    
    # ... [existing code: filter city data, build features_df] ...
    
    # IMPROVED: Use SeasonalGradientBoostingForecaster
    forecaster = SeasonalGradientBoostingForecaster(n_components=45)
    forecaster.fit(city_ghi_series, features_df)
    
    # Store forecaster for later use
    forecasters_by_city[city] = forecaster
    
    # ... [existing code for metrics calculation] ...
    
    # IMPROVED: Recursive forecast with new approach
    history_ghi = city_ghi_series.tolist()
    
    for ts in future_index:
        # Use the hybrid forecaster
        pred_ghi = forecaster.predict(
            ts, history_ghi, 
            X_train.columns,  # Feature column names
            short_lags=[1, 2, 3, 6, 12, 24, 48, 72, 168],
            seasonal_lags=[730, 1460, 2190, 8760],
            rolling_windows=[6, 24, 48, 168, 336, 732]
        )
        
        history_ghi.append(pred_ghi)
        
        all_results.append({
            'city': city,
            'date': ts,
            'actual_ghi': np.nan,
            'forecasted_ghi': pred_ghi,  # Now includes seasonal component!
            'mae': train_mae,
            'rmse': train_rmse,
            'mape': train_mape,
            'data_type': 'forecast'
        })
"""
