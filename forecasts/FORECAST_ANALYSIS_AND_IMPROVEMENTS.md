# Solar Irradiance Forecast Model - Analysis & Improvements for Annual Patterns

## Executive Summary

The current Gradient Boosting model performs well on day/night cycles but **fails to properly represent annual seasonal patterns**. The forecasts show relatively flat trends throughout 2025, without the expected amplitude variations between summer and winter months.

---

## Problem Analysis

### What's Working
✓ **Hour-level (day/night) cycles**: Clearly captured via cyclical encodings (hour_sin/cos)
✓ **Week-to-week patterns**: The 168-hour lag and weekly features help
✓ **Model performance metrics**: MAPE ~8-13% is acceptable for short-term patterns
✓ **Recent history capture**: Short lags (1-72 hours) work well

### What's Not Working
✗ **Annual/seasonal amplitude**: No clear difference between expected summer (high) and winter (low) peaks
✗ **Long-term trend extrapolation**: Tree-based models struggle to extrapolate beyond training domain
✗ **Seasonal lag limitations**: The 8760-hour lag (annual replay) doesn't work for out-of-sample forecasting
✗ **Feature interaction**: month_sin/cos exist but Gradient Boosting isn't strongly learning their coupling with peak magnitudes

---

## Root Causes

### 1. **Nature of Gradient Boosting**
- Tree-based models are excellent for **interpolation** but poor for **extrapolation**
- They learn patterns within the training domain but struggle to generalize beyond it
- The model has seen winter/spring/summer/fall patterns in 2021-2024, but forecasting 2025 requires extrapolation

### 2. **Insufficient Seasonal Signal**
Current seasonal features:
- `month_sin/cos` (month cyclical encoding)
- `doy_sin/cos` (day-of-year cyclical encoding)
- `is_summer`, `is_winter` (binary flags)
- `seasonal_lags` at 730, 1460, 2190, 8760 hours

**Problem**: These features encode *when* in the annual cycle we are, but NOT the expected *magnitude* for that time.

### 3. **Time Series with Limited Historical Depth**
- Only 3 years of training data (2021-2024)
- Only forecasting Q1 2025 (non-overlapping with training)
- Model hasn't learned strong year-over-year amplitude patterns

---

## Solution: 6-Step Improvement Strategy

### SOLUTION 1: Add Seasonal Baseline Component ⭐⭐⭐⭐⭐ (HIGHEST PRIORITY)

**What it does**: Explicitly teaches the model the expected GHI magnitude for each day-of-year across all locations.

**Implementation**:
```python
# For each location, calculate historical average for each day-of-year
seasonal_baseline = df.groupby(df.index.dayofyear)['GHI'].agg(['mean', 'std', 'min', 'max'])
seasonal_baseline.index.name = 'doy'

# Add to features:
# - seasonal_baseline_mean[doy] → expected average for this day
# - seasonal_baseline_std[doy] → variability for this day
# - seasonal_baseline_amplitude[doy] → (max-min from historical) for this day
```

**Why this helps**: Directly injects location-specific seasonal expectations into the model.

---

### SOLUTION 2: Hybrid Ensemble Model ⭐⭐⭐⭐⭐ (HIGHEST PRIORITY)

**What it does**: Separates the forecast into seasonal and residual components, then recombines.

**Formula**:
```
Forecast = Seasonal_Component + Residual_Component
         = (historical_daily_avg for doy) + (GBModel prediction for anomaly)
```

**Implementation Steps**:

1. **Calculate seasonal component**:
   ```python
   # For each hour in forecast period:
   doy = forecast_timestamp.dayofyear
   seasonal_ghi = df[df.index.dayofyear == doy]['GHI'].mean()
   # Also use historical hour-of-day pattern
   hour = forecast_timestamp.hour
   hourly_factor = df[df.index.hour == hour]['GHI'].mean() / daily_avg
   seasonal = seasonal_ghi * hourly_factor
   ```

2. **Train GB model on residuals**:
   ```python
   residuals = actual_ghi - seasonal_baseline
   # Train GB on residuals instead of raw GHI
   gb_model.fit(X_train, residuals)
   ```

3. **Final forecast**:
   ```python
   seasonal_part = get_seasonal_baseline(forecast_date)
   residual_part = gb_model.predict(X_forecast)
   final_forecast = seasonal_part + residual_part
   final_forecast = np.maximum(final_forecast, 0)  # Non-negative
   ```

**Why this helps**: Separates what the model should memorize (annual pattern) from what it should learn (anomalies/variability).

---

### SOLUTION 3: Add Seasonal Amplitude Interaction Features ⭐⭐⭐⭐

**What it does**: Create explicit features that encode how amplitude scales with season.

**New Features to Add**:
```python
# 1. Seasonal amplitude scaling
features['seasonal_amplitude_factor'] = seasonal_baseline_amplitude[doy] / seasonal_baseline_amplitude.mean()

# 2. Expected peak value for this day-of-year
features['expected_daily_peak'] = seasonal_baseline_max[doy]
features['expected_daily_min'] = seasonal_baseline_min[doy]
features['expected_daily_range'] = seasonal_baseline_max[doy] - seasonal_baseline_min[doy]

# 3. Season-lag interaction
# For each seasonal lag (730, 1460, 2190, 8760):
features[f'seasonal_lag_{lag}_scaled'] = historical_ghi[-lag] * seasonal_amplitude_factor

# 4. Month interaction with short lags
for lag in [24, 48, 168]:
    features[f'GHI_lag_{lag}_x_month'] = historical_ghi[-lag] * features['month_sin']
    features[f'GHI_lag_{lag}_x_doy'] = historical_ghi[-lag] * features['doy_sin']
```

**Why this helps**: Helps the model learn how peak magnitudes should change throughout the year.

---

### SOLUTION 4: Time Series Decomposition Component ⭐⭐⭐⭐

**What it does**: Separately model trend, seasonal, and remainder components.

**Implementation**:
```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose 2021-2024 historical data
decomposition = seasonal_decompose(ghi_series, model='additive', period=8760)
trend_component = decomposition.trend
seasonal_component = decomposition.seasonal
residual_component = decomposition.resid

# For each forecast period:
# 1. Project trend forward (linear or LOESS)
forecast_trend = trend_projection[forecast_date]

# 2. Use seasonal pattern
forecast_seasonal = seasonal_component[same_doy]

# 3. GB model predicts residuals
forecast_residual = gb_model.predict(...)

# Combine:
forecast = forecast_trend + forecast_seasonal + forecast_residual
```

**Why this helps**: Classical time series approach specifically designed for this problem.

---

### SOLUTION 5: Strengthen Seasonal Features ⭐⭐⭐

**Current issue**: Cyclical encoding alone isn't powerful enough. Tree models need more explicit guidance.

**Enhancements**:
```python
# Replace simple month categorization with richer encoding
# Sun altitude correlation (affects irradiance)
features['sun_altitude_proxy'] = np.cos(2 * np.pi * (features['doy'] - 172) / 365)
# This approximates sun altitude: peaks around day 172 (June 21)

# Binary season indicators (currently just is_summer/is_winter)
features['is_spring'] = (features['month'].isin([3, 4, 5])).astype(float)
features['is_fall'] = (features['month'].isin([9, 10, 11])).astype(float)

# Season transition smoothing
doy_normalized = features['doy'] / 365
features['season_transition_smooth'] = np.sin(2 * np.pi * doy_normalized)**2
# Smoother encoding than hard boundaries

# Approximate solar declination (affects peak height)
# Declination varies from -23.5° to +23.5° throughout year
features['solar_declination_approx'] = 23.5 * np.sin(2 * np.pi * (features['doy'] - 81) / 365)
```

**Why this helps**: Physical basis for seasonal variation, better than abstract encoding.

---

### SOLUTION 6: Ensemble with SARIMA or Prophet ⭐⭐⭐

**What it does**: Create a weighted ensemble combining GB model with a dedicated seasonal model.

**Option A - Prophet** (simpler):
```python
from fbprophet import Prophet
import pandas as pd

# Train Prophet on historical data
df_prophet = pd.DataFrame({
    'ds': df.index,
    'y': df['GHI']
})
model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
model_prophet.fit(df_prophet)

# Make Prophet forecast
future = model_prophet.make_future_dataframe(periods=8760)
forecast_prophet = model_prophet.predict(future)

# Ensemble:
final_forecast = 0.6 * gb_forecast + 0.4 * prophet_forecast
```

**Option B - SARIMA** (more sophisticated):
```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# S=8760 (annual seasonality)
model_sarima = SARIMAX(ghi_series, order=(1,1,1), seasonal_order=(1,1,1,8760))
model_sarima.fit()
sarima_forecast = model_sarima.get_forecast(steps=8760)

# Ensemble
final_forecast = 0.7 * gb_forecast + 0.3 * sarima_forecast
```

**Why this helps**: Leverages specialized algorithms designed for seasonal time series.

---

## Implementation Priority

### Phase 1 (Immediate - Next Iteration)
1. **Add seasonal baseline features** (Solution 1)
2. **Implement hybrid ensemble** with seasonal decomposition (Solution 2)
3. **Add seasonal amplitude features** (Solution 3)

### Phase 2 (Next Review Cycle)
4. **Time series decomposition component** (Solution 4)
5. **Strengthen seasonal features** with physical basis (Solution 5)

### Phase 3 (Advanced - If Phase 1-2 insufficient)
6. **Ensemble with Prophet/SARIMA** for seasonal models (Solution 6)

---

## Expected Improvements

| Metric | Current | Expected After Phase 1+ |
|--------|---------|------------------------|
| MAPE (Seasonal accuracy) | ~8-13% | ~6-9% |
| Summer/Winter amplitude recognition | Poor | Good |
| Annual pattern smoothness | Too flat | Realistic variance |
| Day/night cycle quality | Excellent ✓ | Excellent ✓ (maintained) |

---

## Code Architecture

### Modified `build_future_features()` function:

```python
def build_future_features(target_date, history_ghi, seasonal_baseline, historical_hourly_pattern):
    features = {}
    
    # EXISTING FEATURES (keep all)
    # ... [short lags, rolling stats, cyclical encodings] ...
    
    # NEW: Seasonal components
    doy = target_date.dayofyear
    hour = target_date.hour
    
    # Baseline seasonal expectation
    features['seasonal_baseline_mean'] = seasonal_baseline['mean'].get(doy, np.mean(history_ghi))
    features['seasonal_baseline_amplitude'] = seasonal_baseline['amplitude'].get(doy, 0)
    features['expected_peak_today'] = seasonal_baseline['max'].get(doy, 0)
    
    # Amplitude scaling factor
    if seasonal_baseline['amplitude'].mean() > 0:
        features['amplitude_scaling'] = seasonal_baseline['amplitude'][doy] / seasonal_baseline['amplitude'].mean()
    else:
        features['amplitude_scaling'] = 1.0
    
    # Physical basis features
    features['solar_declination'] = 23.5 * np.sin(2 * np.pi * (doy - 81) / 365)
    features['sun_altitude_proxy'] = np.cos(2 * np.pi * (doy - 172) / 365)
    
    # Seasonal-lag interactions
    if 8760 < len(history_ghi):
        annual_lag_value = history_ghi[-8760]
        features['seasonal_lag_8760_scaled'] = annual_lag_value * features['amplitude_scaling']
    
    # Month-lag interactions
    for lag in [24, 48, 168]:
        if lag < len(history_ghi):
            features[f'GHI_lag_{lag}_x_amplitude'] = history_ghi[-lag] * features['amplitude_scaling']
    
    return features
```

### New model training approach:

```python
# Calculate seasonal baseline for this city
seasonal_baseline = city_ghi_series.groupby(city_ghi_series.index.dayofyear).agg({
    'mean': 'mean',
    'std': 'std',
    'min': 'min',
    'max': 'max',
    'amplitude': lambda x: x.max() - x.min()
})

# Option A: Train on residuals (Hybrid ensemble)
seasonal_fitted = pd.Series([seasonal_baseline.loc[idx.dayofyear, 'mean'] 
                             for idx in city_ghi_series.index], 
                            index=city_ghi_series.index)
residuals = city_ghi_series - seasonal_fitted

X_train = features_df.drop(columns=[TARGET_COLUMN])
y_train = residuals  # NOT raw GHI
model_gb.fit(X_train_scaled, y_train)

# Option B: Decompose and model separately
decomposition = seasonal_decompose(city_ghi_series, model='additive', period=8760)
y_train_residuals_only = decomposition.resid.dropna()
# Train GBModel on residuals only
```

---

## Testing & Validation

After implementing improvements:

1. **Visual comparison**:
   - Plot 2021-2024 historical (actual) GHI
   - Plot predicted 2025 vs. historical 2024 for same dates
   - Look for summer/winter amplitude differences

2. **Statistical validation**:
   - Calculate seasonal RMSE (separately for winter, spring, summer, fall)
   - Check monthly aggregates match historical patterns
   - Validate day-of-year average matches seasonal baseline

3. **Residual analysis**:
   - Residuals should show white noise patterns, not systematic bias
   - No autocorrelation in residuals

---

## References

- **Seasonal decomposition**: statsmodels STL decomposition
- **SARIMA**: Seasonal ARIMA for time series with seasonal patterns
- **Prophet**: Facebook's automated forecasting tool, designed for seasonality
- **Hybrid models**: Ensemble methods for time series (e.g., Shawe-Taylor et al.)

