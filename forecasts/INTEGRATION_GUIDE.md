# Quick Integration Guide: Adding Seasonal Components to Your Notebook

## 3-Step Quick Implementation

### STEP 1: Import the Improved Forecaster
Add this to your notebook after the imports (Section 1):

```python
# Add to imports section
from improved_forecaster import SeasonalGradientBoostingForecaster

# If you can't use the module, you'll need to copy the SeasonalGradientBoostingForecaster 
# class definition into your notebook cell (or import it as shown above)
```

---

### STEP 2: Modify Feature Engineering (Minimal Change)

In the loop where you process each city, BEFORE training the model, add this:

```python
# EXISTING CODE - keep as is
features_df = features_df.dropna()
features_df = features_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

# NEW: Use improved forecaster with seasonal components
forecaster = SeasonalGradientBoostingForecaster(n_components=45)
forecaster.fit(city_ghi_series, features_df)

# Store for later use in recursive forecasting
forecasters_by_city[city] = forecaster

# Calculate training metrics on the GB model predictions (for compatibility)
X_train = features_df.drop(columns=[TARGET_COLUMN])
y_train = features_df[TARGET_COLUMN]
n_components = min(45, X_train.shape[1])
pca = PCA(n_components=n_components, random_state=42)
X_train_pca = pca.fit_transform(X_train)
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train_pca)
y_train_pred = forecaster.model_gb.predict(X_train_scaled)

# Get metrics (same as before for compatibility)
train_mae = mean_absolute_error(y_train.values, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train.values, y_train_pred))
train_mape = mean_absolute_percentage_error(
    y_train.values[y_train.values > 10],
    y_train_pred[y_train.values > 10] + 1e-10
) * 100
```

---

### STEP 3: Modify Recursive Forecasting

Replace the existing recursive forecast section with:

```python
# Store forecasters for each city (initialize once at the start of feature engineering loop)
forecasters_by_city = {}

# In the loop for each city (after features_df is ready):
history_ghi = city_ghi_series.tolist()
forecast_count = 0

for ts in future_index:
    # IMPROVED: Use forecaster.predict() which includes seasonal component
    pred_ghi = forecaster.predict(
        ts, history_ghi,
        X_train.columns,  # Feature column names
        short_lags=[1, 2, 3, 6, 12, 24, 48, 72, 168],
        seasonal_lags=[730, 1460, 2190, 8760],
        rolling_windows=[6, 24, 48, 168, 336, 732]
    )
    
    history_ghi.append(pred_ghi)
    forecast_count += 1
    
    all_results.append({
        'city': city if city else 'Unknown',
        'date': ts,
        'actual_ghi': np.nan,
        'forecasted_ghi': pred_ghi,  # ← Now includes seasonal awareness!
        'mae': train_mae,
        'rmse': train_rmse,
        'mape': train_mape,
        'data_type': 'forecast'
    })
```

---

## Key Differences (Why This Improves Things)

### What's Different:
1. **Seasonal Baseline** calculated from your 3-year historical data
2. **Enhanced Features** that encode sun altitude and solar position
3. **GB Model trained on residuals**, not raw GHI
   - The model learns: "What's unusual about this prediction?"
   - Not: "Guess the absolute value"
4. **Hybrid prediction** combines seasonal expectation + anomaly prediction

### Example:
```
BEFORE (Current Model):
- Winter day: Model predicts ~400 W/m² 
- Summer day: Model predicts ~350 W/m² (too similar!)

AFTER (Improved Model):
- Winter day: Seasonal baseline ~250 W/m² + Residual +150 = 400 W/m² ✓
- Summer day: Seasonal baseline ~850 W/m² + Residual -500 = 350 W/m² ✓
  (Actually captures the magnitude difference)
```

---

## Implementation Checklist

- [ ] Copy `improved_forecaster.py` to your project directory
- [ ] Add import at top of notebook
- [ ] Initialize `forecasters_by_city = {}` before the city loop
- [ ] Replace model training section (Step 2 above)
- [ ] Replace recursive forecast section (Step 3 above)
- [ ] Run notebook and compare forecast charts
- [ ] Verify summer/winter amplitude differences appear in plots

---

## Expected Changes in Output

**Before**: Flat forecasts throughout the year
```
Jan: 600 W/m² (avg)
Jun: 580 W/m² (avg)  ← TOO SIMILAR!
```

**After**: Clear seasonal variation
```
Jan: 400 W/m² (avg)  ← Winter lower
Jun: 850 W/m² (avg)  ← Summer higher ✓
```

---

## If You Still Want to Use Your Original Model

You can keep the original code and just try this as an alternative. Both `improved_forecaster.py` and your current model can coexist - just compare results.

---

## Advanced: Fine-Tuning Options

If results still need improvement, you can adjust:

```python
# In SeasonalGradientBoostingForecaster.fit():
# These hyperparameters control seasonal weight

1. GBModel parameters:
   - n_estimators=200  (try 300-400 for more detail)
   - max_depth=6       (try 7-8 if overfitting not issue)
   - learning_rate=0.12 (lower = more gradual learning)

2. Feature engineering:
   - Add wind speed / cloud cover if available
   - Adjust seasonal lag windows to match your latitude

3. Consider ensemble weights:
   seasonal_portion = 0.7  # How much to trust seasonal baseline
   residual_portion = 0.3  # How much to trust GB model
   forecast = seasonal_portion * baseline + residual_portion * gb_residual
```

---

## Troubleshooting

**Issue**: Forecasts look worse than before
- **Solution**: The seasonal baseline needs good historical data. Check that your 2021-2024 data is complete and clean.

**Issue**: Forecasts are too flat again
- **Solution**: The GB model is overriding the seasonal signal. Reduce n_estimators or increase learning_rate in the forecaster.

**Issue**: Extreme forecasts (very high/low values)
- **Solution**: Increase regularization in GBModel parameters, or add clipping:
  ```python
  forecast_final = np.clip(forecast_combined, 0, max_ghi_in_training)
  ```

**Issue**: Memory error during fitting
- **Solution**: Reduce n_components or use fewer seasonal lags

---

## Validation Script

After implementing, run this to compare old vs. new forecasts:

```python
# Compare seasonal patterns
import matplotlib.pyplot as plt

# For one city (e.g., Atlanta)
atlanta_forecast = results_df[results_df['city'] == 'Atlanta'].copy()
atlanta_forecast['date'] = pd.to_datetime(atlanta_forecast['date'])
atlanta_forecast = atlanta_forecast.sort_values('date')

# Get month averages
atlanta_forecast['month'] = atlanta_forecast['date'].dt.month
monthly_avg = atlanta_forecast[atlanta_forecast['actual_ghi'].isna()].groupby('month')['forecasted_ghi'].mean()

# Plot
plt.figure(figsize=(12, 6))
plt.plot(monthly_avg.index, monthly_avg.values, 'o-', linewidth=2, markersize=8)
plt.xlabel('Month')
plt.ylabel('Average GHI (W/m²)')
plt.title('Atlanta 2025 Forecast - Monthly Average')
plt.grid(True, alpha=0.3)
plt.xticks(range(1, 13))
plt.show()

# Compare with 2024 historical data
atlanta_hist_2024 = df[df['city'] == 'Atlanta'].loc['2024'].groupby(df.index.month)['GHI'].mean()
plt.figure(figsize=(12, 6))
plt.plot(atlanta_hist_2024.index, atlanta_hist_2024.values, label='2024 Historical', linewidth=2)
plt.plot(monthly_avg.index, monthly_avg.values, label='2025 Forecast', linewidth=2)
plt.xlabel('Month')
plt.ylabel('Average GHI (W/m²)')
plt.title('Atlanta - Comparing 2024 Historical vs 2025 Forecast')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(range(1, 13))
plt.show()
```

The forecast should now show similar seasonal patterns to the historical data!
