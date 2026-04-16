import numpy as np
import pandas as pd
from roi_module import solar_roi_analysis_region

df = pd.read_parquet("tft_forecasts_final_cleaned.parquet")

REGIONAL_PRICES = {
    "Pacific Coast": 26.44570043 / 100,
    "South": 13.33696754 / 100,
    "Desert Southwest": 13.8071552 / 100,
    "Mountain": 13.34155623 / 100,
    "Northeast": 20.28842891 / 100,
    "Midwest": 14.06217236 / 100,
    "Pacific Northwest": 10.93149488 / 100
}

REGIONAL_INSTALL_COST = {
    "South": 35474,
    "Desert Southwest": 28766,
    "Mountain": 33858,
    "Northeast": 32602,
    "Pacific Coast": 27527,
    "Midwest": 37899,
    "Pacific Northwest": 36125
}

PANEL_SIZE = {
    "Small": 10,
    "Medium": 20,
    "Large": 30
}


# PVWATTS-STYLE BENCHMARK ENERGY (kWh/year per region)
# https://pvwatts.nlr.gov/pvwatts.php
pvwatts_kwh_5kw = {
    "Pacific Coast": 6739, #Los angeles
    "South": 5700, #Atlanta
    "Desert Southwest": 7060, #Phoenix
    "Mountain": 6391, #Denver
    "Northeast": 5240, #Boston
    "Midwest": 5227, #Chicago
    "Pacific Northwest": 4375 #Seattle
}

BASE_SYSTEM_KW = 5  # IMPORTANT FIX

pvwatts_kwh_per_kw = {
    k: v / BASE_SYSTEM_KW for k, v in pvwatts_kwh_5kw.items()
}


# BENCHMARK MODEL
def compute_benchmark_roi(region, system_kw=5, system_lifetime=25, degradation_rate=0.005):

    electricity_price = REGIONAL_PRICES[region]
    installation_cost = REGIONAL_INSTALL_COST[region]

    annual_production = pvwatts_kwh_per_kw[region] * system_kw

    total_savings = 0
    yearly_production = annual_production
    payback_year = None

    for year in range(1, system_lifetime + 1):

        if year > 1:
            yearly_production *= (1 - degradation_rate)

        yearly_savings = yearly_production * electricity_price
        total_savings += yearly_savings

        if payback_year is None and total_savings >= installation_cost:
            payback_year = year

    roi = (total_savings - installation_cost) / installation_cost

    return {
        "Benchmark_ROI": roi,
        "Benchmark_Payback": payback_year
    }


# SAFE MODEL EXTRACTION
def safe_model_output(model_result):
    roi = model_result.get("ROI", np.nan)
    payback = model_result.get("Payback_Years", np.nan)

    if roi is None or roi == -1:
        roi = np.nan

    if payback is None or payback == -1:
        payback = np.nan

    return roi, payback

# COMPARISON
def compare_model_vs_benchmark(model_result, benchmark_result, region):

    model_roi, model_payback = safe_model_output(model_result)

    benchmark_roi = benchmark_result["Benchmark_ROI"]
    benchmark_payback = benchmark_result["Benchmark_Payback"]

    abs_error = abs(model_roi - benchmark_roi)

    denom = max(abs(benchmark_roi), 1e-6)
    percent_error = (abs_error / denom) * 100

    return {
        "Region": region,
        "Model_ROI": model_roi,
        "Benchmark_ROI": benchmark_roi,
        "Absolute_Error": abs_error,
        "Percent_Error": percent_error,
        "Model_Payback": model_payback,
        "Benchmark_Payback": benchmark_payback
    }

# RUN EVALUATION
results = []

for region in pvwatts_kwh_5kw.keys():

    model_result = solar_roi_analysis_region(
        region=region,
        hourly_ghi=df[df["region"] == region]["ghi"].values,
        num_panels="Medium"
    )

    benchmark_result = compute_benchmark_roi(region, system_kw=5)

    comparison = compare_model_vs_benchmark(
        model_result,
        benchmark_result,
        region
    )

    results.append(comparison)

results_df = pd.DataFrame(results)


# CLEAN METRICS
clean_df = results_df.replace([np.inf, -np.inf], np.nan).dropna()

mae = clean_df["Absolute_Error"].mean()
mape = clean_df["Percent_Error"].mean()

print("\n=== ROI MODEL vs BENCHMARK ===\n")
print(clean_df)

print("\n=== SUMMARY METRICS ===")
print("MAE:", mae)
print("MAPE (%):", mape)

print(df.groupby("region")["actual_ghi"].count())