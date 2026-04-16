
import pandas as pd

df = pd.read_parquet("gradient_boost_final_cleaned.parquet")

# Regional electricity prices in $/kWh
REGIONAL_PRICES = {
    "Pacific Coast": 26.44570043 / 100,  # convert cents to $/kWh
    "South": 13.33696754 / 100,
    "Desert Southwest": 13.8071552 / 100,
    "Mountain": 13.34155623 / 100,
    "Northeast": 20.28842891 / 100,
    "Midwest": 14.06217236 / 100,
    "Pacific Northwest": 10.93149488 / 100
}

# Regional average installation cost
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

def solar_roi_analysis_region(
    region,
    hourly_ghi,
    panel_wattage=400, 
    num_panels= "Medium",
    performance_loss=0.2,
    degradation_rate=0.005, 
    system_lifetime=25 #The lifetime of solar panels typically ranges from 25 to 30 years
):

    electricity_price = REGIONAL_PRICES[region]
    installation_cost = REGIONAL_INSTALL_COST[region]
    num_panels_count = PANEL_SIZE[num_panels]

    #Apply performance losses and Convert irradiance to kWh
    system_efficiency = 1 - performance_loss 

    # System size (kW)
    system_size_kw = (panel_wattage * num_panels_count) / 1000
    
    ghi_series = pd.Series(hourly_ghi).fillna(0).clip(lower=0)
    total_ghi = sum(max(0, g) for g in hourly_ghi)

    # This is the only assumption in the system
    kwh_per_signal_unit = 0.001  # tune ONCE using historical data

    annual_production = (
        total_ghi
        * kwh_per_signal_unit
        * system_size_kw
        * system_efficiency
    )

    total_savings = 0
    payback_year = None
    yearly_production = annual_production #first year

    #Loop over each year
    for year in range(1, system_lifetime + 1):
        if year > 1:
            yearly_production *= (1 - degradation_rate) # apply yearly degradation

        #Calculate annual savings with local utility cost
        yearly_savings = yearly_production * electricity_price
        #Get cumulative yearly savings to get payback year
        total_savings += yearly_savings

        if payback_year is None and total_savings >= installation_cost:
            payback_year = year

    #roi is the life time (25 years) saving - installation cost
    roi = (total_savings - installation_cost) / installation_cost

    result = {
        "Annual_Production": annual_production, # kWh produced in year 1
        "Payback_Years": payback_year, # the year when installation cost recovered
        "ROI": roi # lifetime ROI over 25 years (~183× installation cost)
    }

    return result


#test case
all_results = []
regions = ["Pacific Coast", "South", "Desert Southwest", "Mountain", "Northeast", "Midwest", "Pacific Northwest"]


for region in regions:

    region_ghi = df[df["region"] == region]["actual_ghi"].dropna().values

    # skip empty regions (important safety)
    if len(region_ghi) == 0:
        print(f"Skipping {region} (no data)")
        continue

    result = solar_roi_analysis_region(
        region=region,
        hourly_ghi=region_ghi,
        panel_wattage=400,
        num_panels="Medium",
        performance_loss=0.2,
        degradation_rate=0.005,
        system_lifetime=25
    )

    result["Region"] = region
    all_results.append(result)

# Convert to DataFrame for clean viewing
results_df = pd.DataFrame(all_results)

print("\n=== ROI RESULTS (ALL REGIONS) ===\n")
print(results_df)

'''
results:

   Annual_Production  Payback_Years        ROI             Region
0         50375.8144              3  10.400369      Pacific Coast
1         43766.5024              7   2.876050              South
2         54339.7888              4   5.143889   Desert Southwest
3         45884.9472              6   3.259083           Mountain
4         36816.1920              5   4.396892          Northeast
5         36355.3984              8   2.177563            Midwest
6         34093.4912             10   1.430210  Pacific Northwest
'''
