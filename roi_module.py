
import pandas as pd

df = pd.read_parquet("tft_forecasts_final_cleaned.parquet")

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

#--------------------------------
#ROI calculation function
#-------------------------------
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

    total_ghi = sum(max(0, g) for g in hourly_ghi)

    kwh_per_signal_unit = 1 / 1000 #GHI is in W/m² → convert to kW/m²

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


#---------------
#test case
#--------------
all_results = []
regions = ["Pacific Coast", "South", "Desert Southwest", "Mountain", "Northeast", "Midwest", "Pacific Northwest"]


for region in regions:

    region_ghi = df[df["region"] == region]["ghi"].dropna().values

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
        system_lifetime=35
    )

    result["Region"] = region
    all_results.append(result)

# Convert to DataFrame for clean viewing
results_df = pd.DataFrame(all_results)

print("\n=== ROI RESULTS (ALL REGIONS) ===\n")
print(results_df)

'''
=== ROI RESULTS (ALL REGIONS) ===

   Annual_Production  Payback_Years       ROI             Region
0       14799.580714              8  4.166333      Pacific Coast
1       13285.876268             22  0.814989              South
2       15860.074430             14  1.766093   Desert Southwest
3       13903.602963             20  0.990716           Mountain
4       11512.899821             15  1.603309          Northeast
5       11392.519403             26  0.535965            Midwest
6       10861.067435             33  0.194211  Pacific Northwest
'''
