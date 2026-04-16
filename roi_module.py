
import pandas as pd

df = pd.read_parquet("gradient_boosting_forecasts.parquet")

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
    system_size_kw = (panel_wattage * PANEL_SIZE[num_panels]) / 1000  # convert W to kW

    #Apply performance losses and Convert irradiance to kWh
    system_efficiency = 1 - performance_loss 
    
    # Sum hourly GHI to get annual total
    hourly_production = [max(0, ghi) * system_size_kw * system_efficiency for ghi in hourly_ghi]
    annual_production = sum(hourly_production) #first year production

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
        "Region": region,
        "Annual_Production": annual_production, # kWh produced in year 1
        "Payback_Years": payback_year, # the year when installation cost recovered
        "ROI": roi # lifetime ROI over 25 years (~183× installation cost)
    }

    return result


#test case
user_region = "Pacific Coast"
region_ghi = df[df['region'] == user_region]['forecasted_ghi'].values

result = solar_roi_analysis_region(
    region = user_region,
    hourly_ghi= region_ghi,
    panel_wattage=400,
    num_panels= 'Medium',
    performance_loss=0.2,
    degradation_rate=0.005,
    system_lifetime=25
)

print(result)
#{'Region': 'Pacific Coast', 'Annual_Production': 815737.4398754122, 'Payback_Years': 1, 'ROI': 183.60660545524235}
