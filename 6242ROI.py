
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

def solar_roi_analysis_region(
    city,
    state,
    region,
    hourly_ghi,
    panel_wattage=400, 
    num_panels=20,
    performance_loss=0.2,
    degradation_rate=0.005, 
    system_lifetime=25 #The lifetime of solar panels typically ranges from 25 to 30 years
):

    electricity_price = REGIONAL_PRICES[region]
    installation_cost = REGIONAL_INSTALL_COST[region]
    system_size_kw = (panel_wattage * num_panels) / 1000  # convert W to kW

    #Apply performance losses and Convert irradiance to kWh
    system_efficiency = 1 - performance_loss 
    
    # Sum hourly GHI to get annual total
    hourly_production = [max(0, ghi) * system_size_kw * system_efficiency for ghi in hourly_ghi]
    annual_production = sum(hourly_production)

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

    #Subtract installation cost
    roi = (total_savings - installation_cost) / installation_cost

    result = {
        "City": city,
        "State": state,
        "Annual_Production": annual_production,
        "Payback_Years": payback_year,
        "ROI": roi
    }

    return result

# Example usage
hourly_ghi_example = [
    1.68, 6.30, 3.27, 5.33, -0.90, 1.92, 36.51, 126.67, 288.68, 382.44,
    461.40, 511.84, 563.56, 522.25, 585.75, 364.38, 153.43, 6.26, 11.42, -8.78,
    18.52, 5.09, -3.26, 1.91
]  # example: first 24 hours

result = solar_roi_analysis_region(
    city="San Diego",
    state="CA",
    region="Pacific Coast",
    hourly_ghi=hourly_ghi_example,
    panel_wattage=400,
    num_panels=20,
    performance_loss=0.2,
    degradation_rate=0.005,
    system_lifetime=25
)

print(result)
#{'City': 'San Diego', 'State': 'CA', 'Annual_Production': 25892.288, 'Payback_Years': 5, 'ROI': 4.859590551439614}
