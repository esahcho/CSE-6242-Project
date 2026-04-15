import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
from data.region_states import REGION_COLORS

CARD_BG = "#13132a"
GRID_COLOR = "#1e1e3a"
TEXT_COLOR = "#e0e0ff"

CITY_MAP = {
    "Pacific Coast": "Los Angeles",
    "South": "Atlanta",
    "Desert": "Phoenix",
    "Southwest": "Phoenix",
    "Mountain": "Denver",
    "Northeast": "Boston",
    "Midwest": "Chicago",
    "Pacific Northwest": "Seattle"
}

def calculate_roi_data(monthly_bill, system_cost, num_panels, region, inflation_rate=0.03):
    city = CITY_MAP.get(region, "Chicago") 
    
    # Parquet data
    try:
        df = pd.read_parquet("forecasts/gradient_boosting_forecasts_by_city.parquet")

        city_data = df[df['city'] == city]
        
        city_data['date'] = pd.to_datetime(city_data['date'])
        forecast_2025 = city_data[city_data['date'].dt.year == 2025]
        
        annual_ghi_estimate = forecast_2025['forecasted_ghi'].sum() 

        # # SUCCESS
        # st.success(f"Successfully loaded ML Data for {city}") 
        
    except Exception as e:
        # ERROR
        # st.error(f"ML Data Error for {city}: {e}") 
        annual_ghi_estimate = 2000000 # if file isn't found

    # depreciation constants
    panel_wattage_kw = 400 / 1000  
    performance_ratio = 0.8
    degradation_rate = 0.005 
    
    # Calculate Energy Production and Utility Rate
    system_capacity_kw = num_panels * panel_wattage_kw
    base_annual_production_kwh = (annual_ghi_estimate / 1000) * performance_ratio * system_capacity_kw

    # st.info(f"DEBUG: Raw GHI = {annual_ghi_estimate:,.0f} | Estimated Production = {base_annual_production_kwh:,.0f} kWh/year")
    
    assumed_monthly_kwh = 900
    local_utility_rate = monthly_bill / assumed_monthly_kwh
    
    years = np.arange(1, 26)
    annual_utility = (monthly_bill * 12) * ((1 + inflation_rate) ** years)
    cumulative_utility = np.cumsum(annual_utility)
    
    cumulative_solar = []
    current_cost = system_cost
    
    for year in years:
        degraded_production = base_annual_production_kwh * ((1 - degradation_rate) ** year)
        annual_usage = assumed_monthly_kwh * 12
        shortfall_kwh = max(0, annual_usage - degraded_production)
        current_utility_rate = local_utility_rate * ((1 + inflation_rate) ** year)
        annual_solar_bill = shortfall_kwh * current_utility_rate
        
        current_cost += annual_solar_bill
        cumulative_solar.append(current_cost)
        
    return pd.DataFrame({
        'Year': years,
        'Utility': cumulative_utility,
        'Solar': cumulative_solar
    })

def plot_breakeven(roi_df, region):
    accent = REGION_COLORS.get(region, "#7c7cff")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=roi_df['Year'], y=roi_df['Utility'], name="Utility Cost", line=dict(color="#E63946", width=3)))
    fig.add_trace(go.Scatter(
        x=roi_df['Year'], y=roi_df['Solar'], 
        name="Solar Investment", 
        line=dict(color="#2ecc71", width=3) # <--- Put any color hex code you want here
    ))
    
    fig.update_layout(
        title=dict(text=f"25-Year ROI Analysis [{region}]", font=dict(color=TEXT_COLOR, size=14, family="monospace")),
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG, font=dict(color=TEXT_COLOR, family="monospace"),
        margin=dict(l=40, r=20, t=50, b=40), height=350,
        xaxis=dict(title="Years", gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, showgrid=True),
        yaxis=dict(title="Cumulative Cost ($)", gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, showgrid=True),
        hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig