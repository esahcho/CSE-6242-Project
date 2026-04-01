import plotly.graph_objects as go
import pandas as pd
import numpy as np
from data.region_states import REGION_COLORS

# Borrowing your teammate's theme colors
CARD_BG = "#13132a"
GRID_COLOR = "#1e1e3a"
TEXT_COLOR = "#e0e0ff"

def calculate_roi_data(monthly_bill, system_cost, inflation_rate=0.03):
    """Generates the 25-year cumulative cost data."""
    years = np.arange(1, 26)
    
    annual_utility = (monthly_bill * 12) * ((1 + inflation_rate) ** years)
    cumulative_utility = np.cumsum(annual_utility)
    cumulative_solar = np.full(25, system_cost) # Flat line for upfront cost
    
    return pd.DataFrame({
        'Year': years,
        'Utility': cumulative_utility,
        'Solar': cumulative_solar
    })

def plot_breakeven(roi_df, region):
    """Generates the Breakeven Plotly chart using the teammate's styling."""
    accent = REGION_COLORS.get(region, "#7c7cff")
    
    fig = go.Figure()
    
    # Utility Line
    fig.add_trace(go.Scatter(
        x=roi_df['Year'], y=roi_df['Utility'], 
        name="Utility Cost", 
        line=dict(color="#E63946", width=3) 
    ))
    
    # Solar Line
    fig.add_trace(go.Scatter(
        x=roi_df['Year'], y=roi_df['Solar'], 
        name="Solar Investment", 
        line=dict(color=accent, width=3) 
    ))
    
    fig.update_layout(
        title=dict(text=f"25-Year ROI Analysis [{region}]", font=dict(color=TEXT_COLOR, size=14, family="monospace")),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_COLOR, family="monospace"),
        margin=dict(l=40, r=20, t=50, b=40),
        height=350,
        xaxis=dict(title="Years", gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, showgrid=True),
        yaxis=dict(title="Cumulative Cost ($)", gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, showgrid=True),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig