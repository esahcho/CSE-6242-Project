from data.region_states import CITY_MAP, REGION_COLORS
import pandas as pd
import plotly.graph_objects as go

CARD_BG = "#13132a"
GRID_COLOR = "#1e1e3a"
TEXT_COLOR = "#e0e0ff"


def plot_forecast_data(region):
    df = pd.read_parquet("forecasts/tft_forecasts_by_city.parquet")
    city = CITY_MAP[region]
    df = df[(df['data_type'] == "forecast") & (df['city'] == city)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['ghi'], name="Forecasted Irradiance", line=dict(color="#2ecc71", width=3)))
    
    fig.update_layout(
        title=dict(text=f"Annual Forecasted Irradiance [{region}]", font=dict(color=TEXT_COLOR, size=14, family="monospace")),
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG, font=dict(color=TEXT_COLOR, family="monospace"),
        margin=dict(l=40, r=20, t=50, b=40), height=350,
        xaxis=dict(title="Date", gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, showgrid=True),
        yaxis=dict(title="Global Horizon Irradiance", gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, showgrid=True),
        hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(
        dtick="M1",
        tickformat="%b",
        ticklabelmode="period"
    )
    
    return fig