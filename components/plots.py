import plotly.graph_objects as go
import numpy as np
from data.region_states import REGION_COLORS

DARK_BG = "#0d0d1a"
CARD_BG = "#13132a"
GRID_COLOR = "#1e1e3a"
TEXT_COLOR = "#e0e0ff"


def _base_layout(title: str, region: str) -> dict:
    accent = REGION_COLORS.get(region, "#7c7cff")
    return dict(
        title=dict(text=title, font=dict(color=TEXT_COLOR, size=13, family="monospace"), x=0.04),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_COLOR, family="monospace"),
        margin=dict(l=40, r=20, t=50, b=40),
        height=280,
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, showgrid=True),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, showgrid=True),
        _accent=accent,  # used below, stripped before passing
    )


def plot_one(int_val: int, region: str) -> go.Figure:
    """
    PLACEHOLDER — Bar chart.
    Replace with real data logic tied to int_val and region.
    """
    rng = np.random.default_rng(int_val + hash(region) % 999)
    accent = REGION_COLORS.get(region, "#7c7cff")
    categories = ["A", "B", "C", "D", "E", "F"]
    values = rng.integers(10, 100, size=len(categories))

    fig = go.Figure(
        go.Bar(
            x=categories,
            y=values,
            marker=dict(
                color=values,
                colorscale=[[0, "#1e1e3a"], [1, accent]],
                line=dict(color=accent, width=0.8),
            ),
            hovertemplate="Category %{x}: %{y}<extra></extra>",
        )
    )

    layout = _base_layout(f"📊 Plot 1 — Bar  [{region}  ·  n={int_val}]", region)
    layout.pop("_accent", None)
    fig.update_layout(**layout)
    return fig


def plot_two(int_val: int, region: str) -> go.Figure:
    """
    PLACEHOLDER — Line chart.
    Replace with real data logic tied to int_val and region.
    """
    rng = np.random.default_rng(int_val * 7 + hash(region) % 777)
    accent = REGION_COLORS.get(region, "#7c7cff")
    x = np.arange(int_val if int_val > 1 else 20)
    y = np.cumsum(rng.normal(0, 1, size=len(x)))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color=accent, width=2),
            fill="tozeroy",
            fillcolor=f"rgba({int(accent[1:3],16)},{int(accent[3:5],16)},{int(accent[5:7],16)},0.12)",
            hovertemplate="t=%{x}  val=%{y:.2f}<extra></extra>",
        )
    )

    layout = _base_layout(f"📈 Plot 2 — Line  [{region}  ·  n={int_val}]", region)
    layout.pop("_accent", None)
    fig.update_layout(**layout)
    return fig


def plot_three(int_val: int, region: str) -> go.Figure:
    """
    PLACEHOLDER — Scatter plot.
    Replace with real data logic tied to int_val and region.
    """
    rng = np.random.default_rng(int_val * 13 + hash(region) % 555)
    accent = REGION_COLORS.get(region, "#7c7cff")
    n = max(int_val, 15)
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)
    sizes = rng.integers(6, 18, n)

    fig = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                size=sizes,
                color=y,
                colorscale=[[0, "#1e1e3a"], [1, accent]],
                opacity=0.85,
                line=dict(color=accent, width=0.5),
            ),
            hovertemplate="(%{x:.2f}, %{y:.2f})<extra></extra>",
        )
    )

    layout = _base_layout(f"🔵 Plot 3 — Scatter  [{region}  ·  n={int_val}]", region)
    layout.pop("_accent", None)
    fig.update_layout(**layout)
    return fig
