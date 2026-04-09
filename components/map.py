import plotly.graph_objects as go
import pandas as pd
from data.region_states import REGION_STATES, REGION_COLORS


def render_map(selected_region: str | None) -> go.Figure:
    """Render a US choropleth map highlighting the selected region."""

    rows = []
    for region, states in REGION_STATES.items():
        for state in states:
            rows.append({"state": state, "region": region})

    df = pd.DataFrame(rows).drop_duplicates(subset="state")

    # Assign color values: highlighted vs muted
    highlight_color = REGION_COLORS.get(selected_region, "#cccccc")

    def get_color(region):
        if selected_region is None:
            return REGION_COLORS.get(region, "#cccccc")
        return REGION_COLORS[region] if region == selected_region else "#2b2d42"

    df["color"] = df["region"].apply(get_color)
    df["opacity"] = df["region"].apply(lambda r: 1.0 if (selected_region is None or r == selected_region) else 0.25)

    fig = go.Figure(
        go.Choropleth(
            locations=df["state"],
            z=df["region"].apply(lambda r: list(REGION_STATES.keys()).index(r)),
            locationmode="USA-states",
            colorscale=[
                [i / (len(REGION_STATES) - 1), REGION_COLORS[r]]
                for i, r in enumerate(REGION_STATES.keys())
            ],
            showscale=False,
            marker_line_color="#0d0d0d",
            marker_line_width=1.2,
            text=df["region"],
            hovertemplate="<b>%{location}</b><br>Region: %{text}<extra></extra>",
        )
    )

    # If a region is selected, overlay muted states with a separate trace
    if selected_region:
        muted_df = df[df["region"] != selected_region]
        highlight_df = df[df["region"] == selected_region]

        fig = go.Figure()

        # Muted states
        fig.add_trace(
            go.Choropleth(
                locations=muted_df["state"],
                z=[0.1] * len(muted_df),
                locationmode="USA-states",
                colorscale=[[0, "#1e1e2e"], [1, "#1e1e2e"]],
                showscale=False,
                marker_line_color="#333355",
                marker_line_width=1,
                hoverinfo="skip",
            )
        )

        # Highlighted states
        fig.add_trace(
            go.Choropleth(
                locations=highlight_df["state"],
                z=[1.0] * len(highlight_df),
                locationmode="USA-states",
                colorscale=[[0, highlight_color], [1, highlight_color]],
                showscale=False,
                marker_line_color="#ffffff",
                marker_line_width=1.8,
                text=highlight_df["region"],
                hovertemplate="<b>%{location}</b><br>%{text}<extra></extra>",
            )
        )
    else:
        # All regions colored
        fig = go.Figure()
        for region, states in REGION_STATES.items():
            region_df = df[df["region"] == region]
            fig.add_trace(
                go.Choropleth(
                    locations=region_df["state"],
                    z=[list(REGION_STATES.keys()).index(region)] * len(region_df),
                    locationmode="USA-states",
                    colorscale=[
                        [0, REGION_COLORS[region]],
                        [1, REGION_COLORS[region]],
                    ],
                    showscale=False,
                    marker_line_color="#0d0d0d",
                    marker_line_width=1.2,
                    text=[region] * len(region_df),
                    hovertemplate="<b>%{location}</b><br>%{text}<extra></extra>",
                )
            )

    fig.update_layout(
        geo=dict(
            scope="usa",
            bgcolor="rgba(0,0,0,0)",
            lakecolor="#0d0d1a",
            landcolor="#1e1e2e",
            subunitcolor="#333355",
            showlakes=True,
            showframe=False,
            projection_type="albers usa",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        height=420,
    )

    return fig
