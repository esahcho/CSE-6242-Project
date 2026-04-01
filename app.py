import streamlit as st
import sys, os

sys.path.insert(0, os.path.dirname(__file__))

from components.map import render_map
from components.plots import plot_one, plot_two, plot_three
from data.region_states import REGION_STATES, REGION_COLORS

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="US Regional Dashboard",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Syne:wght@400;700;800&display=swap');

    html, body, [class*="css"] {
        background-color: #0d0d1a;
        color: #e0e0ff;
        font-family: 'Syne', sans-serif;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #10101f !important;
        border-right: 1px solid #1e1e3a;
    }
    section[data-testid="stSidebar"] * {
        font-family: 'Share Tech Mono', monospace !important;
        color: #c0c0e8 !important;
    }

    /* Title */
    .dashboard-title {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 2.2rem;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #7c7cff 0%, #c084fc 60%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .dashboard-sub {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.78rem;
        color: #4a4a7a;
        margin-top: 2px;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    /* Region badge */
    .region-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 4px;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: 1px;
        margin-top: 8px;
    }

    /* Stat cards */
    .stat-card {
        background: #13132a;
        border: 1px solid #1e1e3a;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 10px;
        font-family: 'Share Tech Mono', monospace;
    }
    .stat-label {
        font-size: 0.7rem;
        color: #4a4a7a;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .stat-value {
        font-size: 1.6rem;
        color: #e0e0ff;
        margin-top: 4px;
    }

    /* Divider */
    hr { border-color: #1e1e3a; }

    /* Plot section header */
    .plot-header {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.7rem;
        color: #4a4a7a;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin: 24px 0 12px 0;
        border-top: 1px solid #1e1e3a;
        padding-top: 20px;
    }

    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 60px 20px;
        font-family: 'Share Tech Mono', monospace;
        color: #2a2a4a;
        font-size: 0.9rem;
        letter-spacing: 2px;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer, header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🗺️ Controls")
    st.markdown("---")

    int_val = st.slider(
        "Number of Solar Panels  (n)",
        min_value=1,
        max_value=30,
        value=5,
        step=1,
        help="Number of Solar Panels to calculate ROI with",
    )

    st.markdown("")

    region_options = list(REGION_STATES.keys())
    selected_region = st.selectbox(
        "Region",
        options=["— select a region —"] + region_options,
        index=0,
    )
    if selected_region == "— select a region —":
        selected_region = None

    st.markdown("---")

    if selected_region:
        accent = REGION_COLORS[selected_region]
        st.markdown(
            f'<div class="region-badge" style="background:{accent}22; '
            f'border:1px solid {accent}; color:{accent};">'
            f"▶ {selected_region}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**{len(REGION_STATES[selected_region])} states** included",
        )
        st.markdown(" · ".join(REGION_STATES[selected_region]))
    else:
        st.caption("Select a region to highlight it on the map and generate plots.")

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown(
    '<p class="dashboard-title">US Solar Panel ROI</p>',
    unsafe_allow_html=True,
)
st.markdown("<hr>", unsafe_allow_html=True)

# Top row: map + info cards
map_col, info_col = st.columns([3, 1], gap="large")

with map_col:
    fig_map = render_map(selected_region)
    st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})

with info_col:
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    st.markdown(
        f'<div class="stat-card">'
        f'<div class="stat-label">Sample Size</div>'
        f'<div class="stat-value">{int_val:,}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    region_display = selected_region if selected_region else "—"
    accent_hex = REGION_COLORS.get(selected_region, "#4a4a7a") if selected_region else "#4a4a7a"
    st.markdown(
        f'<div class="stat-card" style="border-color:{accent_hex}55">'
        f'<div class="stat-label">Selected Region</div>'
        f'<div class="stat-value" style="font-size:1.1rem; color:{accent_hex}; margin-top:6px">'
        f"{region_display}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if selected_region:
        states = REGION_STATES[selected_region]
        st.markdown(
            f'<div class="stat-card">'
            f'<div class="stat-label">States</div>'
            f'<div class="stat-value">{len(states)}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

# ── Plots section ─────────────────────────────────────────────────────────────
st.markdown(
    '<div class="plot-header">▸ Analysis Plots</div>',
    unsafe_allow_html=True,
)

if selected_region:
    p1, p2, p3 = st.columns(3, gap="medium")
    with p1:
        st.plotly_chart(
            plot_one(int_val, selected_region),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with p2:
        st.plotly_chart(
            plot_two(int_val, selected_region),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with p3:
        st.plotly_chart(
            plot_three(int_val, selected_region),
            use_container_width=True,
            config={"displayModeBar": False},
        )
else:
    st.markdown(
        '<div class="empty-state">'
        "[ select a region from the sidebar to generate plots ]"
        "</div>",
        unsafe_allow_html=True,
    )
