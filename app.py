import streamlit as st
import sys, os

sys.path.insert(0, os.path.dirname(__file__))

from components.map import render_map
from components.roi_calculator import calculate_roi_data, plot_breakeven
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

    section[data-testid="stSidebar"] {
        background-color: #10101f !important;
        border-right: 1px solid #1e1e3a;
    }
    section[data-testid="stSidebar"] * {
        font-family: 'Share Tech Mono', monospace !important;
        color: #c0c0e8 !important;
    }

    .dashboard-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: -10px;
        letter-spacing: -0.5px;
    }
    .dashboard-subtitle {
        font-family: 'Share Tech Mono', monospace;
        color: #7c7cff;
        font-size: 0.9rem;
        margin-bottom: 20px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stat-card {
        background-color: #13132a;
        border: 1px solid #1e1e3a;
        border-radius: 8px;
        padding: 15px 20px;
        margin-bottom: 15px;
    }
    .stat-label {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.8rem;
        color: #8a8a9a;
        text-transform: uppercase;
    }
    .stat-value {
        font-family: 'Syne', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #e0e0ff;
        margin-top: 5px;
    }
    .plot-header {
        font-family: 'Share Tech Mono', monospace;
        font-size: 1.1rem;
        color: #ffffff;
        border-bottom: 1px solid #1e1e3a;
        padding-bottom: 10px;
        margin-top: 40px;
        margin-bottom: 20px;
    }
    .sidebar-header {
        font-size: 0.85rem;
        color: #8a8a9a !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: -10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="font-size: 1.5rem; font-weight: 800; color: #fff; font-family: \'Syne\', sans-serif; margin-bottom: 30px;">⚡ GRID_VIEW</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-header">Region Selector</div>', unsafe_allow_html=True)
    region_opts = [None] + list(REGION_STATES.keys())
    selected_region = st.selectbox(
        "Select a region to focus",
        options=region_opts,
        format_func=lambda x: "All Regions (USA)" if x is None else x,
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-header">System Inputs</div>', unsafe_allow_html=True)
    num_panels = st.slider("Number of Solar Panels (400W)", 5, 50, 20, step=1)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-header">Financial Inputs</div>', unsafe_allow_html=True)
    monthly_bill = st.slider("Monthly Electric Bill ($)", 50, 500, 150, step=10)
    system_cost = st.number_input("Estimated System Cost ($)", min_value=3000, value=(num_panels*800), step=500)

# ── Main Content (Always Shows Map) ───────────────────────────────────────────
st.markdown('<div class="dashboard-title">US Regional Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-subtitle">Real-time macro analysis console</div>', unsafe_allow_html=True)

col_map, col_stats = st.columns([3, 1], gap="large")

with col_map:
    fig_map = render_map(selected_region)
    st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})

with col_stats:
    st.markdown("<br><br>", unsafe_allow_html=True)
    region_display = selected_region if selected_region else "All Regions"
    accent_hex = REGION_COLORS.get(selected_region, "#7c7cff") if selected_region else "#e0e0ff"

    st.markdown(
        f'<div class="stat-card" style="border-left: 4px solid {accent_hex}">'
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

# ── Plots section (Only shows if region is selected) ─────────────────────────
st.markdown(
    '<div class="plot-header">▸ ROI & Financial Analysis</div>',
    unsafe_allow_html=True,
)

if selected_region:
    # 1. Run the ML-integrated math
    roi_data = calculate_roi_data(monthly_bill, system_cost, num_panels, selected_region)
    
    # 2. Show the Breakeven Chart
    st.plotly_chart(
        plot_breakeven(roi_data, selected_region),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    
    # 3. Show the metrics
    m1, m2, m3 = st.columns(3)
    net_savings = roi_data['Utility'].iloc[-1] - roi_data['Solar'].iloc[-1]
    
    with m1:
        st.metric("25-Year Utility Cost", f"${int(roi_data['Utility'].iloc[-1]):,}")
    with m2:
        st.metric("25-Year Solar Cost", f"${int(roi_data['Solar'].iloc[-1]):,}")
    with m3:
        st.metric("Estimated Net Savings", f"${int(net_savings):,}")
else:
    # What the user sees before selecting a region (exactly like your teammate had it)
    st.markdown(
        '<div style="color:#8a8a9a; font-family:\'Share Tech Mono\', monospace; text-align:center; padding: 40px; border: 1px dashed #1e1e3a; border-radius: 8px;">'
        'Select a region from the sidebar to view detailed analytics.'
        '</div>',
        unsafe_allow_html=True
    )