import streamlit as st

def render_metrics_cards(
    vehicle_count: int,
    pm25_g_hr: float,
    nox_g_hr: float,
    co2_g_hr: float,
    noise_dba: float
):
    """Renders 5 key metric summary cards."""
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Vehicles", f"{vehicle_count}", delta="Active Count")
    with col2:
        st.metric("Est. PM2.5", f"{pm25_g_hr:.2f} g/hr", delta="Vehicle Share")
    with col3:
        st.metric("Est. NOx", f"{nox_g_hr:.2f} g/hr", delta="Vehicle Share")
    with col4:
        st.metric("Est. CO2 Rate", f"{co2_g_hr:.1f} g/hr", delta="Carbon Rate")
    with col5:
        st.metric("Relative Noise", f"{noise_dba:.1f} dBA", delta="Traffic Noise")
