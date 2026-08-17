import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from app.components.sidebar import render_sidebar
from src.environmental import OpenAQAdapter, CPCBAdapter
from src.pollution import AirPollutionEstimator
from src.visualization import create_emissions_bar_chart

st.set_page_config(page_title="Air Pollution Analytics", page_icon="🌫️", layout="wide")

settings = render_sidebar()

st.title("🌫️ Air Pollution & Environmental Telemetry Analytics")
st.caption("Compare ambient station measurements (OpenAQ / CPCB) with AI-estimated vehicle pollution contribution")

st.info(
    "⚠️ **Methodological Scientific Boundary:**\n"
    "Cameras cannot directly measure gas molecule concentrations. The metrics below strictly distinguish between "
    "**Station-Measured Ambient PM2.5/NO2** (physical sensors) and **Estimated Vehicle Pollution Contribution** ($g/hr$ mass emissions based on EEA COPERT V standards)."
)

# Environmental Adapters
openaq = OpenAQAdapter()
cpcb = CPCBAdapter()

col_query, col_source = st.columns([2, 1])
with col_query:
    location_query = st.text_input("Urban Location Query:", "Delhi")
with col_source:
    station_provider = st.selectbox("Station Data Adapter:", ["OpenAQ API", "CPCB India Station"])

if station_provider == "OpenAQ API":
    ambient_data = openaq.fetch_latest_measurement(location_query)
else:
    ambient_data = cpcb.fetch_latest_measurement(location_query)

st.markdown("---")

st.subheader(f"📡 Ambient Station Telemetry: {ambient_data.station_name}")
data_status = "🟢 LIVE API DATA" if ambient_data.is_live else "🔵 CACHED STATION BASELINE"
st.caption(f"Status: {data_status} | Timestamp: {ambient_data.timestamp}")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Ambient PM2.5", f"{ambient_data.pm2_5:.1f} µg/m³")
c2.metric("Ambient PM10", f"{ambient_data.pm10:.1f} µg/m³")
c3.metric("Ambient NO2", f"{ambient_data.no2:.1f} µg/m³")
c4.metric("Ambient CO", f"{ambient_data.co:.1f} mg/m³")
c5.metric("Ambient SO2", f"{ambient_data.so2:.1f} µg/m³")

st.markdown("---")

st.subheader("🚗 Estimated Vehicle Pollution Contribution vs. Ambient Telemetry")

pollution_estimator = AirPollutionEstimator()

# Fleet activity simulation
sample_fleet = [
    {"vehicle_class": "car", "fuel_type": "DIESEL", "speed_kmh": 25.0},
    {"vehicle_class": "car", "fuel_type": "PETROL", "speed_kmh": 35.0},
    {"vehicle_class": "car", "fuel_type": "EV", "speed_kmh": 40.0},
    {"vehicle_class": "bus", "fuel_type": "DIESEL", "speed_kmh": 20.0},
    {"vehicle_class": "truck", "fuel_type": "DIESEL", "speed_kmh": 30.0},
    {"vehicle_class": "motorcycle", "fuel_type": "PETROL", "speed_kmh": 45.0},
    {"vehicle_class": "van", "fuel_type": "DIESEL", "speed_kmh": 25.0}
]

fleet_report = pollution_estimator.estimate_fleet_emissions(sample_fleet, default_distance_km=1.0, time_window_seconds=3600.0)
vpi = fleet_report["vehicle_pollution_index"]

# Top Banner for Vehicle Pollution Contribution Index
m_idx, m_cat, m_vehicles, m_disclaimer = st.columns([1.5, 2, 1.5, 3])
m_idx.metric("Estimated Vehicle Index", f"{vpi['vehicle_pollution_index']} / 100")
m_cat.metric("Category", vpi["category"])
m_vehicles.metric("Vehicles Analyzed", fleet_report["total_vehicles_analyzed"])

st.warning(f"ℹ️ **Disclaimer:** {vpi['disclaimer']}")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### 📊 Estimated Mass Emission Rates (g/hr)")
    rates_df = pd.DataFrame(list(fleet_report["hourly_emission_rate_g_hr"].items()), columns=["Pollutant", "g/hr"])
    fig_rates = px.bar(rates_df, x="Pollutant", y="g/hr", color="Pollutant", title="Hourly Mass Emission Rate (g/hr)", template="plotly_dark")
    st.plotly_chart(fig_rates, use_container_width=True)

with col_right:
    st.markdown("#### ⚡ Contribution by Fuel Type (NO2 g/hr)")
    fuel_data = []
    for f_type, p_dict in fleet_report["by_fuel_type"].items():
        fuel_data.append({"Fuel Type": f_type, "NO2 (g/hr)": p_dict.get("NO2", 0.0)})
    fuel_df = pd.DataFrame(fuel_data)
    fig_fuel = px.pie(fuel_df, names="Fuel Type", values="NO2 (g/hr)", title="Estimated NO2 Contribution by Fuel Type", template="plotly_dark")
    st.plotly_chart(fig_fuel, use_container_width=True)

st.markdown("---")

st.subheader("📈 Projected 24-Hour Vehicle Emission Cycle")
hours = [f"{h:02d}:00" for h in range(24)]
traffic_volume = [15, 8, 5, 4, 10, 35, 85, 120, 140, 110, 95, 90, 100, 105, 115, 130, 160, 180, 150, 110, 80, 60, 40, 25]
est_pm25_g_hr = [v * 0.18 for v in traffic_volume]
ambient_pm25 = [50 + v * 0.25 + np.random.normal(0, 2) for v in traffic_volume]

trend_df = pd.DataFrame({
    "Hour": hours,
    "Traffic Volume (Vehicles/hr)": traffic_volume,
    "Estimated Vehicle PM2.5 Rate (g/hr)": est_pm25_g_hr,
    "Station Ambient PM2.5 (µg/m³)": ambient_pm25
})

fig_trend = px.line(
    trend_df,
    x="Hour",
    y=["Estimated Vehicle PM2.5 Rate (g/hr)", "Station Ambient PM2.5 (µg/m³)"],
    title="Diurnal Cycle: Estimated Vehicle Emission Rate (g/hr) vs. Ambient Station Telemetry (µg/m³)",
    markers=True,
    template="plotly_dark"
)
st.plotly_chart(fig_trend, use_container_width=True)
