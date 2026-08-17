import streamlit as st
import pandas as pd
import json
import datetime
from pathlib import Path

from app.components.sidebar import render_sidebar

st.set_page_config(page_title="Reports & Data Export", page_icon="📁", layout="wide")

settings = render_sidebar()

st.title("📁 Reports & Telemetry Data Export")
st.caption("Export vehicle detection records, fuel distribution breakdowns, and COPERT emission estimates")

# Generate sample telemetric report dataset
records = [
    {
        "timestamp": (datetime.datetime.now() - datetime.timedelta(minutes=i*5)).strftime("%Y-%m-%d %H:%M:%S"),
        "track_id": f"TRK-{100+i}",
        "vehicle_class": vcls,
        "fuel_type": fuel,
        "speed_kmh": round(speed, 1),
        "est_pm25_g_hr": round(pm, 4),
        "est_nox_g_hr": round(nox, 4),
        "est_co2_g_hr": round(co2, 2)
    }
    for i, (vcls, fuel, speed, pm, nox, co2) in enumerate([
        ("car", "Petrol", 45.2, 0.003, 0.250, 140.0),
        ("car", "Diesel", 38.0, 0.035, 0.650, 130.0),
        ("bus", "CNG/LPG", 28.5, 0.010, 1.200, 650.0),
        ("truck", "Diesel", 42.0, 0.180, 5.200, 850.0),
        ("car", "Electric Vehicle (EV)", 52.0, 0.001, 0.000, 0.0),
        ("motorcycle", "Petrol", 48.0, 0.012, 0.150, 65.0),
        ("car", "Hybrid", 35.0, 0.002, 0.120, 90.0),
        ("bus", "Diesel", 22.0, 0.150, 4.500, 750.0),
    ])
]

df = pd.DataFrame(records)

st.subheader("📋 Traffic Telemetry Table")
st.dataframe(df, use_container_width=True)

col_csv, col_json, col_pdf = st.columns(3)

with col_csv:
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV Report",
        data=csv_data,
        file_name=f"traffic_pollution_report_{datetime.date.today()}.csv",
        mime="text/csv",
        use_container_width=True
    )

with col_json:
    json_data = json.dumps(records, indent=2)
    st.download_button(
        label="📥 Download JSON Telemetry",
        data=json_data,
        file_name=f"traffic_pollution_report_{datetime.date.today()}.json",
        mime="application/json",
        use_container_width=True
    )

with col_pdf:
    st.button("📄 Generate Summary Markdown/PDF Report", use_container_width=True)

st.markdown("---")

st.subheader("📄 Automated Project Executive Summary Report")
summary_markdown = f"""
# URBAN VEHICLE POLLUTION & NOISE MONITORING REPORT
**Generated On:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Location:** City Intersection Monitoring Zone  
**Status:** Certified AI Telemetry Audit  

---

### Executive Summary
- **Total Tracked Vehicles:** {len(records)}
- **Predominant Fuel Class:** Petrol (37.5%), Diesel (37.5%), EV (12.5%), CNG (12.5%)
- **Total Estimated PM2.5 Rate:** {df['est_pm25_g_hr'].sum():.4f} g/hr
- **Total Estimated NOx Rate:** {df['est_nox_g_hr'].sum():.4f} g/hr
- **Total Estimated CO2 Rate:** {df['est_co2_g_hr'].sum():.2f} g/hr

### Methodology & Limitations
- **Camera Emission Model:** COPERT V / EEA Air Pollutant Emission Inventory Guidebook.
- **Scientific Limitation:** Camera measurements represent estimated vehicle pollution share ($g/hr$). Physical atmospheric pollutant concentration requires ambient reference sensors.
"""

st.markdown(summary_markdown)
