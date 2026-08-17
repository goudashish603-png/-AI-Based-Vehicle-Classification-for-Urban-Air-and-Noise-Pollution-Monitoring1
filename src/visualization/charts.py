import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any, List

# Standardized Smart City Color Maps
FUEL_COLOR_MAP = {
    "Petrol": "#3B82F6",    # Blue
    "PETROL": "#3B82F6",
    "Diesel": "#4B5563",    # Dark Gray
    "DIESEL": "#4B5563",
    "EV": "#10B981",        # Green
    "CNG_LPG": "#F97316",   # Orange
    "CNG": "#F97316",
    "Hybrid": "#8B5CF6",    # Purple
    "HYBRID": "#8B5CF6",
    "Unknown": "#9CA3AF",   # Light Gray
    "UNKNOWN": "#9CA3AF",
    "AMBIGUOUS": "#9CA3AF"
}

POLLUTION_COLOR_MAP = {
    "Low": "#10B981",       # Green
    "Moderate": "#FBBF24",  # Yellow
    "High": "#F97316",      # Orange
    "Very High": "#EF4444", # Red
    "Severe": "#EF4444"
}

def create_fuel_donut_chart(fuel_counts: Dict[str, int]) -> go.Figure:
    """Generates fuel distribution donut chart matching requested color scheme."""
    labels = []
    values = []
    colors = []

    for k, v in fuel_counts.items():
        if v > 0 or True:
            lbl = k.capitalize()
            labels.append(lbl)
            values.append(v)
            colors.append(FUEL_COLOR_MAP.get(k, "#9CA3AF"))

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.5,
        marker=dict(colors=colors),
        textinfo='label+percent',
        hoverinfo='label+value+percent'
    )])
    fig.update_layout(
        title="Powertrain Fuel Type Distribution",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    return fig

# Alias for backwards compatibility
create_fuel_distribution_chart = create_fuel_donut_chart

def create_vehicle_pie_chart(type_counts: Dict[str, int]) -> go.Figure:
    """Generates vehicle category pie chart."""
    labels = [k.capitalize() for k in type_counts.keys()]
    values = list(type_counts.values())

    fig = px.pie(
        names=labels,
        values=values,
        title="Vehicle Category Distribution",
        template="plotly_dark",
        hole=0.3
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    return fig

# Alias for backwards compatibility
create_vehicle_distribution_chart = create_vehicle_pie_chart

def create_emissions_bar_chart(pollutant_dict: Dict[str, float]) -> go.Figure:
    """Generates pollutant mass emission rate bar chart."""
    df = pd.DataFrame(list(pollutant_dict.items()), columns=["Pollutant", "Mass Rate (g/hr)"])
    fig = px.bar(
        df,
        x="Pollutant",
        y="Mass Rate (g/hr)",
        color="Pollutant",
        title="Estimated Mass Emission Rates (g/hr)",
        template="plotly_dark"
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_fuel_emissions_bar_chart(fuel_emissions: Dict[str, Dict[str, float]]) -> go.Figure:
    """Generates pollution contribution grouped by fuel type."""
    records = []
    for fuel, pols in fuel_emissions.items():
        f_norm = fuel.capitalize()
        for pol, val in pols.items():
            if pol in ["NO2", "PM2.5", "CO2"]:
                records.append({"Fuel": f_norm, "Pollutant": pol, "g/hr": val})

    df = pd.DataFrame(records)
    if df.empty:
        df = pd.DataFrame([{"Fuel": "Petrol", "Pollutant": "NO2", "g/hr": 0.0}])

    fig = px.bar(
        df,
        x="Fuel",
        y="g/hr",
        color="Fuel",
        color_discrete_map=FUEL_COLOR_MAP,
        title="Estimated Emissions Contribution by Fuel Type",
        template="plotly_dark",
        barmode="group"
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_noise_meter_chart(noise_val: float = 50.0) -> go.Figure:
    """Generates gauge indicator chart for relative noise pollution index."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=noise_val,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Relative Noise Pollution Index (0-100)"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#38BDF8"},
            'steps': [
                {'range': [0, 25], 'color': "#10B981"},
                {'range': [25, 50], 'color': "#FBBF24"},
                {'range': [50, 75], 'color': "#F97316"},
                {'range': [75, 100], 'color': "#EF4444"}
            ]
        }
    ))
    fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
    return fig
