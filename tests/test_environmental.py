import pytest
import pandas as pd
import numpy as np

from src.environmental.base import EnvironmentalMeasurement, normalize_pollutant_name
from src.environmental.openaq_client import OpenAQAdapter
from src.environmental.cpcb_client import CPCBAdapter
from src.environmental.environmental_analysis import EnvironmentalAnalyzer

def test_environmental_measurement_schema():
    m = EnvironmentalMeasurement(
        timestamp="2026-08-12 12:00:00",
        location="Test Station",
        latitude=28.6139,
        longitude=77.2090,
        pollutant="pm25",
        value=45.2,
        unit="µg/m³",
        source="OpenAQ"
    )
    assert m.pollutant == "PM2.5"
    assert m.latitude == 28.6139
    assert m.longitude == 77.2090
    assert m.source == "OpenAQ"

def test_pollutant_normalization():
    assert normalize_pollutant_name("pm2.5") == "PM2.5"
    assert normalize_pollutant_name("pm25") == "PM2.5"
    assert normalize_pollutant_name("no2") == "NO2"
    assert normalize_pollutant_name("co") == "CO"

def test_openaq_and_cpcb_adapters_fallback():
    openaq = OpenAQAdapter(api_key="")
    m_openaq = openaq.fetch_latest_measurement("Delhi")
    assert isinstance(m_openaq, EnvironmentalMeasurement)
    assert not m_openaq.is_live
    assert "Offline" in m_openaq.source

    cpcb = CPCBAdapter(api_key="")
    m_cpcb = cpcb.fetch_latest_measurement("Delhi")
    assert isinstance(m_cpcb, EnvironmentalMeasurement)
    assert not m_cpcb.is_live
    assert "Offline" in m_cpcb.source

def test_correlation_analysis():
    analyzer = EnvironmentalAnalyzer()
    
    # Perfectly correlated series
    traffic = [10, 20, 30, 40, 50, 60, 70, 80]
    pm25 = [15, 25, 35, 45, 55, 65, 75, 85]
    
    res = analyzer.calculate_correlation(traffic, pm25)
    assert pytest.approx(res["pearson_r"], 0.01) == 1.0
    assert pytest.approx(res["spearman_rho"], 0.01) == 1.0
    assert "NOT establish direct physical causation" in res["disclaimer"]

def test_dataframe_correlation():
    analyzer = EnvironmentalAnalyzer()
    df = pd.DataFrame({
        "traffic_count": [100, 150, 200, 250, 300],
        "ambient_pm25": [40, 55, 65, 70, 85]
    })
    
    res = analyzer.analyze_traffic_pollution_dataframe(df, "traffic_count", "ambient_pm25")
    assert res["pearson_r"] > 0.90
