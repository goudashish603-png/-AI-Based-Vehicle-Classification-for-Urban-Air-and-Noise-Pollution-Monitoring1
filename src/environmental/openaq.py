import os
import requests
import datetime
from typing import Dict, Any, Optional

from src.environmental.base import EnvironmentalDataAdapter, AmbientAQIMeasurement
from src.utils.logger import get_logger

logger = get_logger(__name__)

class OpenAQAdapter(EnvironmentalDataAdapter):
    """
    Adapter for OpenAQ Global Air Quality API (v2 / v3).
    Provides real-time PM2.5, PM10, NO2, CO measurements with offline fallback.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAQ_API_KEY", "")
        self.base_url = "https://api.openaq.org/v2"

    def fetch_latest_measurement(self, location_query: str = "Delhi") -> AmbientAQIMeasurement:
        """Fetches measurements from OpenAQ or returns cached mock ambient readings."""
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        try:
            url = f"{self.base_url}/measurements?city={location_query}&limit=10"
            resp = requests.get(url, headers=headers, timeout=4)
            
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                
                if results:
                    params: Dict[str, float] = {}
                    station_name = results[0].get("location", "OpenAQ Station")
                    
                    for r in results:
                        param = r.get("parameter", "").lower()
                        val = r.get("value", 0.0)
                        params[param] = float(val)

                    return AmbientAQIMeasurement(
                        station_id="OPENAQ-" + location_query,
                        station_name=station_name,
                        city=location_query,
                        country="IN",
                        pm2_5=params.get("pm25", 68.5),
                        pm10=params.get("pm10", 142.0),
                        no2=params.get("no2", 45.2),
                        co=params.get("co", 1.2),
                        so2=params.get("so2", 12.4),
                        timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        is_live=True
                    )
        except Exception as e:
            logger.warning(f"OpenAQ API request failed or timed out ({e}). Using offline baseline.")

        # Offline Baseline Fallback
        return AmbientAQIMeasurement(
            station_id="OPENAQ-FALLBACK-01",
            station_name=f"Station Baseline ({location_query})",
            city=location_query,
            country="IN",
            pm2_5=74.2,
            pm10=156.8,
            no2=48.6,
            co=1.4,
            so2=14.1,
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            is_live=False
        )
