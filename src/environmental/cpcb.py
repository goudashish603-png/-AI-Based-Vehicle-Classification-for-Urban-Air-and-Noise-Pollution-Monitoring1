import os
import requests
import datetime
from typing import Optional

from src.environmental.base import EnvironmentalDataAdapter, AmbientAQIMeasurement
from src.utils.logger import get_logger

logger = get_logger(__name__)

class CPCBAdapter(EnvironmentalDataAdapter):
    """
    Adapter for Central Pollution Control Board (CPCB India) Air Quality Monitoring Stations.
    Provides regional Indian station AQI measurements with fallback.
    """
    def __init__(self, station_id: Optional[str] = None):
        self.station_id = station_id or os.environ.get("CPCB_STATION_ID", "CPCB-ND-01")

    def fetch_latest_measurement(self, location_query: str = "Delhi") -> AmbientAQIMeasurement:
        """Fetches station readings for Indian urban monitoring points."""
        # Simulated endpoint adapter logic for standard CPCB station format
        try:
            # If CPCB API URL environment variable is set
            api_url = os.environ.get("CPCB_API_URL", "")
            if api_url:
                resp = requests.get(api_url, timeout=3)
                if resp.status_code == 200:
                    data = resp.json()
                    return AmbientAQIMeasurement(
                        station_id=self.station_id,
                        station_name=data.get("station_name", "CPCB Monitoring Station"),
                        city=location_query,
                        country="IN",
                        pm2_5=float(data.get("pm25", 85.0)),
                        pm10=float(data.get("pm10", 175.0)),
                        no2=float(data.get("no2", 52.0)),
                        co=float(data.get("co", 1.8)),
                        so2=float(data.get("so2", 15.0)),
                        timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        is_live=True
                    )
        except Exception as e:
            logger.warning(f"CPCB request failed ({e}). Returning station reference values.")

        return AmbientAQIMeasurement(
            station_id="CPCB-DELHI-ITO-01",
            station_name="CPCB ITO Intersection Monitoring Station",
            city="Delhi",
            country="IN",
            pm2_5=82.4,
            pm10=168.1,
            no2=54.2,
            co=1.6,
            so2=16.8,
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            is_live=False
        )
