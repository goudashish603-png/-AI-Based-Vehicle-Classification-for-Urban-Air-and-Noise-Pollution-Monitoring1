import os
import time
import datetime
import urllib.request
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from src.environmental.base import EnvironmentalMeasurement, normalize_pollutant_name
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CPCB_BASELINE = {
    "location": "CPCB ITO Station Delhi",
    "latitude": 28.6289,
    "longitude": 77.2409,
    "pm2_5": 62.5,
    "pm10": 125.0,
    "no2": 48.0,
    "co": 1.5,
    "so2": 9.2,
    "o3": 28.0,
    "aqi": 165.0
}

class CPCBAdapter:
    """
    CPCB (Central Pollution Control Board India) Data Client Adapter.
    Uses environment variable CPCB_API_KEY.
    Provides response caching and graceful offline baseline fallback.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_expire_seconds: int = 3600
    ):
        self.api_key = api_key or os.getenv("CPCB_API_KEY", "")
        self.base_url = os.getenv("CPCB_BASE_URL", "https://api.cpcb.gov.in")
        self.cache_ttl = cache_expire_seconds
        
        self._cache: Dict[str, Tuple[float, EnvironmentalMeasurement]] = {}

    def fetch_latest_measurement(self, location_query: str = "Delhi") -> EnvironmentalMeasurement:
        """
        Fetches latest ambient air quality measurement for a given location query.
        Returns cached or offline station baseline if API key is missing or server fails.
        """
        now = time.time()
        cache_key = location_query.lower().strip()

        # 1. Check local cache
        if cache_key in self._cache:
            cached_time, cached_measurement = self._cache[cache_key]
            if now - cached_time < self.cache_ttl:
                logger.info(f"Returning cached CPCB measurement for '{location_query}'")
                return cached_measurement

        # 2. Attempt live API request if API key is provided
        if self.api_key:
            try:
                url = f"{self.base_url}/station_data?city={urllib.parse.quote(location_query)}"
                req = urllib.request.Request(url, headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "User-Agent": "AIVehiclePollutionMonitor/1.0"
                })
                with urllib.request.urlopen(req, timeout=5) as response:
                    data = json.loads(response.read().decode())
                    if data and "station" in data:
                        m_obj = self._parse_cpcb_response(data, location_query)
                        self._cache[cache_key] = (now, m_obj)
                        logger.info(f"Successfully fetched live CPCB measurement for '{location_query}'")
                        return m_obj
            except Exception as e:
                logger.warning(f"CPCB API request failed ({e}). Defaulting to offline station baseline.")

        # 3. Graceful Offline Baseline Fallback
        ts_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        baseline = EnvironmentalMeasurement(
            timestamp=ts_now,
            location=f"CPCB {location_query} Station",
            latitude=DEFAULT_CPCB_BASELINE["latitude"],
            longitude=DEFAULT_CPCB_BASELINE["longitude"],
            pollutant="PM2.5",
            value=DEFAULT_CPCB_BASELINE["pm2_5"],
            unit="µg/m³",
            source="CPCB Station Baseline (Offline)",
            is_live=False,
            station_name=f"CPCB {location_query} Station",
            pm2_5=DEFAULT_CPCB_BASELINE["pm2_5"],
            pm10=DEFAULT_CPCB_BASELINE["pm10"],
            no2=DEFAULT_CPCB_BASELINE["no2"],
            co=DEFAULT_CPCB_BASELINE["co"],
            so2=DEFAULT_CPCB_BASELINE["so2"],
            o3=DEFAULT_CPCB_BASELINE["o3"],
            aqi=DEFAULT_CPCB_BASELINE["aqi"]
        )

        self._cache[cache_key] = (now, baseline)
        return baseline

    def _parse_cpcb_response(self, item: Dict[str, Any], query: str) -> EnvironmentalMeasurement:
        ts = item.get("timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        loc = item.get("station", f"CPCB {query} Station")
        lat = float(item.get("latitude", 28.6289))
        lon = float(item.get("longitude", 77.2409))

        return EnvironmentalMeasurement(
            timestamp=str(ts),
            location=str(loc),
            latitude=lat,
            longitude=lon,
            pollutant="PM2.5",
            value=float(item.get("pm2_5", 60.0)),
            unit="µg/m³",
            source="CPCB Live API",
            is_live=True,
            pm2_5=float(item.get("pm2_5", 60.0)),
            pm10=float(item.get("pm10", 120.0)),
            no2=float(item.get("no2", 45.0)),
            co=float(item.get("co", 1.4)),
            so2=float(item.get("so2", 8.0)),
            o3=float(item.get("o3", 25.0)),
            aqi=float(item.get("aqi", 150.0))
        )
