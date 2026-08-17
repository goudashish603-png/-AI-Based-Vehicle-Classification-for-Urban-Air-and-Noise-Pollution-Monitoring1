import re
import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union

POLLUTANT_NAME_MAP = {
    "pm25": "PM2.5",
    "pm2.5": "PM2.5",
    "pm2_5": "PM2.5",
    "pm10": "PM10",
    "no2": "NO2",
    "co": "CO",
    "so2": "SO2",
    "o3": "O3",
    "ozone": "O3",
    "aqi": "AQI"
}

@dataclass
class EnvironmentalMeasurement:
    """
    Standardized Environmental Station Measurement Object.
    Matches required common schema:
    timestamp, location, latitude, longitude, pollutant, value, unit, source
    """
    timestamp: str
    location: str
    latitude: float
    longitude: float
    pollutant: str
    value: float
    unit: str
    source: str
    is_live: bool = False
    station_name: str = ""
    pm2_5: float = 0.0
    pm10: float = 0.0
    no2: float = 0.0
    co: float = 0.0
    so2: float = 0.0
    o3: float = 0.0
    aqi: float = 0.0

    def __post_init__(self):
        if not self.station_name:
            self.station_name = self.location
        # Normalize pollutant string
        pol_clean = str(self.pollutant).lower().replace(" ", "").replace("_", "")
        self.pollutant = POLLUTANT_NAME_MAP.get(pol_clean, self.pollutant.upper())


def normalize_pollutant_name(raw_name: str) -> str:
    """Normalizes arbitrary pollutant string representations."""
    clean = str(raw_name).lower().replace(" ", "").replace("_", "").replace(".", "")
    return POLLUTANT_NAME_MAP.get(clean, raw_name.upper())
