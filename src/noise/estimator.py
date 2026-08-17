import math
import numpy as np
import wave
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from src.utils.config import load_emissions_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class NoiseEstimate:
    equivalent_sound_level_dba: float # Estimated Leq in dBA
    peak_sound_level_dba: float      # Estimated Lmax in dBA
    relative_noise_index: float       # 0 - 100 Index Scale
    noise_category: str               # Quiet, Moderate, High, Severe
    acoustic_features: Optional[Dict[str, float]] = None


class NoisePollutionEstimator:
    """
    Traffic Noise Propagation Model (CoRTN Standard) & Audio Signal Processing Engine.
    Estimates Relative Noise Pollution Index (dBA) from traffic visual metrics and acoustic streams.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.emissions_cfg = config or load_emissions_config()
        self.base_noise_factors = self.emissions_cfg.get("noise_emission_factors", {
            "car": 72.0, "motorcycle": 78.0, "bus": 84.0, "truck": 86.0
        })
        self.speed_corr = self.emissions_cfg.get("speed_correction_factor", 0.2)

    def estimate_from_traffic(
        self,
        vehicle_records: List[Dict[str, Any]],
        distance_meters: float = 10.0
    ) -> NoiseEstimate:
        """
        Calculates road traffic noise (CoRTN model) based on vehicle composition and speeds.
        
        Args:
            vehicle_records: List of dicts with 'class_name', 'speed_kmh'
            distance_meters: Distance from road centerline to observer
            
        Returns:
            NoiseEstimate object
        """
        if not vehicle_records:
            # Ambient background noise level (~45 dBA)
            return NoiseEstimate(45.0, 48.0, 15.0, "Quiet (Ambient)")

        sum_intensity = 0.0
        max_single_lba = 0.0

        for record in vehicle_records:
            v_cls = record.get("class_name", "car").lower()
            speed = max(10.0, record.get("speed_kmh", 50.0))

            # Base sound power level at 10m
            base_lba = self.base_noise_factors.get(v_cls, 72.0)
            
            # Speed correction factor
            speed_delta = (speed - 50.0) / 10.0
            lba = base_lba + (speed_delta * self.speed_corr)

            # Distance attenuation factor: 20 * log10(R / 10)
            dist_attenuation = 20.0 * math.log10(max(1.0, distance_meters) / 10.0)
            lba_at_observer = lba - dist_attenuation

            # Energy summation: 10^(L/10)
            sum_intensity += 10.0 ** (lba_at_observer / 10.0)
            max_single_lba = max(max_single_lba, lba_at_observer)

        # Compute equivalent continuous sound level Leq
        leq_dba = 10.0 * math.log10(max(1e-6, sum_intensity))
        peak_dba = max(leq_dba + 3.0, max_single_lba + 2.0)

        # Scale to 0-100 Relative Noise Index (Baseline 40 dBA = index 0, 90 dBA = index 100)
        noise_idx = min(100.0, max(0.0, (leq_dba - 40.0) * 2.0))

        # Classify noise level category according to WHO urban noise standards
        if leq_dba < 55.0:
            category = "Quiet / Moderate"
        elif leq_dba < 70.0:
            category = "High Urban Noise"
        elif leq_dba < 82.0:
            category = "Severe Traffic Noise"
        else:
            category = "Hazardous Acoustic Level"

        return NoiseEstimate(
            equivalent_sound_level_dba=round(leq_dba, 2),
            peak_sound_level_dba=round(peak_dba, 2),
            relative_noise_index=round(noise_idx, 1),
            noise_category=category
        )

    def analyze_audio_file(self, wav_path: str) -> NoiseEstimate:
        """
        Analyzes acoustic WAV audio file to compute RMS sound pressure level and peak decibels.
        """
        try:
            with wave.open(wav_path, 'r') as wf:
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                
                frames = wf.readframes(n_frames)
                
                # Parse 16-bit PCM samples
                if sample_width == 2:
                    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
                else:
                    samples = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0

                if n_channels > 1:
                    samples = samples[::n_channels]

                # Calculate Root Mean Square (RMS)
                rms = np.sqrt(np.mean(samples ** 2)) + 1e-6
                peak = np.max(np.abs(samples)) + 1e-6

                # Convert RMS to relative dBA scale (calibrated against 16-bit PCM max 32768)
                db_rms = 20.0 * np.log10(rms / 32768.0) + 94.0 # 94 dBA reference SPL
                db_peak = 20.0 * np.log10(peak / 32768.0) + 94.0

                noise_idx = min(100.0, max(0.0, (db_rms - 40.0) * 2.0))

                if db_rms < 55.0:
                    category = "Quiet / Moderate"
                elif db_rms < 70.0:
                    category = "High Urban Noise"
                else:
                    category = "Severe Traffic Noise"

                features = {
                    "rms_amplitude": float(rms),
                    "peak_amplitude": float(peak),
                    "duration_sec": float(n_frames / framerate),
                    "sample_rate": float(framerate)
                }

                return NoiseEstimate(
                    equivalent_sound_level_dba=round(db_rms, 2),
                    peak_sound_level_dba=round(db_peak, 2),
                    relative_noise_index=round(noise_idx, 1),
                    noise_category=category,
                    acoustic_features=features
                )
        except Exception as e:
            logger.error(f"Error parsing audio file {wav_path}: {e}")
            return NoiseEstimate(50.0, 55.0, 20.0, "Quiet (Fallback)")
