import os
import wave
from pathlib import Path
from typing import List, Dict, Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

class UrbanSoundProcessor:
    """
    Processor for UrbanSound8K and ESC-50 environmental audio datasets.
    """
    def __init__(self, raw_dir: Path):
        self.raw_dir = Path(raw_dir)

    def exists(self) -> bool:
        return self.raw_dir.exists() and len(list(self.raw_dir.rglob("*.wav"))) > 0

    def process(self) -> List[Dict[str, Any]]:
        records = []
        if not self.exists():
            logger.info("UrbanSound8K / ESC-50 dataset not found in raw data directory.")
            return records

        wav_paths = list(self.raw_dir.rglob("*.wav"))
        logger.info(f"Processing {len(wav_paths)} audio files in UrbanSound dataset...")

        for p in wav_paths:
            try:
                with wave.open(str(p), 'rb') as wf:
                    duration = float(wf.getnframes() / wf.getframerate())
                    channels = wf.getnchannels()
            except Exception:
                duration = 0.0
                channels = 0

            records.append({
                "audio_path": str(p.resolve()),
                "dataset": "UrbanSound8K",
                "label": "traffic_noise" if "engine" in p.name.lower() or "car" in p.name.lower() else "ambient",
                "duration_sec": round(duration, 2),
                "channels": channels,
                "split": "train" if hash(p.name) % 10 < 8 else "test"
            })

        return records
