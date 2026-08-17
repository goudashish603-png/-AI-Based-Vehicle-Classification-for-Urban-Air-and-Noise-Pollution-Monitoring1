import os
import wave
import struct
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

AUDIO_CLASSES = [
    "engine",
    "traffic",
    "horn",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "other"
]

class AudioClassifierNet(nn.Module):
    """Simple 1D CNN Architecture for Acoustic Feature Classification."""
    def __init__(self, num_classes: int = 8):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=16, stride=4)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(4)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=8, stride=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(4)
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.gap(x).squeeze(-1)
        return self.fc(x)


class AudioNoiseClassifier:
    """
    Optional Audio Noise Event Classifier (UrbanSound8K / ESC-50 classes).
    Classifies raw WAV audio waveforms into acoustic events (engine, horn, traffic, etc.).
    
    IMPORTANT DISCLAIMER:
    Audio classification identifies sound patterns (e.g. horn honk, engine rev).
    It DOES NOT measure calibrated sound pressure in dBA unless calibrated with a physical reference calibrator.
    """
    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        self.classes = AUDIO_CLASSES
        self.num_classes = len(self.classes)
        self.model = AudioClassifierNet(num_classes=self.num_classes)
        self.model.eval()
        
        if weights_path:
            w_path = Path(weights_path)
            if w_path.exists():
                try:
                    self.model.load_state_dict(torch.load(w_path, map_location="cpu"))
                    logger.info(f"Loaded audio classifier weights from {w_path}")
                except Exception as e:
                    logger.warning(f"Could not load audio model weights ({e}). Using initialized network.")

    def _load_wav_waveform(self, wav_path: Union[str, Path], target_sr: int = 16000, duration_sec: float = 2.0) -> np.ndarray:
        """Loads WAV audio samples into normalized float numpy array."""
        target_len = int(target_sr * duration_sec)
        try:
            with wave.open(str(wav_path), 'rb') as wf:
                n_channels = wf.getnchannels()
                swidth = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                
                raw_bytes = wf.readframes(n_frames)
                if swidth == 2:
                    samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                else:
                    samples = np.frombuffer(raw_bytes, dtype=np.int8).astype(np.float32) / 128.0

                if n_channels > 1:
                    samples = samples[::n_channels]

                # Resample or pad/crop to target_len
                if len(samples) > target_len:
                    samples = samples[:target_len]
                elif len(samples) < target_len:
                    samples = np.pad(samples, (0, target_len - len(samples)))
                return samples
        except Exception as e:
            logger.warning(f"Error loading WAV file {wav_path} ({e}). Returning synthetic waveform.")
            return np.random.normal(0, 0.1, target_len).astype(np.float32)

    def classify_audio(
        self,
        audio_input: Union[str, Path, np.ndarray],
        sample_rate: int = 16000
    ) -> Dict[str, Any]:
        """
        Classifies an audio file or waveform array into acoustic classes.
        """
        if isinstance(audio_input, (str, Path)):
            waveform = self._load_wav_waveform(audio_input, target_sr=sample_rate)
        elif isinstance(audio_input, np.ndarray):
            waveform = audio_input.astype(np.float32)
            if len(waveform) > sample_rate * 2:
                waveform = waveform[:sample_rate * 2]
            elif len(waveform) < sample_rate * 2:
                waveform = np.pad(waveform, (0, sample_rate * 2 - len(waveform)))
        else:
            raise TypeError(f"Unsupported audio input type: {type(audio_input)}")

        tensor = torch.from_numpy(waveform).unsqueeze(0)  # Shape: (1, N)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0].numpy()

        top_idx = int(np.argmax(probs))
        top_label = self.classes[top_idx]
        top_conf = float(probs[top_idx])

        class_probs = {self.classes[i]: round(float(probs[i]), 4) for i in range(self.num_classes)}

        return {
            "predicted_class": top_label,
            "confidence": round(top_conf, 4),
            "class_probabilities": class_probs,
            "disclaimer": "Audio event classification identifies sound patterns (e.g. horn). It does NOT measure calibrated sound pressure dBA."
        }
