"""
Sample Data Generator Script

Creates synthetic sample traffic images, video clips, and noise audio files 
in the data/ raw directory for immediate pipeline testing.
"""
import os
import cv2
import numpy as np
from pathlib import Path
import wave
import struct

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

def generate_sample_images():
    """Generates synthetic traffic scene images containing cars, buses, and trucks."""
    img_dir = RAW_DIR / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    for i in range(3):
        # Create a road scene (background: asphalt grey with lane markers)
        frame = np.ones((720, 1280, 3), dtype=np.uint8) * 60  # dark grey asphalt
        # Draw green side grass
        frame[0:200, :] = [34, 139, 34]
        # Draw yellow double lane divider
        cv2.line(frame, (0, 460), (1280, 460), (0, 215, 255), 4)
        cv2.line(frame, (0, 470), (1280, 470), (0, 215, 255), 4)

        # Draw vehicles (Car: blue box, Bus: red box, Truck: green box)
        # Vehicle 1: Sedan Car
        x1, y1, w1, h1 = 150 + i * 40, 280, 220, 120
        cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (220, 100, 50), -1)  # Body
        cv2.rectangle(frame, (x1 + 30, y1 + 15), (x1 + 160, y1 + 75), (255, 200, 150), -1) # Roof/Window
        cv2.putText(frame, "SEDAN CAR", (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Vehicle 2: City Bus
        x2, y2, w2, h2 = 500 - i * 30, 500, 360, 160
        cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (50, 50, 220), -1)  # Bus body
        cv2.rectangle(frame, (x2 + 20, y2 + 20), (x2 + 340, y2 + 70), (200, 240, 255), -1) # Windows
        cv2.putText(frame, "TRANSIT BUS", (x2 + 10, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Vehicle 3: Delivery Truck
        x3, y3, w3, h3 = 900 + i * 20, 320, 280, 140
        cv2.rectangle(frame, (x3, y3), (x3 + w3, y3 + h3), (50, 180, 50), -1)
        cv2.putText(frame, "CARGO TRUCK", (x3 + 10, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        output_path = img_dir / f"traffic_sample_{i+1}.jpg"
        cv2.imwrite(str(output_path), frame)
        print(f"Generated sample image: {output_path}")

def generate_sample_video():
    """Generates a synthetic traffic video clip (MP4)."""
    vid_dir = RAW_DIR / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)
    video_path = vid_dir / "sample_traffic.mp4"

    fps = 25
    duration_sec = 5
    num_frames = fps * duration_sec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (1280, 720))

    for frame_idx in range(num_frames):
        frame = np.ones((720, 1280, 3), dtype=np.uint8) * 55
        # Road lane markers
        cv2.line(frame, (0, 360), (1280, 360), (255, 255, 255), 2)
        
        # Moving vehicle 1 (Left to Right)
        x_car = int((frame_idx * 12) % 1350 - 150)
        cv2.rectangle(frame, (x_car, 400), (x_car + 180, 490), (220, 120, 40), -1)
        cv2.putText(frame, "CAR", (x_car, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Moving vehicle 2 (Right to Left)
        x_truck = int(1280 - ((frame_idx * 8) % 1400 - 100))
        cv2.rectangle(frame, (x_truck, 220), (x_truck + 240, 320), (40, 180, 80), -1)
        cv2.putText(frame, "TRUCK", (x_truck, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(frame)

    out.release()
    print(f"Generated sample video: {video_path}")

def generate_sample_audio():
    """Generates a synthetic WAV audio clip representing ambient traffic noise."""
    audio_dir = RAW_DIR / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / "traffic_noise.wav"

    sample_rate = 22050
    duration_sec = 3
    num_samples = sample_rate * duration_sec

    # Combine low frequency hum (engine) + white noise (tire roll/wind)
    t = np.linspace(0, duration_sec, num_samples, False)
    engine_hum = 0.4 * np.sin(2 * np.pi * 120 * t) + 0.2 * np.sin(2 * np.pi * 240 * t)
    white_noise = 0.15 * np.random.normal(0, 1, num_samples)
    audio_signal = engine_hum + white_noise

    # Normalize to 16-bit PCM range
    audio_signal = np.clip(audio_signal, -1.0, 1.0)
    scaled = (audio_signal * 32767).astype(np.int16)

    with wave.open(str(audio_path), 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
        wav_file.setframerate(sample_rate)
        for sample in scaled:
            wav_file.writeframes(struct.pack('<h', sample))

    print(f"Generated sample audio: {audio_path}")

if __name__ == "__main__":
    print("Preparing synthetic sample datasets...")
    generate_sample_images()
    generate_sample_video()
    generate_sample_audio()
    print("Sample data preparation complete!")
