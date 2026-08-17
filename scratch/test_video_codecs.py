import cv2
import numpy as np
from pathlib import Path

out_dir = Path("scratch")
out_dir.mkdir(exist_ok=True)

codecs_to_test = ['avc1', 'H264', 'mp4v', 'XVID', 'MJPG']

frame = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(frame, "Test Video Codec", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

for codec in codecs_to_test:
    test_path = out_dir / f"test_{codec}.mp4"
    try:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(test_path), fourcc, 25.0, (640, 480))
        if writer.isOpened():
            for _ in range(25):
                writer.write(frame)
            writer.release()
            print(f"Codec {codec}: SUCCESS (file size: {test_path.stat().st_size} bytes)")
        else:
            print(f"Codec {codec}: FAILED (isOpened returned False)")
    except Exception as e:
        print(f"Codec {codec}: ERROR ({e})")
