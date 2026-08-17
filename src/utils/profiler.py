import time
import os
from typing import Dict, Any

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class PerformanceProfiler:
    """
    Real-Time Performance Profiler.
    Measures component latency breakdown (preprocessing, detection, tracking, classification, postprocessing),
    throughput (FPS), and process memory footprint (MB).
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.t_start = time.time()
        self.preproc_ms = 0.0
        self.detection_ms = 0.0
        self.tracking_ms = 0.0
        self.classification_ms = 0.0
        self.postproc_ms = 0.0
        self.total_frames = 0

    def start_timer() -> float:
        return time.time()

    def stop_timer(self, t0: float) -> float:
        return (time.time() - t0) * 1000.0  # Return milliseconds

    def get_memory_usage_mb(self) -> float:
        """Returns current process RAM memory usage in Megabytes."""
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            return round(process.memory_info().rss / (1024.0 * 1024.0), 2)
        return 0.0

    def get_performance_summary(self, frame_count: int) -> Dict[str, Any]:
        total_time_sec = max(0.001, time.time() - self.t_start)
        fps = round(frame_count / total_time_sec, 2)
        avg_frame_ms = round((total_time_sec * 1000.0) / max(1, frame_count), 2)

        return {
            "total_frames_processed": frame_count,
            "total_processing_time_sec": round(total_time_sec, 2),
            "processing_fps": fps,
            "avg_latency_per_frame_ms": avg_frame_ms,
            "latency_breakdown_ms": {
                "preprocessing_ms": round(self.preproc_ms / max(1, frame_count), 2),
                "detection_ms": round(self.detection_ms / max(1, frame_count), 2),
                "tracking_ms": round(self.tracking_ms / max(1, frame_count), 2),
                "classification_ms": round(self.classification_ms / max(1, frame_count), 2),
                "postprocessing_ms": round(self.postproc_ms / max(1, frame_count), 2)
            },
            "memory_usage_mb": self.get_memory_usage_mb()
        }
