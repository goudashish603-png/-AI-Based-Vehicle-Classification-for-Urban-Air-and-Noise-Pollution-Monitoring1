import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.pipeline import EndToEndPipeline

def main():
    print("Reprocessing predictions with updated FuelTypeMapper...")
    pipeline = EndToEndPipeline(conf_threshold=0.40, process_every_n_frames=2)
    
    # Check for uploaded video or sample video
    out_dir = BASE_DIR / "outputs" / "predictions"
    uploaded_vids = list(out_dir.glob("uploaded_*.mp4"))
    
    if uploaded_vids:
        target_vid = uploaded_vids[0]
        print(f"Processing uploaded video: {target_vid.name}")
    else:
        target_vid = BASE_DIR / "data" / "raw" / "videos" / "sample_traffic.mp4"
        print(f"Processing sample video: {target_vid.name}")

    summary = pipeline.process_video(target_vid, output_dir=out_dir)
    print("\nREPROCESSED SUMMARY RESULT:")
    print(f"Total Vehicles: {summary['total_unique_vehicles']}")
    print(f"Petrol: {summary['petrol_count']}")
    print(f"Diesel: {summary['diesel_count']}")
    print(f"EV: {summary['ev_count']}")
    print(f"CNG/LPG: {summary['cng_lpg_count']}")
    print(f"Hybrid: {summary['hybrid_count']}")
    print(f"Unknown: {summary['unknown_count']}")

if __name__ == "__main__":
    main()
