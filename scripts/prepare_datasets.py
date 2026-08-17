"""
Dataset Preparation CLI Tool

Scans data/raw/ directories for Stanford Cars, CompCars, UA-DETRAC, AI City, UrbanSound8K,
runs validation checks, fuel mapping, duplicate removal, and exports summary statistics.
"""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data.pipeline import DatasetPipeline
from src.data.fuel_lookup import FuelLookupManager, ALLOWED_FUEL_TYPES
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Dataset Preparation & Pipeline Orchestrator")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw dataset directory")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Path to processed output directory")
    parser.add_argument("--update-mapping", nargs=4, metavar=("MFR", "MODEL", "VARIANT", "FUEL"),
                        help="Add/update fuel mapping: MFR MODEL VARIANT FUEL (e.g. Tesla 'Model 3' Standard EV)")

    args = parser.parse_args()

    if args.update_mapping:
        mfr, model, variant, fuel = args.update_mapping
        manager = FuelLookupManager()
        try:
            manager.update_mapping(mfr, model, variant, fuel, source="CLI User Update")
            print(f"Successfully updated fuel mapping for {mfr} {model} ({variant}) -> {fuel}")
        except Exception as e:
            print(f"Error updating mapping: {e}")
        return

    pipeline = DatasetPipeline(
        raw_dir=Path(args.raw_dir),
        processed_dir=Path(args.processed_dir)
    )
    df = pipeline.run_pipeline()
    print(f"\nPipeline finished! Processed {len(df)} total dataset records.")
    print(f"Metadata CSV: {Path(args.processed_dir) / 'unified_dataset_metadata.csv'}")
    print("Report: outputs/reports/dataset_summary.md\n")

if __name__ == "__main__":
    main()
