"""
Admin Utility for Updating & Querying Vehicle Fuel Mapping Database

Manages data/external/fuel_mapping.csv lookup table.
"""
import argparse
import sys
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.fuel_mapping.fuel_mapper import FuelTypeMapper, ALLOWED_FUEL_TYPES
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Admin Utility for Fuel Mapping Database")
    parser.add_argument("--query", nargs=2, metavar=("MFR", "MODEL"), help="Query fuel classification for MFR MODEL")
    parser.add_argument("--add", nargs=4, metavar=("MFR", "MODEL", "VARIANT", "FUEL"), help="Add record: MFR MODEL VARIANT FUEL")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence score for new record")
    parser.add_argument("--source", type=str, default="Admin CLI", help="Source for new record")
    parser.add_argument("--source-url", type=str, default="", help="Source URL for new record")
    parser.add_argument("--notes", type=str, default="", help="Notes for new record")
    parser.add_argument("--list", action="store_true", help="List all fuel mapping entries")

    args = parser.parse_args()
    mapper = FuelTypeMapper()

    if args.query:
        mfr, model = args.query
        res = mapper.infer_fuel_type(mfr, model)
        print("\n" + "=" * 60)
        print("FUEL INFERENCE QUERY RESULT")
        print("=" * 60)
        print(f"Manufacturer    : {mfr}")
        print(f"Model           : {model}")
        print(f"Inferred Fuel   : {res.fuel_type}")
        print(f"Confidence      : {res.confidence:.2f}")
        print(f"Mapping Source  : {res.mapping_source}")
        print(f"Notes           : {res.mapping_notes}")
        print("=" * 60 + "\n")
        return

    if args.add:
        mfr, model, variant, fuel = args.add
        fuel_upper = fuel.upper()
        if fuel_upper not in ALLOWED_FUEL_TYPES:
            print(f"Error: Invalid fuel label '{fuel_upper}'. Must be one of {ALLOWED_FUEL_TYPES}")
            sys.exit(1)

        # Check existing
        df = mapper.mapping_df
        mask = (
            (df["manufacturer"].str.lower() == mfr.lower()) &
            (df["model"].str.lower() == model.lower()) &
            (df["variant"].str.lower() == variant.lower())
        )

        if mask.any():
            df.loc[mask, "fuel_type"] = fuel_upper
            df.loc[mask, "confidence"] = args.confidence
            df.loc[mask, "source"] = args.source
            df.loc[mask, "source_url"] = args.source_url
            df.loc[mask, "notes"] = args.notes
            print(f"Updated existing record for {mfr} {model} ({variant}) -> {fuel_upper}")
        else:
            new_row = pd.DataFrame([{
                "manufacturer": mfr,
                "model": model,
                "variant": variant,
                "fuel_type": fuel_upper,
                "confidence": args.confidence,
                "source": args.source,
                "source_url": args.source_url,
                "notes": args.notes
            }])
            df = pd.concat([df, new_row], ignore_index=True)
            print(f"Added new record for {mfr} {model} ({variant}) -> {fuel_upper}")

        df.to_csv(mapper.csv_path, index=False)
        print(f"Saved fuel mapping database to {mapper.csv_path}")
        return

    if args.list:
        print("\n" + "=" * 60)
        print(f"FUEL MAPPING DATABASE ({len(mapper.mapping_df)} Records)")
        print("=" * 60)
        print(mapper.mapping_df[["manufacturer", "model", "variant", "fuel_type", "confidence"]].to_string(index=False))
        print("=" * 60 + "\n")
        return

    parser.print_help()

if __name__ == "__main__":
    main()
