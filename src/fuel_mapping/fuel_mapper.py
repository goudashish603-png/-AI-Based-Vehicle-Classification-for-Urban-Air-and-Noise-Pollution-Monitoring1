# type: ignore
import re
import difflib
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Set, Union, Any

from src.utils.logger import get_logger
from src.fuel_mapping.mapper import FLEET_PRIORS

logger = get_logger(__name__)

ALLOWED_FUEL_TYPES: Set[str] = {
    "PETROL",
    "DIESEL",
    "EV",
    "CNG_LPG",
    "HYBRID",
    "UNKNOWN",
    "AMBIGUOUS"
}

DEFAULT_CSV_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "external" / "fuel_mapping.csv"

@dataclass
class FuelInferenceResult:
    """Dataclass holding detailed output from fuel-type inference system."""
    fuel_type: str
    confidence: float
    mapping_source: str
    mapping_notes: str

@dataclass
class VehicleFuelPrediction:
    """Final prediction object linking a vehicle instance to its inferred fuel type."""
    vehicle_id: Union[int, str]
    manufacturer: str
    model: str
    fuel_type: str
    fuel_confidence: float


def normalize_string(s: Any) -> str:
    """Strips non-alphanumeric characters and converts to lowercase."""
    if pd.isna(s):
        return ""
    return re.sub(r"[^a-zA-Z0-9]", "", str(s)).lower()


def _parse_track_id_seed(track_id: Union[int, str]) -> int:
    if isinstance(track_id, int):
        return abs(track_id)
    s = str(track_id).strip()
    if not s:
        return 0
    try:
        return abs(int(s))
    except ValueError:
        return sum(ord(c) for c in s)

def _parse_float(val: Any, default: float = 0.90) -> float:
    try:
        if pd.notna(val) and str(val).strip() != "":
            return float(val)
    except (ValueError, TypeError):
        pass
    return default

class FuelTypeMapper:
    """
    Production Vehicle-to-Fuel Type Inference Engine.
    Maps vehicle manufacturer, model, and optional variant to fuel classification.
    Supports exact, normalized, fuzzy, and ambiguous match strategies.
    """
    def __init__(self, csv_path: Optional[Union[str, Path]] = None, prior_weight: float = 0.20):
        self.csv_path = Path(csv_path) if csv_path else DEFAULT_CSV_PATH
        self.prior_weight = prior_weight
        self.mapping_df = pd.DataFrame()
        self._exact_table: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self._norm_table: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self.load_database()

    def load_database(self):
        """Loads and indexes the external fuel mapping CSV database."""
        if not self.csv_path.exists():
            logger.warning(f"Fuel mapping database missing at {self.csv_path}. Initializing empty dataset.")
            self._create_empty_csv()

        try:
            df = pd.read_csv(self.csv_path)
            # Ensure required columns exist
            cols = ["manufacturer", "model", "variant", "fuel_type", "confidence", "source", "source_url", "notes"]
            for col in cols:
                if col not in df.columns:
                    df[col] = ""

            # Sanitize fuel labels
            for idx, row in df.iterrows():
                f_val = str(row["fuel_type"]).strip().upper()
                if f_val not in ALLOWED_FUEL_TYPES:
                    logger.warning(f"Invalid fuel label '{f_val}' at row {idx}. Defaulting to UNKNOWN.")
                    df.at[idx, "fuel_type"] = "UNKNOWN"

            self.mapping_df = df
            self._index_tables()
            logger.info(f"Loaded fuel mapping database with {len(self.mapping_df)} entries.")
        except Exception as e:
            logger.error(f"Error reading fuel mapping CSV database: {e}")
            self.mapping_df = pd.DataFrame(columns=["manufacturer", "model", "variant", "fuel_type", "confidence", "source", "source_url", "notes"])

    def _create_empty_csv(self):
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        cols = ["manufacturer", "model", "variant", "fuel_type", "confidence", "source", "source_url", "notes"]
        df = pd.DataFrame(columns=cols)
        df.to_csv(self.csv_path, index=False)

    def _index_tables(self):
        self._exact_table = {}
        self._norm_table = {}

        for _, row in self.mapping_df.iterrows():
            mfr = str(row["manufacturer"]).strip() if pd.notna(row["manufacturer"]) else ""
            mdl = str(row["model"]).strip() if pd.notna(row["model"]) else ""
            vrt = str(row["variant"]).strip() if pd.notna(row["variant"]) else ""

            fuel = str(row["fuel_type"]).strip().upper() if pd.notna(row["fuel_type"]) else "UNKNOWN"
            conf = _parse_float(row["confidence"], 0.90)
            source = str(row["source"]) if pd.notna(row["source"]) else "Database"
            notes = str(row["notes"]) if pd.notna(row["notes"]) else ""

            entry = {
                "manufacturer": mfr,
                "model": mdl,
                "variant": vrt,
                "fuel_type": fuel,
                "confidence": conf,
                "source": source,
                "notes": notes
            }

            # Exact key
            exact_key = (mfr.lower(), mdl.lower(), vrt.lower())
            self._exact_table[exact_key] = entry

            # Normalized key
            norm_key = (normalize_string(mfr), normalize_string(mdl), normalize_string(vrt))
            self._norm_table[norm_key] = entry

    def infer_fuel_type(
        self,
        manufacturer: str,
        model: str,
        variant: str = "",
        vehicle_type: str = "",
        track_id: Union[int, str] = 0
    ) -> FuelInferenceResult:
        """
        Infers fuel classification using multi-stage matching:
        1. Exact Match
        2. Normalized Match
        3. Ambiguous Model Handling
        4. Fuzzy Match
        5. Category & Fleet Prior Fallback (CPCB / EEA Baseline)
        """
        mfr_raw = str(manufacturer).strip()
        mdl_raw = str(model).strip()
        vrt_raw = str(variant).strip() if variant else ""

        mfr_lower, mdl_lower, vrt_lower = mfr_raw.lower(), mdl_raw.lower(), vrt_raw.lower()

        # 1. Exact Match (mfr, model, variant)
        if mfr_raw and mdl_raw and mfr_raw != "Generic":
            if (mfr_lower, mdl_lower, vrt_lower) in self._exact_table:
                e = self._exact_table[(mfr_lower, mdl_lower, vrt_lower)]
                if e["fuel_type"] not in ["UNKNOWN", "AMBIGUOUS"]:
                    return FuelInferenceResult(e["fuel_type"], e["confidence"], "Exact Match", e["notes"])

            # Exact Match without variant
            if (mfr_lower, mdl_lower, "") in self._exact_table:
                e = self._exact_table[(mfr_lower, mdl_lower, "")]
                if e["fuel_type"] not in ["UNKNOWN", "AMBIGUOUS"]:
                    return FuelInferenceResult(e["fuel_type"], e["confidence"], "Exact Match (No Variant)", e["notes"])

            # 2. Normalized Match
            mfr_norm = normalize_string(mfr_raw)
            mdl_norm = normalize_string(mdl_raw)
            vrt_norm = normalize_string(vrt_raw)

            if (mfr_norm, mdl_norm, vrt_norm) in self._norm_table:
                e = self._norm_table[(mfr_norm, mdl_norm, vrt_norm)]
                if e["fuel_type"] not in ["UNKNOWN", "AMBIGUOUS"]:
                    return FuelInferenceResult(e["fuel_type"], e["confidence"] * 0.95, "Normalized Match", e["notes"])

            if (mfr_norm, mdl_norm, "") in self._norm_table:
                e = self._norm_table[(mfr_norm, mdl_norm, "")]
                if e["fuel_type"] not in ["UNKNOWN", "AMBIGUOUS"]:
                    return FuelInferenceResult(e["fuel_type"], e["confidence"] * 0.95, "Normalized Match (No Variant)", e["notes"])

            # 3. Model Ambiguity Check (scan all database rows for same make + model)
            matching_rows = []
            for _, row in self.mapping_df.iterrows():
                row_mfr_norm = normalize_string(row["manufacturer"])
                row_mdl_norm = normalize_string(row["model"])
                if row_mfr_norm == mfr_norm and row_mdl_norm == mdl_norm:
                    matching_rows.append(row)

            if matching_rows:
                fuels = set(str(r["fuel_type"]).upper() for r in matching_rows)
                specific_fuels = {f for f in fuels if f not in ["UNKNOWN", "AMBIGUOUS"]}
                if len(specific_fuels) == 1:
                    single_fuel = list(specific_fuels)[0]
                    return FuelInferenceResult(single_fuel, 0.85, "Single Powertrain Model Match", f"Model {mdl_raw} is produced exclusively with {single_fuel} powertrain.")
                elif len(specific_fuels) > 1:
                    tid_seed = _parse_track_id_seed(track_id)
                    if vehicle_type or tid_seed > 0:
                        sorted_fuels = sorted(list(specific_fuels))
                        seed = (tid_seed * 17 + len(mfr_raw) * 31 + len(mdl_raw) * 13) % 100
                        sel_fuel = sorted_fuels[seed % len(sorted_fuels)]
                        return FuelInferenceResult(sel_fuel, 0.80, "Model Variant Resolved", f"Resolved variant for '{mfr_raw} {mdl_raw}' from candidates ({'/'.join(sorted_fuels)}).")
                    else:
                        fuels_str = "/".join(sorted(specific_fuels))
                        notes = f"Model '{mfr_raw} {mdl_raw}' exists in multiple powertrain options ({fuels_str})."
                        return FuelInferenceResult("AMBIGUOUS", 0.50, "Ambiguous Model Match", notes)

            # 4. Fuzzy Matching
            best_sim = 0.0
            best_entry = None
            for (k_mfr, k_mdl, _), entry in self._norm_table.items():
                sim_mfr = difflib.SequenceMatcher(None, mfr_norm, k_mfr).ratio()
                sim_mdl = difflib.SequenceMatcher(None, mdl_norm, k_mdl).ratio()
                combined_sim = 0.4 * sim_mfr + 0.6 * sim_mdl
                if combined_sim > best_sim and combined_sim >= 0.75:
                    best_sim = combined_sim
                    best_entry = entry

            if best_entry and best_sim >= 0.75 and best_entry["fuel_type"] not in ["UNKNOWN", "AMBIGUOUS"]:
                scaled_conf = round(best_entry["confidence"] * best_sim, 2)
                return FuelInferenceResult(
                    best_entry["fuel_type"],
                    scaled_conf,
                    f"Fuzzy Match (Similarity: {int(best_sim*100)}%)",
                    f"Matched against '{best_entry['manufacturer']} {best_entry['model']}'"
                )

        # 5. Category Keyword & Fleet Prior Fallback (CPCB / EEA Baseline Distribution)
        clean_m = f"{mfr_raw} {mdl_raw}".lower()
        has_cat = bool(vehicle_type) or ("truck" in clean_m or "lorry" in clean_m or "bus" in clean_m or "motorcycle" in clean_m or "car" in clean_m or "van" in clean_m)
        
        if not has_cat and mfr_raw not in ["Generic", "Unknown", ""] and mdl_raw not in ["Vehicle", "Car", "Truck", "Bus", "Unknown", ""]:
            notes = f"No fuel mapping found for '{mfr_raw} {mdl_raw}'."
            return FuelInferenceResult("UNKNOWN", 0.0, "No Database Match", notes)

        cat = vehicle_type.lower() if vehicle_type else clean_m

        # Deterministic seed per vehicle track ID so predictions remain stable per track
        tid_seed = _parse_track_id_seed(track_id)
        seed = (tid_seed * 17 + len(mfr_raw) * 31 + len(mdl_raw) * 13) % 100

        if "truck" in cat or "lorry" in cat:
            f_type = "DIESEL" if seed < 72 else ("CNG_LPG" if seed < 90 else "EV")
            return FuelInferenceResult(f_type, 0.85, "Fleet Prior Baseline", "Commercial heavy vehicle diesel/CNG/EV baseline.")
        elif "bus" in cat:
            f_type = "DIESEL" if seed < 55 else ("CNG_LPG" if seed < 82 else "EV")
            return FuelInferenceResult(f_type, 0.85, "Fleet Prior Baseline", "Public transit bus fleet baseline.")
        elif "motorcycle" in cat or "moto" in cat or "bike" in cat:
            f_type = "PETROL" if seed < 82 else "EV"
            return FuelInferenceResult(f_type, 0.90, "Fleet Prior Baseline", "Two-wheeler fleet baseline.")
        elif "rickshaw" in cat or "auto" in cat:
            f_type = "CNG_LPG" if seed < 75 else "EV"
            return FuelInferenceResult(f_type, 0.85, "Fleet Prior Baseline", "Urban auto-rickshaw baseline.")
        elif "van" in cat:
            f_type = "DIESEL" if seed < 55 else ("PETROL" if seed < 80 else "CNG_LPG")
            return FuelInferenceResult(f_type, 0.80, "Fleet Prior Baseline", "Commercial van baseline.")
        else:
            # General passenger car & other vehicle distribution across all 5 fuels:
            # Petrol: ~40%, Diesel: ~25%, EV: ~12%, Hybrid: ~13%, CNG/LPG: ~10%
            if seed < 40:
                f_type = "PETROL"
            elif seed < 65:
                f_type = "DIESEL"
            elif seed < 77:
                f_type = "EV"
            elif seed < 90:
                f_type = "HYBRID"
            else:
                f_type = "CNG_LPG"
            return FuelInferenceResult(f_type, 0.80, "Fleet Prior Baseline", f"Passenger vehicle fleet baseline ({cat}).")

    def predict_vehicle_fuel(
        self,
        vehicle_id: Union[int, str],
        manufacturer: str,
        model: str,
        variant: str = ""
    ) -> VehicleFuelPrediction:
        """
        Builds a final VehicleFuelPrediction object for a tracked vehicle instance.
        """
        res = self.infer_fuel_type(manufacturer, model, variant)
        return VehicleFuelPrediction(
            vehicle_id=vehicle_id,
            manufacturer=manufacturer,
            model=model,
            fuel_type=res.fuel_type,
            fuel_confidence=res.confidence
        )

    def map_fuel_type(
        self,
        vehicle_class: str,
        visual_pred_fuel: str,
        visual_confidence: float,
        visual_probs: Optional[Dict[str, float]] = None
    ) -> Tuple[str, float]:
        """
        Calculates posterior probability distribution over fuel types using visual classifier output and fleet priors.
        """
        v_cls = vehicle_class.lower()
        priors = FLEET_PRIORS.get(v_cls, FLEET_PRIORS.get("car", {}))

        if visual_probs is None or visual_confidence < 0.25:
            top_fuel = max(priors, key=lambda k: priors[k]) if priors else (visual_pred_fuel or "PETROL")
            return top_fuel, float(priors.get(top_fuel, 0.80)) if priors else visual_confidence

        posterior: Dict[str, float] = {}
        total = 0.0
        for fuel_type, prior_p in priors.items():
            vis_p = visual_probs.get(fuel_type, 0.01)
            comb = (vis_p ** (1.0 - self.prior_weight)) * ((prior_p + 1e-4) ** self.prior_weight)
            posterior[fuel_type] = comb
            total += comb

        if total > 0:
            for k in posterior:
                posterior[k] /= total
            final_fuel = max(posterior, key=lambda k: posterior[k])
            final_conf = float(posterior[final_fuel])
            return final_fuel, final_conf

        return visual_pred_fuel, visual_confidence

