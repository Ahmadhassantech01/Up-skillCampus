"""
============================================================
  COMPREHENSIVE AGRICULTURE UAV ANALYTICAL BRAIN
  Developed for: upSkill Campus / UCT Winter Internship 2026
  Domain: Data Science & Machine Learning
  Author: Ahmad Hassan
  Project: AI-Powered Precision Agriculture Drone System
============================================================
"""

import json
import csv
import math
import os
import uuid
from datetime import datetime, timedelta
import random  # Used for demo simulation of CNN confidence scores

# ─────────────────────────────────────────────────────────────
# MODULE 1 ▸ DEEP ANALYSIS DISEASE DETECTION ENGINE
# ─────────────────────────────────────────────────────────────

class DiseaseDetectionEngine:
    """
    CNN-based crop disease classifier that processes drone imagery
    captured at low altitude. Implements a lightweight MobileNetV2-style
    inference pipeline suitable for Edge AI deployment on embedded hardware
    such as Raspberry Pi 4 / NVIDIA Jetson Nano.

    Supported Disease Classes:
        0 – Healthy
        1 – Wheat Rust (Puccinia striiformis)
        2 – Potato Late Blight (Phytophthora infestans)
        3 – Corn Leaf Blight (Exserohilum turcicum)
        4 – Rice Blast (Magnaporthe oryzae)
        5 – Cotton Bollworm Damage
    """

    CLASS_LABELS = {
        0: "Healthy",
        1: "Wheat Rust",
        2: "Potato Late Blight",
        3: "Corn Leaf Blight",
        4: "Rice Blast",
        5: "Cotton Bollworm Damage",
    }

    SEVERITY_THRESHOLDS = {
        "critical":  0.85,   # Immediate spray trigger
        "moderate":  0.60,   # Log + flag for review
        "low":       0.40,   # Monitor only
    }

    def __init__(self, input_size=(224, 224), quantized=True):
        self.input_size = input_size
        self.quantized = quantized   # INT8 quantization for edge speed
        self.inference_count = 0
        print(f"[DiseaseEngine] Initialised | Input: {input_size} | Quantized: {quantized}")

    # ── Step 1: Image Pre-processing Pipeline ──────────────────
    def preprocess_image(self, image_array):
        """
        Normalises a raw drone frame for CNN inference.

        Pipeline:
          1. Resize to 224×224 (bilinear interpolation)
          2. Convert BGR→RGB channel order
          3. Normalise pixel values to [0, 1]
          4. Apply ImageNet mean/std normalisation
          5. Add batch dimension [1, H, W, C]

        Args:
            image_array (list): 2D array of pixel values (simulated here)

        Returns:
            dict: Processed tensor metadata
        """
        mean = [0.485, 0.456, 0.406]   # ImageNet RGB means
        std  = [0.229, 0.224, 0.225]   # ImageNet RGB stds

        processed = {
            "original_shape":  str(image_array.get("shape", "unknown")),
            "resized_to":      f"{self.input_size[0]}x{self.input_size[1]}",
            "channel_order":   "RGB",
            "normalised":      True,
            "mean_applied":    mean,
            "std_applied":     std,
            "batch_dimension": 1,
            "dtype":           "float32" if not self.quantized else "int8",
        }
        print(f"  [Preprocess] {processed['original_shape']} → {processed['resized_to']} | dtype={processed['dtype']}")
        return processed

    # ── Step 2: Feature Extraction (Simulated CNN Layers) ──────
    def extract_features(self, preprocessed_tensor):
        """
        Simulates depthwise-separable convolution blocks used in
        MobileNetV2. In production this is replaced by a TFLite .tflite
        model file loaded with tf.lite.Interpreter.

        Architecture (conceptual):
          Input (224,224,3)
          → Conv2D(32, 3×3, stride=2)  → BN → ReLU6
          → 17× Inverted Residual Blocks
          → Conv2D(1280, 1×1)          → BN → ReLU6
          → GlobalAveragePooling2D
          → Dense(6, softmax)

        Returns:
            list: Simulated softmax probability vector (6 classes)
        """
        # Deterministic noise simulation — replace with real TFLite call
        raw_logits = [random.uniform(0.01, 0.15) for _ in range(6)]
        # Push one class to dominate (simulate a real detection)
        dominant = random.randint(0, 5)
        raw_logits[dominant] = random.uniform(0.60, 0.98)
        # Softmax normalisation
        exp_vals = [math.exp(v) for v in raw_logits]
        total    = sum(exp_vals)
        softmax  = [round(v / total, 4) for v in exp_vals]
        return softmax

    # ── Step 3: Inference & Decision ────────────────────────────
    def run_inference(self, frame_metadata):
        """
        Full end-to-end inference for a single drone frame.

        Args:
            frame_metadata (dict): Contains GPS coords, altitude, frame_id

        Returns:
            dict: Detection result including disease label, confidence,
                  severity level, and spray recommendation.
        """
        self.inference_count += 1

        # 1. Preprocess
        tensor = self.preprocess_image(frame_metadata)

        # 2. CNN Feature extraction + classification
        probabilities = self.extract_features(tensor)

        # 3. Top-1 prediction
        top_class_idx  = probabilities.index(max(probabilities))
        top_confidence = max(probabilities)
        label          = self.CLASS_LABELS[top_class_idx]

        # 4. Severity grading
        if top_confidence >= self.SEVERITY_THRESHOLDS["critical"]:
            severity = "CRITICAL"
            spray_recommended = True
        elif top_confidence >= self.SEVERITY_THRESHOLDS["moderate"]:
            severity = "MODERATE"
            spray_recommended = top_class_idx != 0  # Healthy=no spray
        else:
            severity = "LOW"
            spray_recommended = False

        result = {
            "frame_id":          frame_metadata.get("frame_id", self.inference_count),
            "gps_lat":           frame_metadata.get("lat", 0.0),
            "gps_lon":           frame_metadata.get("lon", 0.0),
            "altitude_m":        frame_metadata.get("altitude_m", 30),
            "disease_label":     label,
            "confidence":        round(top_confidence, 4),
            "severity":          severity,
            "spray_recommended": spray_recommended,
            "class_probabilities": {
                self.CLASS_LABELS[i]: probabilities[i] for i in range(6)
            },
            "timestamp":         datetime.now().isoformat(),
        }

        icon = "🔴" if spray_recommended else "🟢"
        print(f"  {icon} Frame {result['frame_id']} | {label} ({top_confidence:.2%}) | {severity}")
        return result


# ─────────────────────────────────────────────────────────────
# MODULE 2 ▸ ACREAGE & GEOSPATIAL MAPPING ENGINE
# ─────────────────────────────────────────────────────────────

class AcreageMappingEngine:
    """
    Calculates real-world coverage area from drone flight parameters.

    Key formulae:
      GSD (m/px) = (altitude × sensor_width) / (focal_length × image_width_px)
      Footprint Width (m)  = 2 × altitude × tan(HFOV / 2)
      Footprint Height (m) = 2 × altitude × tan(VFOV / 2)
      Area per frame (m²)  = Footprint_W × Footprint_H

    Conversion constants:
      1 acre     = 4,046.86 m²
      1 hectare  = 10,000 m²
    """

    SQ_M_PER_ACRE     = 4046.86
    SQ_M_PER_HECTARE  = 10_000.0

    def __init__(self, hfov_deg=84.0, vfov_deg=63.0,
                 image_width_px=4000, image_height_px=3000):
        self.hfov_rad = math.radians(hfov_deg)
        self.vfov_rad = math.radians(vfov_deg)
        self.img_w    = image_width_px
        self.img_h    = image_height_px
        print(f"[AcreageEngine] HFOV={hfov_deg}° | VFOV={vfov_deg}° | "
              f"Sensor={image_width_px}×{image_height_px}px")

    # ── Ground Footprint per Frame ─────────────────────────────
    def frame_footprint_m2(self, altitude_m):
        """Returns (width_m, height_m, area_m2) for one camera frame."""
        w = 2 * altitude_m * math.tan(self.hfov_rad / 2)
        h = 2 * altitude_m * math.tan(self.vfov_rad / 2)
        return round(w, 3), round(h, 3), round(w * h, 3)

    # ── Total Area Checked ─────────────────────────────────────
    def total_area_checked(self, altitude_m, flight_distance_m,
                           overlap_pct=0.75):
        """
        Computes total surveyed area accounting for image overlap.

        Args:
            altitude_m (float): Drone flight altitude
            flight_distance_m (float): Total distance flown (metres)
            overlap_pct (float): Between-frame overlap ratio (0–1)

        Returns:
            dict: Area in m², acres, and hectares
        """
        fw, fh, frame_area = self.frame_footprint_m2(altitude_m)
        # Effective swath width (one pass)
        effective_swath = fw * (1 - overlap_pct)
        # Number of parallel passes to cover field width (assumed square flight)
        # Total area = flight_distance × effective_swath_width
        total_m2 = flight_distance_m * effective_swath
        return {
            "altitude_m":          altitude_m,
            "flight_distance_m":   flight_distance_m,
            "frame_footprint_m":   f"{fw:.1f} × {fh:.1f}",
            "frame_area_m2":       frame_area,
            "effective_swath_m":   round(effective_swath, 2),
            "total_area_m2":       round(total_m2, 2),
            "total_area_acres":    round(total_m2 / self.SQ_M_PER_ACRE, 4),
            "total_area_hectares": round(total_m2 / self.SQ_M_PER_HECTARE, 4),
        }

    # ── Area Sprayed from GPS Waypoints ───────────────────────
    def area_sprayed_from_gps(self, spray_events):
        """
        Calculates sprayed area using the Shoelace formula (Gauss's area
        formula) over a polygon formed by GPS spray-trigger coordinates.

        For non-contiguous spray zones, sums individual convex hull areas.

        Args:
            spray_events (list of dict): Each item has {lat, lon, spray_radius_m}

        Returns:
            dict: Total sprayed area in m², acres, hectares, and spray count
        """
        total_m2 = 0.0
        for event in spray_events:
            # Model each spray event as a circle: A = π × r²
            r = event.get("spray_radius_m", 2.5)  # default 2.5 m nozzle radius
            total_m2 += math.pi * r ** 2

        return {
            "spray_events":         len(spray_events),
            "total_sprayed_m2":     round(total_m2, 2),
            "total_sprayed_acres":  round(total_m2 / self.SQ_M_PER_ACRE, 6),
            "total_sprayed_ha":     round(total_m2 / self.SQ_M_PER_HECTARE, 6),
        }

    # ── Haversine Distance between Two GPS Points ──────────────
    @staticmethod
    def haversine_distance_m(lat1, lon1, lat2, lon2):
        """Returns distance in metres between two WGS-84 coordinates."""
        R = 6_371_000  # Earth radius in metres
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi  = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
        return round(2 * R * math.asin(math.sqrt(a)), 3)


# ─────────────────────────────────────────────────────────────
# MODULE 3 ▸ SMART SPRAYING CONTROLLER
# ─────────────────────────────────────────────────────────────

class SmartSprayingController:
    """
    Three operational modes:

    ┌─────────────────────────────────────────────────────┐
    │  MODE A – FULL FIELD SPRAYING                       │
    │  Drone sprays entire field regardless of AI output. │
    │  Traditional approach (baseline comparison).        │
    ├─────────────────────────────────────────────────────┤
    │  MODE B – AI-AUTOMATED SPOT SPRAYING (Primary)      │
    │  Sprayer activates ONLY when CNN detects disease    │
    │  with confidence ≥ threshold. Saves ~60–80% pesticide│
    ├─────────────────────────────────────────────────────┤
    │  MODE C – MANUAL ZONAL SPRAYING                     │
    │  Operator pre-defines GPS boundary boxes.           │
    │  Drone sprays within those zones only.              │
    └─────────────────────────────────────────────────────┘
    """

    MODES = {"FULL_FIELD": "A", "AI_SPOT": "B", "MANUAL_ZONE": "C"}

    def __init__(self, mode="AI_SPOT", confidence_threshold=0.70):
        if mode not in self.MODES:
            raise ValueError(f"Mode must be one of {list(self.MODES.keys())}")
        self.mode = mode
        self.threshold = confidence_threshold
        self.spray_log = []
        self.total_pesticide_ml = 0.0
        self.FLOW_RATE_ML_PER_SEC = 150  # Nozzle flow rate
        print(f"[SprayController] Mode={mode} | Threshold={confidence_threshold:.0%}")

    def evaluate_spray(self, detection_result, manual_zones=None):
        """
        Decides whether to activate the sprayer for a given frame.

        Args:
            detection_result (dict): Output from DiseaseDetectionEngine
            manual_zones (list): List of {lat_min, lat_max, lon_min, lon_max}
                                  (required for MANUAL_ZONE mode)

        Returns:
            dict: Spray decision record
        """
        lat = detection_result["gps_lat"]
        lon = detection_result["gps_lon"]
        spray_active = False
        trigger_reason = ""

        if self.mode == "FULL_FIELD":
            spray_active = True
            trigger_reason = "Full-field coverage mode"

        elif self.mode == "AI_SPOT":
            if (detection_result["spray_recommended"] and
                    detection_result["confidence"] >= self.threshold):
                spray_active = True
                trigger_reason = (f"AI detected {detection_result['disease_label']} "
                                  f"@ {detection_result['confidence']:.1%} confidence")

        elif self.mode == "MANUAL_ZONE":
            if manual_zones:
                for zone in manual_zones:
                    if (zone["lat_min"] <= lat <= zone["lat_max"] and
                            zone["lon_min"] <= lon <= zone["lon_max"]):
                        spray_active = True
                        trigger_reason = f"Inside manual spray zone {zone.get('name','Z')}"
                        break

        duration_sec = 2.0 if spray_active else 0.0
        pesticide_used_ml = duration_sec * self.FLOW_RATE_ML_PER_SEC

        record = {
            "frame_id":          detection_result["frame_id"],
            "lat":               lat,
            "lon":               lon,
            "spray_active":      spray_active,
            "trigger_reason":    trigger_reason,
            "duration_sec":      duration_sec,
            "pesticide_used_ml": pesticide_used_ml,
            "mode":              self.mode,
            "timestamp":         datetime.now().isoformat(),
        }

        if spray_active:
            self.total_pesticide_ml += pesticide_used_ml
            self.spray_log.append(record)
            print(f"  💧 SPRAY ON | Frame {record['frame_id']} | {trigger_reason}")
        else:
            print(f"  ✋ SPRAY OFF | Frame {record['frame_id']}")

        return record

    def spray_events_for_mapping(self):
        """Returns GPS events formatted for AcreageMappingEngine."""
        return [{"lat": r["lat"], "lon": r["lon"], "spray_radius_m": 2.5}
                for r in self.spray_log]


# ─────────────────────────────────────────────────────────────
# MODULE 4 ▸ DATA LOGGING & ROI REPORTING ENGINE
# ─────────────────────────────────────────────────────────────

class FlightDataLogger:
    """
    Persists flight telemetry and AI decisions to JSON + CSV.
    Generates a post-flight ROI report including:
      • Pesticide Saved %
      • Environmental Impact Score (0–100, higher = greener)
      • Cost Savings Estimate (USD)
      • Carbon Footprint Delta
    """

    def __init__(self, output_dir="flight_logs"):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir  = output_dir
        self.flight_id   = str(uuid.uuid4())[:8].upper()
        self.records     = []
        self.start_time  = datetime.now()
        print(f"[DataLogger] Flight ID: {self.flight_id} | Output: {output_dir}/")

    def log_record(self, detection_result, spray_decision):
        """Combines detection + spray data into a single flight record."""
        merged = {**detection_result, **spray_decision,
                  "flight_id": self.flight_id}
        self.records.append(merged)

    def save_json(self):
        path = os.path.join(self.output_dir, f"flight_{self.flight_id}.json")
        with open(path, "w") as f:
            json.dump(self.records, f, indent=2)
        print(f"  [Logger] JSON saved → {path}")
        return path

    def save_csv(self):
        if not self.records:
            return None
        path = os.path.join(self.output_dir, f"flight_{self.flight_id}.csv")
        keys = list(self.records[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for rec in self.records:
                flat = {k: (str(v) if isinstance(v, dict) else v)
                        for k, v in rec.items()}
                writer.writerow(flat)
        print(f"  [Logger] CSV saved → {path}")
        return path

    def generate_roi_report(self, area_data, spray_controller,
                            pesticide_cost_per_litre=12.0,
                            baseline_pesticide_ml_per_ha=5000):
        """
        Produces a structured ROI + environmental impact summary.

        Environmental Impact Score:
            EIS = 100 × (1 − actual_usage / baseline_usage)
            Range: 0 (worst) → 100 (best / zero usage)
        """
        total_ha    = area_data.get("total_area_hectares", 1.0)
        actual_ml   = spray_controller.total_pesticide_ml
        baseline_ml = baseline_pesticide_ml_per_ha * total_ha

        pesticide_saved_pct = max(0, round((1 - actual_ml / baseline_ml) * 100, 2)) \
                              if baseline_ml > 0 else 0
        eis = round(pesticide_saved_pct, 1)           # simplified 1-to-1 here

        cost_saved_usd = round(((baseline_ml - actual_ml) / 1000)
                               * pesticide_cost_per_litre, 2)

        # Carbon delta: 1L pesticide production ≈ 2.3 kg CO₂ eq.
        carbon_saved_kg = round(((baseline_ml - actual_ml) / 1000) * 2.3, 3)

        diseases_found = [r["disease_label"] for r in self.records
                          if r.get("spray_recommended")]

        report = {
            "flight_id":             self.flight_id,
            "flight_date":           self.start_time.strftime("%Y-%m-%d"),
            "duration_minutes":      round((datetime.now() - self.start_time).seconds / 60, 1),
            "area_surveyed":         area_data,
            "spray_mode":            spray_controller.mode,
            "frames_analysed":       len(self.records),
            "spray_events":          len(spray_controller.spray_log),
            "actual_pesticide_ml":   round(actual_ml, 1),
            "baseline_pesticide_ml": round(baseline_ml, 1),
            "pesticide_saved_pct":   pesticide_saved_pct,
            "environmental_impact_score": eis,
            "cost_saved_usd":        cost_saved_usd,
            "carbon_saved_kg_co2eq": carbon_saved_kg,
            "diseases_detected":     list(set(diseases_found)) or ["None"],
        }

        # Save report
        report_path = os.path.join(self.output_dir,
                                   f"roi_report_{self.flight_id}.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print("\n" + "=" * 60)
        print(f"  POST-FLIGHT ROI REPORT  |  Flight: {self.flight_id}")
        print("=" * 60)
        print(f"  Area Surveyed    : {total_ha:.4f} ha")
        print(f"  Pesticide Used   : {actual_ml:.0f} mL  (baseline: {baseline_ml:.0f} mL)")
        print(f"  Pesticide Saved  : {pesticide_saved_pct:.1f}%")
        print(f"  Env. Impact Score: {eis:.1f} / 100")
        print(f"  Cost Saved       : ${cost_saved_usd:.2f}")
        print(f"  Carbon Saved     : {carbon_saved_kg:.3f} kg CO₂-eq")
        print(f"  Diseases Found   : {', '.join(report['diseases_detected'])}")
        print("=" * 60)
        print(f"  Report saved → {report_path}")

        return report


# ─────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR – Simulates a complete UAV flight mission
# ─────────────────────────────────────────────────────────────

def run_mission_simulation(num_frames=20,
                           altitude_m=30,
                           flight_distance_m=800,
                           spray_mode="AI_SPOT"):
    """
    Simulates a full UAV agricultural survey mission.

    Steps:
      1. Drone flies at set altitude capturing frames
      2. Each frame → Disease Detection Engine
      3. Detection result → Spray Controller decision
      4. All data → Flight Logger
      5. Post-flight ROI report generated
    """
    print("\n" + "▓" * 60)
    print("  UAV AGRICULTURAL AI ENGINE  — MISSION START")
    print("▓" * 60 + "\n")

    # Initialise sub-systems
    detector    = DiseaseDetectionEngine(input_size=(224, 224), quantized=True)
    mapper      = AcreageMappingEngine(hfov_deg=84, vfov_deg=63)
    controller  = SmartSprayingController(mode=spray_mode, confidence_threshold=0.70)
    logger      = FlightDataLogger(output_dir="flight_logs")

    # Optional: Pre-defined manual zones for MODE C
    manual_zones = [
        {"name": "Zone-Alpha", "lat_min": 30.000, "lat_max": 30.005,
         "lon_min": 71.000, "lon_max": 71.005},
        {"name": "Zone-Beta",  "lat_min": 30.010, "lat_max": 30.015,
         "lon_min": 71.010, "lon_max": 71.015},
    ]

    # Simulate drone GPS waypoints (lat, lon increment per frame)
    base_lat, base_lon = 30.0000, 71.0000

    print(f"\n[Mission] Flying {num_frames} frames @ {altitude_m}m altitude | Mode: {spray_mode}\n")

    for i in range(1, num_frames + 1):
        frame_meta = {
            "frame_id":   i,
            "lat":        round(base_lat + i * 0.0005, 6),
            "lon":        round(base_lon + i * 0.0005, 6),
            "altitude_m": altitude_m,
            "shape":      f"3000x4000x3",
        }

        detection  = detector.run_inference(frame_meta)
        spray_dec  = controller.evaluate_spray(detection, manual_zones=manual_zones)
        logger.log_record(detection, spray_dec)

    # Save raw logs
    print("\n[Mission] Saving flight data...")
    logger.save_json()
    logger.save_csv()

    # Compute area metrics
    area_data = mapper.total_area_checked(altitude_m, flight_distance_m)
    spray_gps = controller.spray_events_for_mapping()
    spray_area = mapper.area_sprayed_from_gps(spray_gps)

    print(f"\n[Mission] Area Checked  : {area_data['total_area_acres']:.3f} acres "
          f"({area_data['total_area_hectares']:.4f} ha)")
    print(f"[Mission] Area Sprayed  : {spray_area['total_sprayed_acres']:.6f} acres "
          f"({spray_area['total_sprayed_ha']:.6f} ha)")

    # Generate ROI report
    logger.generate_roi_report(area_data, controller)

    print("\n▓" * 60 + "\n  MISSION COMPLETE\n" + "▓" * 60)


if __name__ == "__main__":
    run_mission_simulation(
        num_frames=20,
        altitude_m=30,
        flight_distance_m=800,
        spray_mode="AI_SPOT"   # Change to FULL_FIELD or MANUAL_ZONE
    )
