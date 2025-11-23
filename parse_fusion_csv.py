import csv 
import json
from collections import defaultdict

# Path to the fused CSV
FUSED_CSV_PATH = "ocr_out/evac_v12_full_fused_v3/summary.csv"
OUTPUT_JSON_PATH = "ocr_out/evac_v12_full_fused_v3/parsed_summary.json"

# Storage structure: {image_filename: [list of detections]}
parsed_data = defaultdict(list)

with open(FUSED_CSV_PATH, mode="r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        image_name = row["image"]
        detection = {
            "class": row.get("cls", "").strip(),
            "confidence": float(row.get("conf", "1.0")),
            "text": row.get("ocr_clean", "").strip() or row.get("ocr_raw", "").strip() or "no_text"
        }
        parsed_data[image_name].append(detection)

# Save to JSON for debugging
with open(OUTPUT_JSON_PATH, mode="w", encoding="utf-8") as out:
    json.dump(parsed_data, out, indent=2, ensure_ascii=False)

print(f"✅ Parsed {len(parsed_data)} images and saved to {OUTPUT_JSON_PATH}")
