import json, cv2, os
from pathlib import Path

# === Config ===
JSON_PATH = "/home/coder/thesis-yolo/ocr_out/evac_v12_full_fused_v3/parsed_summary.json"
IMAGE_DIR = "/home/coder/thesis-yolo/ocr_out/evac_v12_full_fused_v3/images"
OUT_DIR = "/home/coder/thesis-yolo/ocr_out/evac_v12_full_fused_v3/visualized"
os.makedirs(OUT_DIR, exist_ok=True)

# === Load JSON ===
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# === Visualization loop ===
for image_name, detections in data.items():
    img_path = os.path.join(IMAGE_DIR, image_name)
    if not os.path.exists(img_path):
        print(f"⚠️ Missing image: {image_name}")
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Could not read: {image_name}")
        continue

    for det in detections:
        cls = det["class"]
        conf = det.get("confidence", 1.0)
        label = det.get("text", "no_text")
        x, y, w, h = 30, 30, 120, 50  # Dummy placeholder box
        color = (0, 255, 0) if label.lower() != "no_text" else (0, 165, 255)

        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, f"{cls} | {label} | {conf:.2f}",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    out_path = os.path.join(OUT_DIR, image_name)
    cv2.imwrite(out_path, img)
    print(f"✅ Saved: {out_path}")
