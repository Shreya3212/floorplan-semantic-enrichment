import json, re, os, pandas as pd
from collections import defaultdict

JSON_PATH = "ocr_out/evac_v12_full_fused_v3/parsed_summary.json"
OUT_CSV = "ocr_out/evac_v12_full_fused_v3/enrichment.csv"
OUT_JSON = "ocr_out/evac_v12_full_fused_v3/enrichment.json"

det_map = {
    "exit": ("IfcDoor", "EXIT"),
    "stairs": ("IfcStair", "STAIR"),
    "fire extinguisher": ("IfcFireSuppressionTerminal", "FIREEXTINGUISHER"),
    "hydrant": ("IfcFireSuppressionTerminal", "HYDRANT"),
}

vocab = {
    "EXIT": ["EXIT", "AUSGANG", "NOTAUSGANG", "WAY OUT", "FLUCHTWEG"],
    "STAIRS": ["STAIR", "STAIRS", "TREP", "ST."],
    "EXTINGUISHER": ["FIRE EXTINGUISHER", "EXTINGUISHER", "FEUERLÖSCHER", "FEUERLOESCHER"],
    "HYDRANT": ["HYDRANT", "WANDHYDRANT"]
}

def norm_text(t):
    t = str(t or "").upper()
    t = re.sub(r"[^A-Z0-9/→\\-\\s]", "", t)
    for k, alts in vocab.items():
        if any(a in t for a in alts):
            return k
    return ""

with open(JSON_PATH, "r", encoding="utf-8") as f:
    parsed = json.load(f)

out_rows = []
for img, detections in parsed.items():
    for det in detections:
        cls = det.get("class", "").lower()
        conf = float(det.get("confidence", 0))
        text = det.get("text", "")
        if cls not in det_map:
            continue
        ifc_class, ifc_type = det_map[cls]
        ocr_norm = norm_text(text)

        chosen_class, chosen_type = ifc_class, ifc_type
        conflict = False

        map_from_text = {
            "EXIT": ("IfcDoor", "EXIT"),
            "STAIRS": ("IfcStair", "STAIR"),
            "EXTINGUISHER": ("IfcFireSuppressionTerminal", "FIREEXTINGUISHER"),
            "HYDRANT": ("IfcFireSuppressionTerminal", "HYDRANT")
        }.get(ocr_norm)

        if map_from_text:
            if map_from_text != (ifc_class, ifc_type):
                if conf <= 0.45:
                    chosen_class, chosen_type = map_from_text
                else:
                    conflict = True

        score = conf
        if map_from_text and map_from_text == (ifc_class, ifc_type):
            score = min(1.0, score + 0.2)

        out_rows.append({
            "image": img,
            "ifc_class": chosen_class,
            "ifc_type": chosen_type,
            "confidence": round(score, 3),
            "det_class": cls,
            "ocr_text": text,
            "conflict": conflict,
        })

df = pd.DataFrame(out_rows)
df.to_csv(OUT_CSV, index=False)
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(out_rows, f, indent=2, ensure_ascii=False)

print(f"✅ Wrote {OUT_CSV} and {OUT_JSON} with {len(out_rows)} items.")
