import pandas as pd, json, re, os, sys
import argparse

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--json", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
args = parser.parse_args()

# Detection to IFC mapping
det_map = {
    "exit": ("IfcDoor", "EXIT"),
    "stairs": ("IfcStair", "STAIR"),
    "fire extinguisher": ("IfcFireSuppressionTerminal", "FIREEXTINGUISHER"),
    "hydrant": ("IfcFireSuppressionTerminal", "HYDRANT"),
}

# Text-based vocab keywords
vocab = {
    "EXIT": ["EXIT", "AUSGANG", "NOTAUSGANG", "WAY OUT", "FLUCHTWEG"],
    "STAIRS": ["STAIR", "STAIRS", "TREP", "ST."],
    "EXTINGUISHER": ["FIRE EXTINGUISHER", "EXTINGUISHER", "FEUERLÖSCHER", "FEUERLOESCHER"],
    "HYDRANT": ["HYDRANT", "WANDHYDRANT"]
}

def norm_text(t):
    t = str(t or "").upper()
    t = re.sub(r"[^A-Z0-9/→\-\s]", "", t)
    for k, alts in vocab.items():
        if any(a in t for a in alts):
            return k
    return ""

rows = []

# Read detection + OCR fused JSON
with open(args.json, "r") as f:
    data = json.load(f)

for image_name, objects in data.items():
    for r in objects:
        det_cls = str(r.get("cls", "")).lower()
        det_conf = float(r.get("conf", 0))
        ocr_t = r.get("ocr_text") or r.get("ocr_clean") or r.get("ocr_raw") or ""
        ocr_conf = 1.0 if ocr_t and ocr_t != "no_text" else 0.0
        if det_cls not in det_map:
            continue

        ifc_class, ifc_type = det_map[det_cls]
        ocr_norm = norm_text(ocr_t)
        chosen_class, chosen_type = ifc_class, ifc_type
        conflict = False

        # Try mapping from OCR text
        map_from_text = {
            "EXIT": ("IfcDoor", "EXIT"),
            "STAIRS": ("IfcStair", "STAIR"),
            "EXTINGUISHER": ("IfcFireSuppressionTerminal", "FIREEXTINGUISHER"),
            "HYDRANT": ("IfcFireSuppressionTerminal", "HYDRANT"),
        }.get(ocr_norm)

        if map_from_text:
            if map_from_text != (ifc_class, ifc_type):
                if ocr_conf >= 0.65 and det_conf <= 0.45:
                    chosen_class, chosen_type = map_from_text
                else:
                    conflict = True

        score = det_conf
        if map_from_text and map_from_text == (ifc_class, ifc_type):
            score = min(1.0, score + 0.2)

        rows.append({
            "image": image_name,
            "ifc_class": chosen_class,
            "ifc_type": chosen_type,
            "confidence": round(score, 3),
            "det_class": det_cls,
            "det_conf": round(det_conf, 3),
            "ocr_text": ocr_t,
            "ocr_conf": round(ocr_conf, 3),
            "conflict": conflict
        })

# Output writing
os.makedirs(args.out, exist_ok=True)
out_csv = os.path.join(args.out, "enrichment.csv")
out_json = os.path.join(args.out, "enrichment.json")
pd.DataFrame(rows).to_csv(out_csv, index=False)
with open(out_json, "w") as f:
    json.dump(rows, f, indent=2)

print("✅ Wrote:", out_csv, "and", out_json, "items:", len(rows))


