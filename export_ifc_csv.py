import json
import csv

# === INPUT / OUTPUT PATHS ===
input_json_path = "ocr_out/evac_v12_full_fused_v4/graph_output.json"
output_csv_path = "ocr_out/evac_v12_full_fused_v4/ifc_export.csv"

# === LOAD JSON ===
with open(input_json_path, "r") as f:
    data = json.load(f)

print("Loaded data type:", type(data))

# === HANDLE CASE: dict with 'nodes' key ===
nodes = []
if isinstance(data, dict) and "nodes" in data:
    nodes = data["nodes"]
elif isinstance(data, list):
    # In case you have a list of graphs
    for g in data:
        if "nodes" in g:
            nodes.extend(g["nodes"])
else:
    raise ValueError("❌ Unexpected JSON structure — no 'nodes' found.")

print(f"Found {len(nodes)} nodes to export.")

# === WRITE TO CSV ===
with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["id", "image", "ifc_class", "ifc_type", "ocr_text", "confidence"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for node in nodes:
        writer.writerow({
            "id": node.get("id", ""),
            "image": node.get("image", ""),
            "ifc_class": node.get("ifc_class", ""),
            "ifc_type": node.get("ifc_type", ""),
            "ocr_text": node.get("ocr_text", ""),
            "confidence": node.get("confidence", "")
        })

print(f"✅ IFC export CSV successfully saved to: {output_csv_path}")
