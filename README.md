# Semantic Enrichment of Incomplete Floorplans with a Focus on Fire Exit Routes

This repository contains the implementation of my master's thesis at the Technical University of Munich (TUM): **"Semantic Enrichment of Incomplete Floorplans with a Focus on Fire Exit Routes"**.

The pipeline detects fire-safety-related symbols in 2D evacuation/floorplans using **YOLOv8** and **EasyOCR**, fuses detections and text, maps them to **IFC entities**, and builds a graph-based representation for rule-based reasoning on fire exit routes. :contentReference[oaicite:0]{index=0}

---

## 1. Repository Structure

> Note: You can adjust this list to match your final file layout.

- `EvacuationPlanDataset/` – sample evacuation plan images and labels (if included)
- `out/` – example outputs (overlays, fused CSVs, enrichment, graphs, rule-based results)
- `runs/` – YOLO training / inference runs (optional to publish)
- `fuse_yolo_ocr.py` – fuse YOLO detections with OCR text and validate symbols
- `map_to_semantics.py` – map fused detections to IFC entity types and semantic labels
- `visualize_graph_json.py` – visualize the Graph-BIM representation
- `run_rule_based_checkup.py` – apply rule-based checks on the graph
- `summarise_rule_results.py` – summarize rule-check results
- `export_ifc_csv.py` – export enriched elements as CSV for IFC conversion
- `requirements.txt` – Python dependencies (YOLOv8, EasyOCR, pandas, shapely, etc.)

---

## 2. Setup

### 2.1. Environment

- Python **3.10+** recommended
- OS: Windows / Linux

Create and activate a virtual environment (optional but recommended):

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate


