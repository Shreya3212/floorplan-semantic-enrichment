# Semantic Enrichment of Incomplete Floorplans with a Focus on Fire Exit Routes

This repository contains the implementation of my master’s thesis at the Technical University of Munich (TUM):

> **“Semantic Enrichment of Incomplete Floorplans with a Focus on Fire Exit Routes”**

The pipeline detects fire-safety-related symbols in 2D evacuation/floorplans using **YOLOv8** and **EasyOCR**, fuses detections and text, maps them to **IFC entities**, and builds a graph-based representation for rule-based reasoning on fire exit routes.

---

## 1. Repository Structure

> You can adjust this list to match your final file layout.

- `EvacuationPlanDataset/` – sample evacuation plan images and labels (if included)
- `ocr_out/` – OCR results and intermediate outputs  
- `out/` – example outputs (overlays, fused CSVs, enrichment, graphs, rule-based results)
- `runs/` – YOLO training / inference runs (optional to publish)
- `yolov8n.pt` – trained YOLOv8 model weights (or link in README if hosted externally)

**Core scripts**

- `run_full_yolo.py` – run YOLOv8 detections on input floorplan images  
- `fuse_yolo_ocr.py` – fuse YOLO detections with OCR text and validate symbols  
- `map_to_semantics.py` – map fused detections to IFC entity types and semantic labels  
- `graph_export.py` – export graph / Graph-BIM representation  
- `run_rule_based_checks.py` – apply rule-based checks on the graph  
- `summarize_rule_results.py` – summarise rule-check results (tables / CSV)  
- `visualize_detections.py` – visualise YOLO detections on the floorplans  
- `visualize_graph_json.py` – visualise the resulting graph (nodes, edges, IFC types)  
- `export_ifc_csv.py` – export enriched elements as CSV for IFC conversion  
- `requirements.txt` – Python dependencies (YOLOv8, EasyOCR, pandas, shapely, etc.)

Utility scripts (optional but included in this repo):

- `deduplicate.py`, `deduplicate_and_remove.py` – dataset and label cleaning
- `parse_fusion_csv.py`, `plot_ifc_type_counts.py`, `fig_detection_count.png`, etc.

---

## 2. Setup

### 2.1 Environment

- **Python**: 3.10+ recommended  
- **OS**: Windows / Linux

Create and activate a virtual environment (optional but recommended):

```bash
# On Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\activate

# On Linux/macOS
python -m venv .venv
source .venv/bin/activate


