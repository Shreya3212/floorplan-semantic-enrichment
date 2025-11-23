#!/usr/bin/env python3
import argparse, re, json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
import easyocr

# ---------- Utilities ----------
def clean_ocr(text, conf, min_conf=0.3):
    if conf < min_conf:
        return None
    text = text.strip().upper()
    if "EXIT" in text:
        return "EXIT"
    if "STAIR" in text:
        return "STAIRS"
    if "EXT" in text or "FIRE" in text:
        return "FIRE_EXTINGUISHER"
    return None

def yolo_txt_to_xyxy(txt_path, img_w, img_h):
    boxes = []
    if not txt_path.exists():
        return boxes
    for line in txt_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:5])
        conf = float(parts[5]) if len(parts) == 6 else 1.0
        x1 = int((xc - w/2) * img_w)
        y1 = int((yc - h/2) * img_h)
        x2 = int((xc + w/2) * img_w)
        y2 = int((yc + h/2) * img_h)
        boxes.append([x1, y1, x2, y2, cls_id, conf])
    return boxes

def poly_to_box(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    areaA = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    areaB = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / (areaA + areaB - inter + 1e-6)

def normalize_text(t):
    t = t.lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def text_matches(cls_name, text):
    t = normalize_text(text)
    if cls_name == "exit":
        return "exit" in t
    if cls_name == "fire extinguisher":
        return "extinguisher" in t or "fire extinguisher" in t
    if cls_name == "hydrant":
        return "hydrant" in t
    if cls_name == "stairs":
        return "stair" in t
    return False

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--classes", default="exit,fire extinguisher,hydrant,stairs")
    ap.add_argument("--out", default="fused_out")
    ap.add_argument("--ocr_gpu", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--iou", type=float, default=0.3)
    args = ap.parse_args()

    images_dir = Path(args.images)
    pred_dir = Path(args.pred)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    out_img = out_dir / "images"; out_img.mkdir(exist_ok=True)
    out_csv = out_dir / "summary.csv"

    class_names = [c.strip() for c in args.classes.split(",")]
    reader = easyocr.Reader(['en'], gpu=args.ocr_gpu)

    img_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in [".png",".jpg",".jpeg",".bmp",".tif",".tiff"]])
    if args.limit:
        img_paths = img_paths[:args.limit]

    rows = []

    for i, img_p in enumerate(img_paths, 1):
        img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        lab_p = pred_dir / "labels" / (img_p.stem + ".txt")
        yolo_boxes = yolo_txt_to_xyxy(lab_p, w, h)

        ocr = reader.readtext(str(img_p), detail=1, paragraph=False)
        ocr_boxes = []
        for poly, text, score in ocr:
            try:
                box = poly_to_box(poly)
                raw_text = text
                cleaned = clean_ocr(text, score)
                if cleaned is None:
                    continue
                ocr_boxes.append({
                    "bbox": box,
                    "text_raw": raw_text,
                    "text_clean": cleaned,
                    "score": float(score)
                })
            except Exception:
                continue

        for (x1,y1,x2,y2, cls_id, conf) in yolo_boxes:
            cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
            status = "no_text"
            matched_raw = ""
            matched_clean = ""
            best_iou = 0.0

            for ob in ocr_boxes:
                ii = iou_xyxy([x1,y1,x2,y2], ob["bbox"])
                if ii > args.iou and ii > best_iou:
                    best_iou = ii
                    matched_raw = ob["text_raw"]
                    matched_clean = ob["text_clean"]
                    if text_matches(cls_name, ob["text_clean"]):
                        status = "validated"
                    else:
                        status = "conflict"

            if status == "validated":
                color = (0, 255, 0)       # green
            elif status == "conflict":
                color = (0, 165, 255)     # orange
            else:
                color = (0, 255, 255)     # yellow for no_text

            cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
            label = f"{cls_name} {conf:.2f} | {status}"
            if matched_clean:
                label += f" | '{matched_clean}'"
            cv2.putText(img, label, (x1, max(20,y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            rows.append({
    "image": img_p.name,
    "cls": cls_name,
    "conf": round(conf,3),
    "status": status,
    "ocr_raw": matched_raw,
    "ocr_clean": matched_clean,
    "best_iou": round(best_iou,3),
    "bbox_x": x1,
    "bbox_y": y1,
    "bbox_w": x2 - x1,
    "bbox_h": y2 - y1
})

        cv2.imwrite(str(out_img / img_p.name), img)
        print(f"[{i}/{len(img_paths)}] -> {img_p.name}")

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(out_csv, index=False)
        by = df.groupby(["cls","status"]).size().reset_index(name="count")
        print("\nSummary:")
        print(by.to_string(index=False))
        print(f"\nSaved: {out_csv}")
    else:
        print("No detections found to fuse.")

if __name__ == "__main__":
    main()
