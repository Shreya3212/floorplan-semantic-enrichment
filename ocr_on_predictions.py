import os, glob, argparse, math
import cv2
import numpy as np
import pandas as pd
import easyocr

# ---------- Class map: adjust if needed ----------
CLASS_MAP = {
    0: "exit",
    1: "fire_extinguisher",
    2: "hydrant",
    3: "stairs",
}

def build_argparser():
    ap = argparse.ArgumentParser(description="OCR on YOLO predictions for floor plans")
    ap.add_argument("--pred-root", type=str, required=True,
                    help="Root folder containing predict*/ images + labels/")
    ap.add_argument("--out", type=str, required=True,
                    help="Output folder (CSV + overlays)")
    ap.add_argument("--langs", type=str, default="en,de",
                    help="EasyOCR languages, comma-separated (e.g., 'en,de')")
    ap.add_argument("--pad", type=int, default=6,
                    help="Padding (pixels) around bbox before OCR")
    ap.add_argument("--min_side", type=int, default=24,
                    help="Min side of crop after scaling; will upscale if smaller")
    ap.add_argument("--double_pass", action="store_true",
                    help="Try a second-pass OCR with different preprocessing if first is empty")
    ap.add_argument("--conf_thresh", type=float, default=0.20,
                    help="Minimum YOLO confidence to run OCR on that bbox")
    return ap

def list_prediction_sets(root):
    """
    Returns a list of (images_dir, labels_dir) under predict-like folders.
    Supports roboflow/yolo typical structure: predict*/images + predict*/labels
    Also works if images live directly in predict* and labels in predict*/labels.
    """
    pairs = []
    for pred_path in sorted(glob.glob(os.path.join(root, "**"), recursive=True)):
        if not os.path.isdir(pred_path): 
            continue
        labels_dir = os.path.join(pred_path, "labels")
        images_dir = os.path.join(pred_path, "images")
        if os.path.isdir(labels_dir) and os.path.isdir(images_dir):
            pairs.append((images_dir, labels_dir))
            continue
        # fallback: images directly in pred_path and labels in pred_path/labels
        if os.path.isdir(labels_dir):
            pairs.append((pred_path, labels_dir))
    return pairs

def yolo_txt_to_boxes(label_path, img_w, img_h):
    """
    YOLO format per line: <cls> <cx> <cy> <w> <h> [conf]
    Normalized in [0,1]. Returns list of dicts.
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
            conf = float(parts[5]) if len(parts) >= 6 else 1.0
            x = (cx - w/2.0) * img_w
            y = (cy - h/2.0) * img_h
            w_pix = w * img_w
            h_pix = h * img_h
            boxes.append({
                "cls": cls,
                "cls_name": CLASS_MAP.get(cls, f"class_{cls}"),
                "x": int(round(x)),
                "y": int(round(y)),
                "w": int(round(w_pix)),
                "h": int(round(h_pix)),
                "conf": conf,
            })
    return boxes

def safe_crop(img, x, y, w, h, pad=0):
    H, W = img.shape[:2]
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    return img[y0:y1, x0:x1]

def preprocess_for_ocr(crop):
    """
    Light, robust preprocessing for CAD labels:
    - Convert to gray
    - CLAHE for contrast
    - Otsu binarization
    - Light morphology to clean noise
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g2 = clahe.apply(gray)
    # Otsu
    th = cv2.threshold(g2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # Optional small opening to remove specks
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    return th

def preprocess_alt(crop):
    """
    Alternate preprocessing for second pass:
    - Adaptive threshold (good for uneven lighting)
    - Invert if text appears light
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 7
    )
    # Heuristic invert if background seems dark
    if np.mean(th) > 127:
        th = cv2.bitwise_not(th)
    return th

def ensure_min_size(crop, min_side=24):
    """
    Upscale tiny crops to help OCR read small glyphs.
    """
    h, w = crop.shape[:2]
    scale = max(1.0, float(min_side)/float(min(h, w)))
    if scale > 1.01:
        crop = cv2.resize(crop, (int(round(w*scale)), int(round(h*scale))), interpolation=cv2.INTER_CUBIC)
    return crop

def ocr_easy(reader, img):
    out = reader.readtext(img, detail=1, paragraph=False)
    texts = []
    for item in out:
        if len(item) >= 3:
            _, txt, conf = item[:3]
        elif len(item) == 2:
            _, txt = item
            conf = 0.0
        else:
            txt, conf = "", 0.0
        texts.append((txt.strip(), float(conf)))
    if not texts:
        return "", 0.0
    # pick by longest text then confidence
    return max(texts, key=lambda t: (len(t[0]), t[1]))

def main():
    args = build_argparser().parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "overlays"), exist_ok=True)

    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    print(f"Using EasyOCR languages: {langs}")
    reader = easyocr.Reader(langs, gpu=True)  # switch to False if GPU not supported

    rows = []
    pred_sets = list_prediction_sets(args.pred_root)
    if not pred_sets:
        print(f"No predict folders found under: {args.pred_root}")
        return

    total_imgs = 0
    for images_dir, labels_dir in pred_sets:
        # find images by common extensions
        img_paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp"):
            img_paths.extend(glob.glob(os.path.join(images_dir, ext)))
        img_paths.sort()

        for img_path in img_paths:
            total_imgs += 1
            base = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(labels_dir, base + ".txt")
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"[WARN] Cannot read image: {img_path}")
                continue
            H, W = img.shape[:2]
            boxes = yolo_txt_to_boxes(label_path, W, H)

            overlay = img.copy()
            for b in boxes:
                if b["conf"] < args.conf_thresh:
                    continue
                crop = safe_crop(img, b["x"], b["y"], b["w"], b["h"], pad=args.pad)
                if crop is None:
                    continue

                # Preprocess + upscale
                p1 = preprocess_for_ocr(crop)
                p1 = ensure_min_size(p1, args.min_side)
                txt, conf = ocr_easy(reader, p1)

                # Optional second pass if empty
                if args.double_pass and (not txt or txt.strip() == ""):
                    p2 = preprocess_alt(crop)
                    p2 = ensure_min_size(p2, args.min_side)
                    txt2, conf2 = ocr_easy(reader, p2)
                    if conf2 > conf:
                        txt, conf = txt2, conf2

                # Visualize
                x, y, w, h = b["x"], b["y"], b["w"], b["h"]
                cv2.rectangle(overlay, (x, y), (x+w, y+h), (0,255,0), 1)
                label = f'{b["cls_name"]} {b["conf"]:.2f}'
                if txt:
                    label += f' | "{txt[:18]}"'
                cv2.putText(overlay, label, (x, max(12, y-4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1, cv2.LINE_AA)

                rows.append({
                    "image_path": img_path,
                    "label_path": label_path if os.path.exists(label_path) else "",
                    "class_id": b["cls"],
                    "class_name": b["cls_name"],
                    "conf_det": round(float(b["conf"]), 4),
                    "x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"],
                    "ocr_text": (txt or "").strip(),
                    "ocr_conf": round(float(conf), 4),
                })

            # write overlay per image
            out_name = base + "_ocr_overlay.png"
            cv2.imwrite(os.path.join(args.out, "overlays", out_name), overlay)

            print(f"Processed {total_imgs}: {os.path.basename(img_path)} "
                  f"→ {len(boxes)} dets, OCR rows now {len(rows)}")

    # Save CSV
    df = pd.DataFrame(rows, columns=[
        "image_path","label_path","class_id","class_name","conf_det",
        "x","y","w","h","ocr_text","ocr_conf"
    ])
    csv_path = os.path.join(args.out, "floorplan_ocr_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")
    print(f"Overlays dir: {os.path.join(args.out, 'overlays')}")
    print("Done.")

if __name__ == "__main__":
    main()
