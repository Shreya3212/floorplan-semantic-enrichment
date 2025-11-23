from ultralytics import YOLO

model = YOLO("/home/coder/thesis-yolo/EvacuationPlanDataset/EvacuationPlanDataset/runs/detect/train4/weights/best.pt")

model.predict(
    source="/home/coder/thesis-yolo/EvacuationPlanDataset/EvacuationPlanDataset/train/images",
    save=True,
    save_txt=True,
    conf=0.25,
    project="/home/coder/thesis-yolo/runs_gpu",
    name="evac_v12_pred_full"
)
