from ultralytics import YOLO

# Load your trained weights
model = YOLO("/home/coder/thesis-yolo/runs_gpu/evac_v12/weights/best.pt")

# Run prediction on your existing prediction images
results = model.predict(
    source="/home/coder/thesis-yolo/runs_gpu/evac_v12_pred/images",
    save=True,
    save_txt=True,
    save_conf=True,
    project="/home/coder/thesis-yolo/runs_gpu",
    name="evac_v12_pred_with_labels",
    exist_ok=True
)

print("✅ Prediction done. Check runs_gpu/evac_v12_pred_with_labels for images + labels.")
