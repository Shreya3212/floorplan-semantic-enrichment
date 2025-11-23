import os
import imagehash
from PIL import Image
from collections import defaultdict

# Set your image directory here
img_dir = '/home/coder/thesis-yolo/EvacuationPlanDataset/EvacuationPlanDataset/train/images'

hashes = defaultdict(list)

# Collect all image files
for fname in os.listdir(img_dir):
    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(img_dir, fname)
        try:
            with Image.open(path) as img:
                h = imagehash.phash(img)
                hashes[str(h)].append(path)
        except Exception as e:
            print(f"Failed to process {fname}: {e}")

# Remove duplicates (keep the first in each group)
removed = 0
for h, paths in hashes.items():
    if len(paths) > 1:
        # Keep first, delete others
        for dup in paths[1:]:
            try:
                os.remove(dup)
                print(f"Removed duplicate: {dup}")
                removed += 1
            except Exception as e:
                print(f"Error removing {dup}: {e}")

print(f"\n✅ Done. Removed {removed} duplicate images.")
