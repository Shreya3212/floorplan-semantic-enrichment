# Python script to detect duplicates
import os
import imagehash
from PIL import Image
from collections import defaultdict

folder = '/home/coder/thesis-yolo/EvacuationPlanDataset/EvacuationPlanDataset/train/images'
hashes = defaultdict(list)

for filename in os.listdir(folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(folder, filename)
        try:
            with Image.open(path) as img:
                h = str(imagehash.phash(img))
                hashes[h].append(filename)
        except Exception as e:
            print(f"Error with {filename}: {e}")

# Print duplicate groups
for h, files in hashes.items():
    if len(files) > 1:
        print(f"\nDuplicates for hash {h}:")
        for f in files:
            print(f" - {f}")
