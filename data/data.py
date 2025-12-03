import os
import json
import urllib.request
from tqdm import tqdm
from pathlib import Path

# Choose your classes
SELECTED_CLASSES = ["thumbs_up", "palm", "fist", "pointing_left", "pointing_right"]

# Directory to store data
data_dir = Path("hagrid_dataset")
data_dir.mkdir(exist_ok=True)

# Download annotations
annotations_url = "https://hagrid-dataset.s3.ru-hacks.com/hagrid/annotations/hagrid_1024.json"
annotations_path = data_dir / "annotations.json"

urllib.request.urlretrieve(annotations_url, annotations_path)

with open(annotations_path, "r") as f:
    annotations = json.load(f)

# Filter for selected classes
filtered = [item for item in annotations if item["gesture"] in SELECTED_CLASSES]

print("Total selected samples:", len(filtered))

# Download images
img_dir = data_dir / "images"
img_dir.mkdir(exist_ok=True)

for item in tqdm(filtered[:20000]):  # Limit for faster training (recommended: 20k–50k)
    url = item["image"]
    filename = img_dir / f"{item['image'].split('/')[-1]}"
    try:
        urllib.request.urlretrieve(url, filename)
    except:
        pass