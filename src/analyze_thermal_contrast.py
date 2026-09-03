from pathlib import Path

import numpy as np
from PIL import Image


DATASET_DIR = Path("MUDCAD-X/dataset")


results = []


for label_path in DATASET_DIR.rglob("label.png"):

    capture = label_path.parent
    lwir_path = capture / "lwir.png"

    if not lwir_path.exists():
        continue

    label = np.array(Image.open(label_path))
    lwir = np.array(Image.open(lwir_path)).astype(np.float32)

    # Ground-truth target pixels
    target_mask = np.any(label != 0, axis=2)

    # Skip captures without target pixels
    if target_mask.sum() == 0:
        continue

    target_mean = lwir[target_mask].mean()
    background_mean = lwir[~target_mask].mean()

    difference = target_mean - background_mean

    results.append(difference)


results = np.array(results)


print("\nTHERMAL CONTRAST ANALYSIS")
print("----------------------------")

print("Captures analyzed:", len(results))

print("Mean contrast:", results.mean())
print("Median contrast:", np.median(results))
print("Minimum contrast:", results.min())
print("Maximum contrast:", results.max())

print(
    "Targets thermally different from background:",
    np.sum(results != 0)
)

print(
    "Positive contrast:",
    np.sum(results > 0)
)

print(
    "Negative contrast:",
    np.sum(results < 0)
)