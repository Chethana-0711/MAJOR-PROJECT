from pathlib import Path
import random
import cv2


DATASET_DIR = Path("MUDCAD-X/dataset")
LABEL_FILE = Path("data/yolo/labels/train/train_0001.txt")

# Recreate the exact same split used by our converter
captures = []

for label_path in DATASET_DIR.rglob("label.png"):
    capture_dir = label_path.parent

    vis_path = capture_dir / "vis.png"
    lwir_path = capture_dir / "lwir.png"

    if vis_path.exists() and lwir_path.exists():
        captures.append({
            "dir": capture_dir,
            "vis": vis_path,
            "lwir": lwir_path,
            "label": label_path
        })

random.seed(42)
random.shuffle(captures)

capture = captures[1]

print("Original capture:")
print(capture["dir"])

print("\nVIS:")
print(capture["vis"])

print("\nLabel:")
print(capture["label"])


# Load VIS image
image = cv2.imread(str(capture["vis"]))

height, width = image.shape[:2]

# Read YOLO annotations
lines = LABEL_FILE.read_text().strip().splitlines()

for line in lines:

    values = line.split()

    class_id = int(values[0])
    x_center = float(values[1])
    y_center = float(values[2])
    box_width = float(values[3])
    box_height = float(values[4])

    # Convert YOLO format back to pixels
    x_center *= width
    y_center *= height
    box_width *= width
    box_height *= height

    x1 = int(x_center - box_width / 2)
    y1 = int(y_center - box_height / 2)
    x2 = int(x_center + box_width / 2)
    y2 = int(y_center + box_height / 2)

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    cv2.putText(
        image,
        f"class {class_id}",
        (x1, max(y1 - 5, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
    )


output = Path("outputs/verify_train_0001.jpg")
output.parent.mkdir(exist_ok=True)

cv2.imwrite(str(output), image)

print("\nSaved verification image:")
print(output)