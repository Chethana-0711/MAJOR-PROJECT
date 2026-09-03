from pathlib import Path
from PIL import Image

capture = Path(
    "MUDCAD-X/dataset/01_Spring/2022-05-10/rk1/a/1/h0/0"
)

for name in ["vis.png", "lwir.png", "label.png"]:
    path = capture / name
    image = Image.open(path)

    print(f"\n{name}")
    print("Size:", image.size)
    print("Mode:", image.mode)
    print("Format:", image.format)