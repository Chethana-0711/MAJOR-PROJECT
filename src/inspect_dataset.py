from pathlib import Path

DATASET_DIR = Path("MUDCAD-X/dataset")

files = list(DATASET_DIR.rglob("*"))

print(f"Total files: {len([f for f in files if f.is_file()])}")

extensions = {}

for file in files:
    if file.is_file():
        ext = file.suffix.lower()
        extensions[ext] = extensions.get(ext, 0) + 1

print("\nFile types:")
for ext, count in sorted(extensions.items()):
    print(f"{ext}: {count}")

print("\nImportant files:")

for name in ["vis.png", "lwir.png", "label.png"]:
    matches = list(DATASET_DIR.rglob(name))
    print(f"{name}: {len(matches)}")