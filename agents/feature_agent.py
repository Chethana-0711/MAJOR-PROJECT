from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


# ==================================================
# SETTINGS
# ==================================================

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "multimodal_cnn.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 64
STRIDE = 32
BATCH_SIZE = 128

THRESHOLD = 0.70
MIN_AREA = 20
LOCAL_EVIDENCE_THRESHOLD = 0.75


# ==================================================
# CNN BRANCH
# ==================================================

class CNNBranch(nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                input_channels, 32,
                kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32, 64,
                kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                64, 128,
                kernel_size=3, padding=1
            ),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.features(x)
        return x.flatten(1)


# ==================================================
# MULTIMODAL CNN
# ==================================================

class MultimodalCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.rgb_branch = CNNBranch(3)
        self.lwir_branch = CNNBranch(1)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, rgb, lwir):
        rgb_features = self.rgb_branch(rgb)
        lwir_features = self.lwir_branch(lwir)

        combined = torch.cat(
            [rgb_features, lwir_features],
            dim=1
        )

        return self.classifier(combined).squeeze(1)


# ==================================================
# FEATURE AGENT
# ==================================================

class FeatureAgent:
    """

    combines patch probabilities into an image-level probability map,
    and extracts candidate regions.
    """

    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Multimodal CNN model not found: {MODEL_PATH}"
            )

        print(f"Feature Agent device: {DEVICE}")

        self.model = MultimodalCNN()
        self.model.load_state_dict(
            torch.load(MODEL_PATH, map_location=DEVICE)
        )
        self.model.to(DEVICE)
        self.model.eval()

    # ==================================================
    # PATCH PREPARATION
    # ==================================================

    def prepare_rgb_patch(self, patch):
        patch = cv2.resize(
            patch, (IMAGE_SIZE, IMAGE_SIZE)
        )
        patch = cv2.cvtColor(
            patch, cv2.COLOR_BGR2RGB
        )
        patch = patch.astype(np.float32) / 255.0

        return torch.from_numpy(patch).permute(2, 0, 1)

    def prepare_lwir_patch(self, patch):
        patch = cv2.resize(
            patch, (IMAGE_SIZE, IMAGE_SIZE)
        )
        patch = patch.astype(np.float32) / 255.0

        return torch.from_numpy(patch).unsqueeze(0)

    # ==================================================
    # GENERATE PATCHES
    # ==================================================

    def generate_patches(self, image):
        height, width = image.shape[:2]

        patches = []
        boxes = []

        y_positions = range(
            0, max(height - IMAGE_SIZE + 1, 1), STRIDE
        )
        x_positions = range(
            0, max(width - IMAGE_SIZE + 1, 1), STRIDE
        )

        for y in y_positions:
            for x in x_positions:
                x2 = min(x + IMAGE_SIZE, width)
                y2 = min(y + IMAGE_SIZE, height)

                patch = image[y:y2, x:x2]

                # This implementation only processes full-size patches.
                if (
                    patch.shape[0] != IMAGE_SIZE
                    or patch.shape[1] != IMAGE_SIZE
                ):
                    continue

                patches.append(patch)
                boxes.append([x, y, IMAGE_SIZE, IMAGE_SIZE])

        return patches, boxes

    # ==================================================
    # BATCH PREDICTION:


    def predict_batches(self, rgb_patches, lwir_patches):
        if len(rgb_patches) != len(lwir_patches):
            raise ValueError(
                "RGB and LWIR patch counts do not match. "
                "Ensure the images are aligned and have the same dimensions."
            )

        probabilities = []
        total = len(rgb_patches)

        for start in range(0, total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total)

            rgb_batch = torch.stack([
                self.prepare_rgb_patch(p)
                for p in rgb_patches[start:end]
            ]).to(DEVICE)

            lwir_batch = torch.stack([
                self.prepare_lwir_patch(p)
                for p in lwir_patches[start:end]
            ]).to(DEVICE)

            with torch.no_grad():
                logits = self.model(rgb_batch, lwir_batch)
                batch_probabilities = torch.sigmoid(logits)

            probabilities.extend(
                batch_probabilities.cpu().numpy().tolist()
            )

        return np.asarray(probabilities, dtype=np.float32)

    # ==================================================
    # ANALYZE: BOTH IMAGES REQUIRED
    # ==================================================

    def analyze(self, rgb_path, lwir_path):
        if rgb_path is None or lwir_path is None:
            raise ValueError(
                "Both RGB and LWIR image paths are required."
            )

        rgb_path = Path(rgb_path)
        lwir_path = Path(lwir_path)

        if not rgb_path.exists():
            raise FileNotFoundError(
                f"RGB image not found: {rgb_path}"
            )

        if not lwir_path.exists():
            raise FileNotFoundError(
                f"LWIR image not found: {lwir_path}"
            )

        rgb_image = cv2.imread(str(rgb_path))
        lwir_image = cv2.imread(
            str(lwir_path),
            cv2.IMREAD_GRAYSCALE
        )

        if rgb_image is None:
            raise ValueError(f"Could not load RGB image: {rgb_path}")

        if lwir_image is None:
            raise ValueError(f"Could not load LWIR image: {lwir_path}")

        # Resize LWIR to the RGB image dimensions so patch positions match.
        height, width = rgb_image.shape[:2]
        if lwir_image.shape[:2] != (height, width):
            lwir_image = cv2.resize(
                lwir_image, (width, height)
            )

        rgb_patches, boxes = self.generate_patches(rgb_image)
        lwir_patches, lwir_boxes = self.generate_patches(lwir_image)

        # Both images must produce the same patch grid.
        if boxes != lwir_boxes:
            raise ValueError(
                "RGB and LWIR patch grids do not match."
            )

        probabilities = self.predict_batches(
            rgb_patches, lwir_patches
        )

        # Build a full-image probability map by averaging probabilities
        # from every patch covering each pixel.
        probability_map = np.zeros(
            (height, width), dtype=np.float32
        )
        count_map = np.zeros(
            (height, width), dtype=np.float32
        )

        for probability, box in zip(probabilities, boxes):
            x, y, w, h = box
            probability_map[y:y+h, x:x+w] += probability
            count_map[y:y+h, x:x+w] += 1

        valid = count_map > 0
        probability_map[valid] /= count_map[valid]

        # Threshold the probability map to create a binary mask.
        detection_mask = (
            probability_map >= THRESHOLD
        ).astype(np.uint8) * 255

        # Remove tiny noise, then close small gaps.
        kernel = np.ones((5, 5), np.uint8)
        detection_mask = cv2.morphologyEx(
            detection_mask, cv2.MORPH_OPEN, kernel
        )
        detection_mask = cv2.morphologyEx(
            detection_mask, cv2.MORPH_CLOSE, kernel
        )

        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                detection_mask,
                connectivity=8
            )
        )

        regions = []

        for i in range(1, num_labels):
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])

            if area < MIN_AREA:
                continue

            region_mask = labels == i
            region_probabilities = probability_map[region_mask]

            regions.append({
                "bbox": [x, y, w, h],
                "area": area,
                "evidence": float(region_probabilities.mean()),
                "max_probability": float(region_probabilities.max())
            })

        regions.sort(
            key=lambda region: region["evidence"],
            reverse=True
        )

        if len(probabilities) > 0:
            overall_evidence = float(np.mean(probabilities))
            max_probability = float(np.max(probabilities))
            max_index = int(np.argmax(probabilities))
        else:
            overall_evidence = 0.0
            max_probability = 0.0
            max_index = -1

        localized_evidence_regions = []

        # Preserve the strongest patch as localized evidence when it
        # exceeds the evidence threshold but no formal region survives.
        if (
            max_index >= 0
            and max_probability >= LOCAL_EVIDENCE_THRESHOLD
        ):
            peak_box = boxes[max_index]
            localized_evidence_regions.append({
                "bbox": [
                    int(peak_box[0]),
                    int(peak_box[1]),
                    int(peak_box[2]),
                    int(peak_box[3])
                ],
                "area": int(peak_box[2] * peak_box[3]),
                "evidence": max_probability,
                "max_probability": max_probability,
                "evidence_type": "localized"
            })

        if regions:
            status = "DETECTED"
        elif localized_evidence_regions:
            status = "EVIDENCE_PRESENT"
        else:
            status = "NO_DETECTION"

        return {
            "agent": "feature",
            "status": status,
            "modality": "rgb+lwir",
            "rgb_available": True,
            "lwir_available": True,
            "overall_evidence": overall_evidence,
            "max_probability": max_probability,
            "regions": regions,
            "localized_evidence_regions": localized_evidence_regions,
            "probability_map": probability_map,
            "detection_mask": detection_mask,
            "num_patches": len(probabilities)
        }

    # ==================================================
    # SAVE VISUALIZATION
    # ==================================================

    def save_visualization(
        self,
        rgb_path,
        lwir_path,
        output_path="outputs/feature_analysis.png"
    ):
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True, exist_ok=True
        )

        result = self.analyze(
            rgb_path=rgb_path,
            lwir_path=lwir_path
        )

        rgb_image = cv2.imread(str(rgb_path))
        if rgb_image is None:
            raise ValueError(f"Could not load RGB image for visualization: {rgb_path}")

        probability_map = result["probability_map"]
        modality = result["modality"]

        # Probability heatmap.
        heatmap = (
            np.clip(probability_map, 0, 1) * 255
        ).astype(np.uint8)
        heatmap = cv2.applyColorMap(
            heatmap, cv2.COLORMAP_JET
        )

        overlay = cv2.addWeighted(
            rgb_image, 0.55, heatmap, 0.45, 0
        )

        regions_image = rgb_image.copy()
        for index, region in enumerate(
            result["regions"][:10], start=1
        ):
            x, y, w, h = region["bbox"]

            cv2.rectangle(
                regions_image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            label = f"R{index} E={region['evidence']:.2f}"
            cv2.putText(
                regions_image,
                label,
                (x, max(y - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

        def add_title(image, title):
            canvas = image.copy()
            cv2.rectangle(
                canvas,
                (0, 0),
                (canvas.shape[1], 38),
                (0, 0, 0),
                -1
            )
            cv2.putText(
                canvas,
                title,
                (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            return canvas

        input_display = add_title(
            rgb_image, f"RGB INPUT ({modality.upper()})"
        )
        heatmap_display = add_title(
            heatmap, "CNN FEATURE PROBABILITY MAP"
        )
        overlay_display = add_title(
            overlay, "FEATURE EVIDENCE OVERLAY"
        )
        regions_display = add_title(
            regions_image, "FEATURE CANDIDATE REGIONS"
        )

        top_row = np.hstack([
            input_display, heatmap_display
        ])
        bottom_row = np.hstack([
            overlay_display, regions_display
        ])
        grid = np.vstack([top_row, bottom_row])

        summary_height = 145
        summary = np.zeros(
            (summary_height, grid.shape[1], 3),
            dtype=np.uint8
        )

        summary_lines = [
            "FEATURE AGENT",
            f"Mode: {modality.upper()}",
            f"Status: {result['status']}",
            f"Patches scanned: {result['num_patches']}",
            f"Overall evidence: {result['overall_evidence']:.3f}",
            f"Maximum probability: {result['max_probability']:.3f}",
            f"Candidate regions: {len(result['regions'])}"
        ]

        y_position = 22
        for line in summary_lines:
            cv2.putText(
                summary,
                line,
                (15, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            y_position += 18

        final_visualization = np.vstack([grid, summary])
        cv2.imwrite(str(output_path), final_visualization)

        return output_path


# ==================================================
# STANDALONE TEST
# ==================================================

if __name__ == "__main__":
    dataset = Path("MUDCAD-X/dataset")

    rgb_path = next(dataset.rglob("vis.png"))
    lwir_path = rgb_path.parent / "lwir.png"

    agent = FeatureAgent()
    result = agent.analyze(
        rgb_path=rgb_path,
        lwir_path=lwir_path
    )

    print("\nFEATURE AGENT")
    print("=========================")
    print(f"Modality: {result['modality']}")
    print(f"Status: {result['status']}")
    print(f"RGB available: {result['rgb_available']}")
    print(f"LWIR available: {result['lwir_available']}")
    print(f"Patches scanned: {result['num_patches']}")
    print(f"Overall feature evidence: {result['overall_evidence']:.3f}")
    print(f"Maximum feature probability: {result['max_probability']:.3f}")
    print(f"Feature regions: {len(result['regions'])}")
    print(
        "Localized evidence regions: "
        f"{len(result['localized_evidence_regions'])}"
    )

    for i, region in enumerate(result["regions"][:10], start=1):
        print(f"\nRegion {i}")
        print(f"  Box: {region['bbox']}")
        print(f"  Area: {region['area']}")
        print(f"  Evidence: {region['evidence']:.3f}")
        print(f"  Max probability: {region['max_probability']:.3f}")

    visualization_path = agent.save_visualization(
        rgb_path=rgb_path,
        lwir_path=lwir_path
    )
    print(f"\nVisualization saved to:\n{visualization_path}")
