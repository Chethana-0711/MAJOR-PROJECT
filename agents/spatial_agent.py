from pathlib import Path

import cv2
from ultralytics import YOLO


class SpatialAgent:
    """
    Spatial Agent

    Uses a trained YOLO model to detect camouflage-related
    objects and spatial regions in RGB/VIS imagery.

    Supports CPU inference by default.
    """

    def __init__(
        self,
        model_path="runs/detect/spatial_train-3/weights/best.pt",
        confidence_threshold=0.25,
        image_size=512,
        device="cpu",
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.device = device

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLO model weights not found: {self.model_path}\n"
                "Check that the model file exists at the specified path."
            )

        print("Loading Spatial Agent...")
        print(f"Model: {self.model_path}")
        print(f"Device: {self.device}")

        self.model = YOLO(str(self.model_path))

        print("Spatial Agent loaded successfully.")

    # --------------------------------------------------
    # ANALYZE RGB IMAGE
    # --------------------------------------------------

    def analyze(self, image_path):
        """
        Run YOLO inference on an RGB/VIS image.

        Returns:
            Dictionary containing detections and summary statistics.
        """

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"RGB image not found: {image_path}"
            )

        prediction = self.model.predict(
            source=str(image_path),
            imgsz=self.image_size,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )[0]

        detections = []

        # Extract bounding boxes and confidence scores.
        if prediction.boxes is not None:
            for box in prediction.boxes:
                coordinates = (
                    box.xyxy[0].detach().cpu().numpy()
                )

                x1, y1, x2, y2 = coordinates

                confidence = float(
                    box.conf[0].detach().cpu().item()
                )

                class_id = int(
                    box.cls[0].detach().cpu().item()
                )

                class_name = self.model.names[class_id]

                width = max(0.0, float(x2 - x1))
                height = max(0.0, float(y2 - y1))

                detections.append(
                    {
                        "class": class_name,
                        "class_id": class_id,
                        "bbox": [
                            float(x1),
                            float(y1),
                            width,
                            height,
                        ],
                        "confidence": confidence,
                        "area": width * height,
                    }
                )

        # --------------------------------------------------
        # SUMMARY STATISTICS
        # --------------------------------------------------

        detection_count = len(detections)

        if detection_count > 0:
            average_confidence = sum(
                detection["confidence"]
                for detection in detections
            ) / detection_count

            maximum_confidence = max(
                detection["confidence"]
                for detection in detections
            )
        else:
            average_confidence = 0.0
            maximum_confidence = 0.0

        class_counts = {}

        for detection in detections:
            class_name = detection["class"]

            class_counts[class_name] = (
                class_counts.get(class_name, 0) + 1
            )

        return {
            "agent": "spatial",
            "image": str(image_path),
            "detections": detections,
            "detection_count": detection_count,
            "average_confidence": average_confidence,
            "maximum_confidence": maximum_confidence,
            "class_counts": class_counts,
        }

    # --------------------------------------------------
    # SAVE DETECTION VISUALIZATION
    # --------------------------------------------------

    def save_visualization(
        self,
        image_path,
        output_path="outputs/spatial_detection.png",
    ):
        """
        Save an annotated image containing YOLO detections.

        Returns:
            Path to the saved visualization.
        """

        image_path = Path(image_path)
        output_path = Path(output_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"RGB image not found: {image_path}"
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        results = self.model.predict(
            source=str(image_path),
            imgsz=self.image_size,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )

        if not results:
            raise RuntimeError(
                f"YOLO returned no results for: {image_path}"
            )

        annotated_image = results[0].plot()

        success = cv2.imwrite(
            str(output_path),
            annotated_image,
        )

        if not success:
            raise OSError(
                f"Failed to save visualization: {output_path}"
            )

        print(f"Spatial visualization saved to: {output_path}")

        return output_path


# --------------------------------------------------
# OPTIONAL STANDALONE TEST
# --------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test the Spatial Agent on an RGB image."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the RGB/VIS image.",
    )

    parser.add_argument(
        "--model",
        default="runs/detect/spatial_train-3/weights/best.pt",
        help="Path to trained YOLO weights.",
    )

    parser.add_argument(
        "--output",
        default="outputs/spatial_detection.png",
        help="Path for the annotated output image.",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "0"],
        help="Inference device: cpu or CUDA GPU 0.",
    )

    args = parser.parse_args()

    agent = SpatialAgent(
        model_path=args.model,
        device=args.device,
    )

    result = agent.analyze(args.image)

    print("\nSPATIAL AGENT RESULTS")
    print("=" * 50)
    print(f"Image: {result['image']}")
    print(f"Detections: {result['detection_count']}")
    print(
        f"Average confidence: "
        f"{result['average_confidence']:.4f}"
    )
    print(
        f"Maximum confidence: "
        f"{result['maximum_confidence']:.4f}"
    )
    print(f"Class counts: {result['class_counts']}")

    output_path = agent.save_visualization(
        args.image,
        args.output,
    )

    print(f"Visualization: {output_path}")