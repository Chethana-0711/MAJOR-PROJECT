from pathlib import Path

from ultralytics import YOLO


class SpatialAgent:
    """
    Spatial Agent

    Uses a trained YOLO model to detect camouflage-related
    objects and spatial regions in RGB/VIS imagery.
    """

    def __init__(
        self,
        model_path="runs/detect/spatial_train-3/weights/best.pt",
        confidence_threshold=0.25,
        image_size=512,
        device=0,
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.device = device

        print("Loading Spatial Agent...")
        self.model = YOLO(self.model_path)

    # --------------------------------------------------
    # ANALYZE IMAGE
    # --------------------------------------------------

    def analyze(self, image_path):
      

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

        for box in prediction.boxes:

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
            )

            confidence = float(
                box.conf[0]
                .cpu()
                .numpy()
            )

            class_id = int(
                box.cls[0]
                .cpu()
                .numpy()
            )

            class_name = self.model.names[class_id]

            width = float(x2 - x1)
            height = float(y2 - y1)

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

       
        # SUMMARY STATISTICS
        

        detection_count = len(detections)

        if detections:
            average_confidence = sum(
                d["confidence"] for d in detections
            ) / detection_count

            maximum_confidence = max(
                d["confidence"] for d in detections
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

    # ---------------------
    # VISUALIZATIO
    

    def save_visualization(
        self,
        image_path,
        output_path="outputs/spatial_detection.png",
    ):

        image_path = Path(image_path)
        output_path = Path(output_path)

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

        annotated_image = results[0].plot()

        import cv2

        cv2.imwrite(
            str(output_path),
            annotated_image,
        )

        return output_path