import cv2
import numpy as np
import joblib

from pathlib import Path


# ============================================================
# SPECTRAL AGENT
# ============================================================

class SpectralAgent:


    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        min_area=20,
        model_path="models/spectral_rf.pkl"
    ):

        self.min_area = min_area

        self.model_path = Path(
            model_path
        )

        if not self.model_path.exists():

            raise FileNotFoundError(
                f"Spectral RF model not found: "
                f"{self.model_path}"
            )

        # ----------------------------------------------------
        # LOAD RANDOM FOREST
        # ----------------------------------------------------

        self.model = joblib.load(
            self.model_path
        )

    # ========================================================
    # FEATURE EXTRACTION
    # ========================================================

    def _extract_features(
        self,
        lwir
    ):
        """
        Create the same four feature types used by the
        trained Random Forest.
        """

        lwir_float = (
            lwir.astype(
                np.float32
            )
        )

        # ----------------------------------------------------
        # NORMALIZED LWIR INTENSITY
        # ----------------------------------------------------

        normalized_image = (
            lwir_float /
            255.0
        )

        # ----------------------------------------------------
        # LOCAL MEAN
        # ----------------------------------------------------

        local_mean = cv2.GaussianBlur(
            normalized_image,
            (15, 15),
            0
        )

        # ----------------------------------------------------
        # LOCAL STANDARD DEVIATION
        # ----------------------------------------------------

        local_sq_mean = cv2.GaussianBlur(
            normalized_image ** 2,
            (15, 15),
            0
        )

        variance = (
            local_sq_mean -
            local_mean ** 2
        )

        variance = np.maximum(
            variance,
            0
        )

        local_std = np.sqrt(
            variance
        )

        # ----------------------------------------------------
        # LOCAL CONTRAST
        # ----------------------------------------------------

        local_contrast = np.abs(
            normalized_image -
            local_mean
        )

        # ----------------------------------------------------
        # PIXEL FEATURE MATRIX
        # ----------------------------------------------------

        features = np.column_stack([

            normalized_image.ravel(),

            local_mean.ravel(),

            local_std.ravel(),

            local_contrast.ravel()

        ])

        return (
            features,
            normalized_image,
            local_mean,
            local_std,
            local_contrast
        )

    # ============================
    # RANDOM FOREST PREDICTION
    # ==========

    def _predict_probability(
        self,
        features
    ):
        """
        Generate target probability for every pixel.

        If the Random Forest was trained with feature names,
        preserve those names to avoid sklearn feature-name
        warnings.
        """

        # ----------------------------------------------------
        # Try to preserve trained feature names
        # ----------------------------------------------------

        if hasattr(
            self.model,
            "feature_names_in_"
        ):

            import pandas as pd

            feature_names = list(
                self.model.feature_names_in_
            )

            feature_data = pd.DataFrame(
                features,
                columns=feature_names
            )

            probabilities = (
                self.model.predict_proba(
                    feature_data
                )[:, 1]
            )

        else:

            probabilities = (
                self.model.predict_proba(
                    features
                )[:, 1]
            )

        return probabilities

    # ========================================================
    # ANALYZE LWIR
    # ========================================================

    def analyze(
        self,
        lwir_path
    ):
        """
        Analyse one LWIR image.
        """

        lwir_path = Path(
            lwir_path
        )

        if not lwir_path.exists():

            raise FileNotFoundError(
                f"LWIR image not found: "
                f"{lwir_path}"
            )

        # ====================================================
        # LOAD IMAGE
        # ====================================================

        lwir = cv2.imread(
            str(lwir_path),
            cv2.IMREAD_GRAYSCALE
        )

        if lwir is None:

            raise ValueError(
                f"Could not load LWIR image: "
                f"{lwir_path}"
            )

        # ====================================================
        # EXTRACT RF FEATURES
        # ====================================================

        (
            features,
            normalized_image,
            local_mean,
            local_std,
            local_contrast
        ) = self._extract_features(
            lwir
        )

        # ====================================================
        # RANDOM FOREST PROBABILITY
        # ====================================================

        probabilities = (
            self._predict_probability(
                features
            )
        )

        target_probability = (
            probabilities.reshape(
                lwir.shape
            )
        )

        target_probability = np.clip(
            target_probability,
            0.0,
            1.0
        )

        # ====================================================
        # RF EVIDENCE STATISTICS
        # ====================================================

        overall_ml_evidence = float(
            np.mean(
                target_probability
            )
        )

        max_probability = float(
            np.max(
                target_probability
            )
        )

        # ====================================================
        # CANDIDATE THRESHOLD
        # ==============================n =


        RF_THRESHOLD = 0.70

        candidate_mask = (
            target_probability >=
            RF_THRESHOLD
        ).astype(
            np.uint8
        ) * 255

        # ====================================================
        # MORPHOLOGICAL CLEANING
        # =============

        kernel = np.ones(
            (3, 3),
            np.uint8
        )

        candidate_mask = cv2.morphologyEx(
            candidate_mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        # ====================================================
        # CONNECTED COMPONENTS
        # ====================

        (
            num_labels,
            labels,
            stats,
            centroids
        ) = cv2.connectedComponentsWithStats(
            candidate_mask,
            connectivity=8
        )

        regions = []

        localized_evidence_regions = []

        # ====================================================
        # ANALYSE EACH RF COMPONENT
        # ====================================================

        for i in range(
            1,
            num_labels
        ):

            x = int(
                stats[
                    i,
                    cv2.CC_STAT_LEFT
                ]
            )

            y = int(
                stats[
                    i,
                    cv2.CC_STAT_TOP
                ]
            )

            w = int(
                stats[
                    i,
                    cv2.CC_STAT_WIDTH
                ]
            )

            h = int(
                stats[
                    i,
                    cv2.CC_STAT_HEIGHT
                ]
            )

            area = int(
                stats[
                    i,
                    cv2.CC_STAT_AREA
                ]
            )

            # ------------------------------------------------
            # COMPONENT PIXELS
            # ------------------------------------------------

            region_mask = (
                labels == i
            )

            region_probability = (
                target_probability[
                    region_mask
                ]
            )

            mean_probability = float(
                region_probability.mean()
            )

            max_region_probability = float(
                region_probability.max()
            )

            # =================================================
            # LARGE ENOUGH -- FORMAL REGION
            # ====================

            if area >= self.min_area:

                regions.append({

                    "bbox": [
                        x,
                        y,
                        w,
                        h
                    ],

                    "area":
                        area,

                    "ml_probability":
                        mean_probability,

                    "max_ml_probability":
                        max_region_probability,

                    "evidence":
                        mean_probability,

                    "max_evidence":
                        max_region_probability,

                    "evidence_type":
                        "region"
                })

            # =========================
            # TOO SMALL -- PRESERVE AS EVIDENCE
            # =================================================

            else:

                localized_evidence_regions.append({

                    "bbox": [
                        x,
                        y,
                        w,
                        h
                    ],

                    "area":
                        area,

                    "evidence":
                        mean_probability,

                    "max_probability":
                        max_region_probability,

                    "evidence_type":
                        "localized"
                })

        # ====================================================
        # SORT BY STRENGTH
        # ==============================================

        regions.sort(
            key=lambda region:
                region["evidence"],
            reverse=True
        )

        localized_evidence_regions.sort(
            key=lambda region:
                region["max_probability"],
            reverse=True
        )

        # ====================================================
        # STATUS
        # ====================================================

        if len(regions) > 0:

            status = "DETECTED"

        elif len(
            localized_evidence_regions
        ) > 0:

            status = "EVIDENCE_PRESENT"

        else:

            status = "NO_DETECTION"

        # ====================================================
        # COMBINED EVIDENCE
        #


        overall_evidence = (
            overall_ml_evidence
        )

        max_combined_evidence = (
            max_probability
        )

        # ====================================================
        # RETURN

        return {

            "agent":
                "spectral",

            "image":
                str(lwir_path),

            "status":
                status,

            # LWIR STATISTICS
            # ------------------------------------------------

            "mean_lwir":
                float(
                    lwir.mean()
                ),

            "std_lwir":
                float(
                    lwir.std()
                ),

            # ------------------------------------------------
            # RF EVIDENCE
            # ------------------------------------------------

            "overall_ml_evidence":
                overall_ml_evidence,

            "overall_thermal_evidence":
                overall_ml_evidence,

            "overall_evidence":
                overall_evidence,

            "max_probability":
                max_probability,

            "max_combined_evidence":
                max_combined_evidence,

            # ------------------------------------------------
            # REGIONS
            # -----------------------------

            "regions":
                regions,

            "localized_evidence_regions":
                localized_evidence_regions,

            # ------------------------------------------------
            # THRESHOLDS
            # ------------------------------------------------

            "candidate_threshold":
                RF_THRESHOLD,

            "local_rf_threshold":
                RF_THRESHOLD,

            # ------------------------------------------------
            # INTERMEDIATE OUTPUTS
            # ------------------------------------------------

            "_lwir":
                lwir,

            "_normalized_image":
                normalized_image,

            "_local_mean":
                local_mean,

            "_local_std":
                local_std,

            "_local_contrast":
                local_contrast,

            "_target_probability":
                target_probability,

            "_candidate_mask":
                candidate_mask
        }

    # ========================================================
    # VISUALIZATION
    # ========================================================

    def save_visualization(
        self,
        lwir_path,
        output_path="outputs/spectral_analysis.png"
    ):
        """
        Creates a 2x2 Spectral Agent visualization:

            1. LWIR INPUT
            2. RANDOM FOREST PROBABILITY
            3. RF EVIDENCE MAP
            4. THERMAL CANDIDATE REGIONS

        Green boxes:
            Formal RF candidate regions.

        Orange boxes:
            Strong localized RF evidence that did not
            satisfy the minimum region area.
        """

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # ====================================================
        # ANALYZE
        # ====================================================

        result = self.analyze(
            lwir_path
        )

        lwir = result[
            "_lwir"
        ]

        target_probability = result[
            "_target_probability"
        ]

        candidate_mask = result[
            "_candidate_mask"
        ]

        # ====================================================
        # LWIR DISPLAY
        # ====================

        lwir_display = cv2.normalize(
            lwir,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(
            np.uint8
        )

        lwir_display = cv2.applyColorMap(
            lwir_display,
            cv2.COLORMAP_INFERNO
        )

        # ====================================================
        # RANDOM FOREST PROBABILITY
        # ====================================================

        probability_display = (
            target_probability *
            255
        ).astype(
            np.uint8
        )

        probability_display = cv2.applyColorMap(
            probability_display,
            cv2.COLORMAP_JET
        )

        # ====================================================
        # RF EVIDENCE MAP
        # ====================================================

        evidence_display = (
            target_probability *
            255
        ).astype(
            np.uint8
        )

        evidence_display = cv2.applyColorMap(
            evidence_display,
            cv2.COLORMAP_JET
        )

        # Add the threshold contour so it is visually clear
        # which RF probabilities become candidates.

        contours, _ = cv2.findContours(
            candidate_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            evidence_display,
            contours,
            -1,
            (255, 255, 255),
            1
        )

        # ====================================================
        # CANDIDATE REGION DISPLAY
        # ====================================================

        candidate_display = lwir_display.copy()

        # ----------------------------------------------------
        # Formal regions
        # ----------------------------------------------------

        for index, region in enumerate(
            result["regions"][:30],
            start=1
        ):

            x, y, w, h = (
                region["bbox"]
            )

            cv2.rectangle(
                candidate_display,
                (x, y),
                (
                    x + w,
                    y + h
                ),
                (0, 255, 0),
                2
            )

            label = (
                f"R{index} "
                f"{region['evidence']:.2f}"
            )

            cv2.putText(
                candidate_display,
                label,
                (
                    x,
                    max(
                        y - 5,
                        20
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

        # ----------------------------------------------------
        # Localized RF evidence
        # ----------------------------------------------------

        for index, region in enumerate(
            result[
                "localized_evidence_regions"
            ][:30],
            start=1
        ):

            x, y, w, h = (
                region["bbox"]
            )

            cv2.rectangle(
                candidate_display,
                (x, y),
                (
                    x + w,
                    y + h
                ),
                (0, 180, 255),
                1
            )

            label = (
                f"E{index} "
                f"{region['max_probability']:.2f}"
            )

            cv2.putText(
                candidate_display,
                label,
                (
                    x,
                    min(
                        y + h + 13,
                        candidate_display.shape[0] - 5
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 180, 255),
                1,
                cv2.LINE_AA
            )

        # ====================================================
        # TITLES
        # ====================================================

        def add_title(
            img,
            title
        ):

            canvas = img.copy()

            cv2.rectangle(
                canvas,
                (0, 0),
                (
                    canvas.shape[1],
                    34
                ),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                canvas,
                title,
                (8, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            return canvas

        lwir_display = add_title(
            lwir_display,
            "LWIR INPUT"
        )

        probability_display = add_title(
            probability_display,
            "RANDOM FOREST PROBABILITY"
        )

        evidence_display = add_title(
            evidence_display,
            "RF EVIDENCE MAP"
        )

        candidate_display = add_title(
            candidate_display,
            "THERMAL CANDIDATE REGIONS"
        )

        # ====================================================
        # BUILD 2x2 GRID
        # ====================================================

        top_row = np.hstack([
            lwir_display,
            probability_display
        ])

        bottom_row = np.hstack([
            evidence_display,
            candidate_display
        ])

        visualization = np.vstack([
            top_row,
            bottom_row
        ])

        # ====================================================
        # SAVE
        # ====================================================

        cv2.imwrite(
            str(output_path),
            visualization
        )

        print(
            f"Visualization saved to: "
            f"{output_path}"
        )

        return output_path


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    DATASET_DIR = Path(
        "MUDCAD-X/dataset"
    )

    lwir_path = next(
        DATASET_DIR.rglob(
            "lwir.png"
        )
    )

    agent = SpectralAgent()

    result = agent.analyze(
        lwir_path
    )

    print()
    print(
        "SPECTRAL AGENT"
    )
    print(
        "========================="
    )

    print(
        f"Status: "
        f"{result['status']}"
    )

    print(
        f"LWIR mean: "
        f"{result['mean_lwir']:.2f}"
    )

    print(
        f"LWIR std: "
        f"{result['std_lwir']:.2f}"
    )

    print(
        f"Random Forest mean probability: "
        f"{result['overall_ml_evidence']:.3f}"
    )

    print(
        f"Maximum RF probability: "
        f"{result['max_probability']:.3f}"
    )

    print(
        f"RF threshold: "
        f"{result['candidate_threshold']:.3f}"
    )

    print(
        f"Thermal regions: "
        f"{len(result['regions'])}"
    )

    print(
        f"Localized evidence regions: "
        f"{len(result['localized_evidence_regions'])}"
    )

    # ========================================================
    # FORMAL REGIONS
    # ========================================================

    for i, region in enumerate(
        result["regions"][:20],
        start=1
    ):

        print()
        print(
            f"Region {i}"
        )

        print(
            f"  Box: "
            f"{region['bbox']}"
        )

        print(
            f"  Area: "
            f"{region['area']}"
        )

        print(
            f"  Mean RF probability: "
            f"{region['ml_probability']:.3f}"
        )

        print(
            f"  Maximum RF probability: "
            f"{region['max_ml_probability']:.3f}"
        )

        print(
            f"  Evidence: "
            f"{region['evidence']:.3f}"
        )

    # ========================================================
    # LOCALIZED EVIDENCE
    # ========================================================

    for i, region in enumerate(
        result[
            "localized_evidence_regions"
        ][:20],
        start=1
    ):

        print()
        print(
            f"Localized Evidence {i}"
        )

        print(
            f"  Box: "
            f"{region['bbox']}"
        )

        print(
            f"  Area: "
            f"{region['area']}"
        )

        print(
            f"  Mean RF probability: "
            f"{region['evidence']:.3f}"
        )

        print(
            f"  Maximum RF probability: "
            f"{region['max_probability']:.3f}"
        )

    # ========================================================
    # VISUALIZATION
    # ========================================================

    agent.save_visualization(
        lwir_path
    )
