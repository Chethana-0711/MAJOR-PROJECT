from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

DATA_DIR = Path("data/cnn")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "multimodal_cnn.pt"

IMAGE_SIZE = 64
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.001

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------
# DATASET
# --------------------------------------------------

class MultimodalDataset(Dataset):

    def __init__(self, split):

        self.samples = []

        split_dir = DATA_DIR / split

        for class_name, label in [
            ("target", 1),
            ("background", 0)
        ]:

            target_dir = split_dir / class_name

            rgb_files = sorted(
                target_dir.glob("*_rgb.jpg")
            )

            for rgb_path in rgb_files:

                lwir_path = Path(
                    str(rgb_path).replace(
                        "_rgb.jpg",
                        "_lwir.jpg"
                    )
                )

                if lwir_path.exists():

                    self.samples.append(
                        (
                            rgb_path,
                            lwir_path,
                            label
                        )
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        rgb_path, lwir_path, label = (
            self.samples[index]
        )

        # RGB
        rgb = cv2.imread(
            str(rgb_path)
        )

        rgb = cv2.cvtColor(
            rgb,
            cv2.COLOR_BGR2RGB
        )

        rgb = cv2.resize(
            rgb,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        rgb = (
            rgb.astype(np.float32) /
            255.0
        )

        # LWIR
        lwir = cv2.imread(
            str(lwir_path),
            cv2.IMREAD_GRAYSCALE
        )

        lwir = cv2.resize(
            lwir,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        lwir = (
            lwir.astype(np.float32) /
            255.0
        )

        # Convert to PyTorch format
        rgb = torch.tensor(
            rgb
        ).permute(2, 0, 1)

        lwir = torch.tensor(
            lwir
        ).unsqueeze(0)

        return (
            rgb,
            lwir,
            torch.tensor(
                label,
                dtype=torch.float32
            )
        )


# --------------------------------------------------
# CNN BRANCH
# --------------------------------------------------

class CNNBranch(nn.Module):

    def __init__(self, input_channels):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                input_channels,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

    def forward(self, x):

        x = self.features(x)

        return x.flatten(1)


# --------------------------------------------------
# MULTIMODAL CNN
# --------------------------------------------------

class MultimodalCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.rgb_branch = CNNBranch(3)

        self.lwir_branch = CNNBranch(1)

        self.classifier = nn.Sequential(

            nn.Linear(
                128 + 128,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                1
            )
        )

    def forward(
        self,
        rgb,
        lwir
    ):

        rgb_features = (
            self.rgb_branch(rgb)
        )

        lwir_features = (
            self.lwir_branch(lwir)
        )

        combined = torch.cat(
            [
                rgb_features,
                lwir_features
            ],
            dim=1
        )

        return self.classifier(
            combined
        ).squeeze(1)


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

def evaluate(
    model,
    loader
):

    model.eval()

    correct = 0
    total = 0

    true_positive = 0
    false_positive = 0
    false_negative = 0

    with torch.no_grad():

        for rgb, lwir, labels in loader:

            rgb = rgb.to(DEVICE)
            lwir = lwir.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(
                rgb,
                lwir
            )

            probabilities = torch.sigmoid(
                outputs
            )

            predictions = (
                probabilities >= 0.5
            ).float()

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.numel()

            true_positive += (
                ((predictions == 1) &
                 (labels == 1))
            ).sum().item()

            false_positive += (
                ((predictions == 1) &
                 (labels == 0))
            ).sum().item()

            false_negative += (
                ((predictions == 0) &
                 (labels == 1))
            ).sum().item()

    accuracy = correct / total

    precision = (
        true_positive /
        (true_positive + false_positive + 1e-8)
    )

    recall = (
        true_positive /
        (true_positive + false_negative + 1e-8)
    )

    f1 = (
        2 * precision * recall /
        (precision + recall + 1e-8)
    )

    return accuracy, precision, recall, f1


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print(
        f"Using device: {DEVICE}"
    )

    print(
        "\nLoading datasets..."
    )

    train_dataset = MultimodalDataset(
        "train"
    )

    val_dataset = MultimodalDataset(
        "val"
    )

    test_dataset = MultimodalDataset(
        "test"
    )

    print(
        f"Train samples: {len(train_dataset)}"
    )

    print(
        f"Val samples:   {len(val_dataset)}"
    )

    print(
        f"Test samples:  {len(test_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------
    # MODEL
    # --------------------------------------------------

    model = MultimodalCNN().to(
        DEVICE
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_f1 = 0.0

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------
    # TRAIN
    # --------------------------------------------------

    print(
        "\nStarting training..."
    )

    for epoch in range(
        EPOCHS
    ):

        model.train()

        running_loss = 0.0

        for rgb, lwir, labels in train_loader:

            rgb = rgb.to(DEVICE)
            lwir = lwir.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(
                rgb,
                lwir
            )

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item() *
                labels.size(0)
            )

        train_loss = (
            running_loss /
            len(train_dataset)
        )

        (
            val_accuracy,
            val_precision,
            val_recall,
            val_f1
        ) = evaluate(
            model,
            val_loader
        )

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
        )

        print(
            f"Train loss: {train_loss:.4f}"
        )

        print(
            f"Val accuracy:  {val_accuracy:.4f}"
        )

        print(
            f"Val precision: {val_precision:.4f}"
        )

        print(
            f"Val recall:    {val_recall:.4f}"
        )

        print(
            f"Val F1:        {val_f1:.4f}"
        )

        # Save best model
        if val_f1 > best_val_f1:

            best_val_f1 = val_f1

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print(
                "✓ Best model saved"
            )

    # --------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------

    print(
        "\nLoading best model..."
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    (
        test_accuracy,
        test_precision,
        test_recall,
        test_f1
    ) = evaluate(
        model,
        test_loader
    )

    print(
        "\n"
        + "=" * 55
    )

    print(
        "MULTIMODAL CNN RESULTS"
    )

    print(
        "=" * 55
    )

    print(
        f"Test accuracy:  {test_accuracy:.4f}"
    )

    print(
        f"Test precision: {test_precision:.4f}"
    )

    print(
        f"Test recall:    {test_recall:.4f}"
    )

    print(
        f"Test F1:        {test_f1:.4f}"
    )

    print(
        f"\nModel saved to: {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()