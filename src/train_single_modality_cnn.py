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

IMAGE_SIZE = 64
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.001

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------
# DATASET
# --------------------------------------------------

class SingleModalityDataset(Dataset):

    def __init__(self, split, modality):

        self.samples = []
        self.modality = modality

        split_dir = DATA_DIR / split

        for class_name, label in [
            ("target", 1),
            ("background", 0)
        ]:

            class_dir = split_dir / class_name

            files = sorted(
                class_dir.glob(
                    f"*_{modality}.jpg"
                )
            )

            for file_path in files:

                self.samples.append(
                    (
                        file_path,
                        label
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        path, label = self.samples[index]

        if self.modality == "rgb":

            image = cv2.imread(
                str(path)
            )

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

        else:

            image = cv2.imread(
                str(path),
                cv2.IMREAD_GRAYSCALE
            )

        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        image = (
            image.astype(np.float32) /
            255.0
        )

        if self.modality == "rgb":

            image = torch.tensor(
                image
            ).permute(2, 0, 1)

        else:

            image = torch.tensor(
                image
            ).unsqueeze(0)

        return (
            image,
            torch.tensor(
                label,
                dtype=torch.float32
            )
        )


# --------------------------------------------------
# CNN
# --------------------------------------------------

class FeatureCNN(nn.Module):

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

        # Convert 128 CNN features into
        # one target/background prediction
        self.classifier = nn.Linear(
            128,
            1
        )

    def forward(self, x):

        x = self.features(x)

        x = x.flatten(1)

        x = self.classifier(x)

        return x.squeeze(1)

# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

def evaluate(model, loader):

    model.eval()

    correct = 0
    total = 0

    tp = 0
    fp = 0
    fn = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

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

            tp += (
                ((predictions == 1) &
                 (labels == 1))
            ).sum().item()

            fp += (
                ((predictions == 1) &
                 (labels == 0))
            ).sum().item()

            fn += (
                ((predictions == 0) &
                 (labels == 1))
            ).sum().item()

    accuracy = correct / total

    precision = (
        tp /
        (tp + fp + 1e-8)
    )

    recall = (
        tp /
        (tp + fn + 1e-8)
    )

    f1 = (
        2 * precision * recall /
        (precision + recall + 1e-8)
    )

    return (
        accuracy,
        precision,
        recall,
        f1
    )


# --------------------------------------------------
# TRAIN ONE MODEL
# --------------------------------------------------

def train_model(modality):

    print("\n")
    print("=" * 55)
    print(
        f"TRAINING {modality.upper()} CNN"
    )
    print("=" * 55)

    train_dataset = SingleModalityDataset(
        "train",
        modality
    )

    val_dataset = SingleModalityDataset(
        "val",
        modality
    )

    test_dataset = SingleModalityDataset(
        "test",
        modality
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

    channels = (
        3 if modality == "rgb"
        else 1
    )

    model = FeatureCNN(
        channels
    ).to(DEVICE)

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

    model_path = (
        MODEL_DIR /
        f"{modality}_cnn.pt"
    )

    # --------------------------------------------------
    # TRAIN
    # --------------------------------------------------

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(
                images
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

        if val_f1 > best_val_f1:

            best_val_f1 = val_f1

            torch.save(
                model.state_dict(),
                model_path
            )

            print(
                "✓ Best model saved"
            )

    # --------------------------------------------------
    # TEST
    # --------------------------------------------------

    model.load_state_dict(
        torch.load(
            model_path,
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
        "\nFINAL TEST RESULTS"
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
        f"Model saved to: {model_path}"
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    print(
        f"Using device: {DEVICE}"
    )

    # RGB-only CNN
    train_model("rgb")

    # Thermal-only CNN
    train_model("lwir")