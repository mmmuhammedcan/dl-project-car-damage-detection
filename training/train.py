"""
Train the vehicle-damage ResNet50 classifier with a correct transfer-learning head.

Fixes a bug in the original notebook: the classification head was assigned to
`self.model.classifier`, an attribute ResNet50 doesn't use in its forward pass
(that's an EfficientNet convention). ResNet50's forward always routes through
`self.model.fc`, so the real head must be assigned there.

Produces a held-out TEST set (never touched during training/model selection) and
reports accuracy, macro-F1, per-class precision/recall, and a confusion matrix.
"""
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

SEED = 42
DATASET_PATH = "../dataset"  # ImageFolder layout: dataset/<class_name>/*.jpg — not included in this repo (~640MB)
BATCH_SIZE = 32
EPOCHS = 15
LR = 0.005
DROPOUT = 0.2
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15  # remainder -> test

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

base_dataset = datasets.ImageFolder(root=DATASET_PATH)
class_names = base_dataset.classes
num_classes = len(class_names)
print("classes:", class_names)

# Stratified split by index, done manually (no extra dependency needed beyond sklearn already used below).
targets = np.array(base_dataset.targets)
rng = np.random.RandomState(SEED)
train_idx, val_idx, test_idx = [], [], []
for c in range(num_classes):
    idx = np.where(targets == c)[0]
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(round(n * TRAIN_FRAC))
    n_val = int(round(n * VAL_FRAC))
    train_idx.extend(idx[:n_train])
    val_idx.extend(idx[n_train:n_train + n_val])
    test_idx.extend(idx[n_train + n_val:])

train_ds = Subset(datasets.ImageFolder(root=DATASET_PATH, transform=train_transform), train_idx)
val_ds = Subset(datasets.ImageFolder(root=DATASET_PATH, transform=eval_transform), val_idx)
test_ds = Subset(datasets.ImageFolder(root=DATASET_PATH, transform=eval_transform), test_idx)
print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.2):
        super().__init__()
        self.model = models.resnet50(weights="DEFAULT")
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes),
        )
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)


def evaluate(model, loader):
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
    return all_labels, all_preds


def main():
    model = CarClassifierResNet(num_classes, dropout_rate=DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    best_val_f1 = -1.0
    best_state = None
    start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        val_labels, val_preds = evaluate(model, val_loader)
        val_f1 = f1_score(val_labels, val_preds, average="macro")
        val_acc = np.mean(np.array(val_labels) == np.array(val_preds))
        print(f"epoch {epoch+1}/{EPOCHS} loss={epoch_loss:.4f} val_acc={val_acc:.4f} val_macro_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"training time: {time.time()-start:.1f}s, best val macro-F1: {best_val_f1:.4f}")

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), "../model/saved_model.pth")

    test_labels, test_preds = evaluate(model, test_loader)
    report = classification_report(test_labels, test_preds, target_names=class_names, digits=3)
    cm = confusion_matrix(test_labels, test_preds, labels=list(range(num_classes)))
    test_acc = float(np.mean(np.array(test_labels) == np.array(test_preds)))
    test_macro_f1 = float(f1_score(test_labels, test_preds, average="macro"))

    print("\n=== TEST SET RESULTS (held out, never used for training/model selection) ===")
    print(report)
    print("confusion matrix (rows=true, cols=pred):")
    print(class_names)
    print(cm)

    results = {
        "class_names": class_names,
        "split_sizes": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        "test_accuracy": test_acc,
        "test_macro_f1": test_macro_f1,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "best_val_macro_f1": best_val_f1,
        "hyperparameters": {"lr": LR, "dropout": DROPOUT, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "seed": SEED},
    }
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: ../model/saved_model.pth, eval_results.json")


if __name__ == "__main__":
    main()
