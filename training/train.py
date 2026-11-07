import json, os, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

DATA_DIR = os.environ.get("DATA_DIR", "PlantVillage-Dataset/raw/color")
EPOCHS = int(os.environ.get("EPOCHS", 3))
BATCH_SIZE = 64
LR = 1e-3
PRETRAINED = os.environ.get("PRETRAINED", "1") == "1"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Appareil :", device)

# Même prétraitement que l'API (api/app/model.py) : resize 224x224 + normalisation ImageNet
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(), normalize,
])
test_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

# Deux vues du même dossier (transformations différentes), découpées 80 % / 20 %
train_full = datasets.ImageFolder(DATA_DIR, transform=train_tf)
test_full = datasets.ImageFolder(DATA_DIR, transform=test_tf)
classes = train_full.classes
idx = np.random.default_rng(42).permutation(len(train_full))
split = int(0.8 * len(idx))
train_ds, test_ds = Subset(train_full, idx[:split]), Subset(test_full, idx[split:])
print(f"{len(classes)} classes | {len(train_ds)} images d'entraînement | {len(test_ds)} de test")

workers = 2 if device == "cuda" else 0
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=workers, pin_memory=True)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=workers, pin_memory=True)

weights = models.MobileNet_V3_Large_Weights.DEFAULT if PRETRAINED else None
model = models.mobilenet_v3_large(weights=weights)
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(classes))
model = model.to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
loss_fn = nn.CrossEntropyLoss()


def evaluate():
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_dl:
            y_pred += model(x.to(device)).argmax(1).cpu().tolist()
            y_true += y.tolist()
    return np.array(y_true), np.array(y_pred)


history = []
t_start = time.time()
for epoch in range(EPOCHS):
    model.train()
    t0, total_loss = time.time(), 0.0
    for i, (x, y) in enumerate(train_dl):
        opt.zero_grad()
        loss = loss_fn(model(x.to(device)), y.to(device))
        loss.backward()
        opt.step()
        total_loss += loss.item()
        if i % 100 == 0:
            print(f"  epoch {epoch+1} | batch {i}/{len(train_dl)} | loss {loss.item():.4f}")
    sched.step()
    y_true, y_pred = evaluate()
    acc = float((y_true == y_pred).mean())
    history.append({"epoch": epoch + 1, "train_loss": total_loss / len(train_dl), "test_accuracy": acc})
    print(f"Epoch {epoch+1}/{EPOCHS} | loss {total_loss/len(train_dl):.4f} | test accuracy {acc:.4f} | {time.time()-t0:.0f}s")

f1 = float(f1_score(y_true, y_pred, average="macro"))
print(f"\nAccuracy finale : {acc:.4f} | F1 macro : {f1:.4f} | durée totale : {(time.time()-t_start)/60:.1f} min")
print(classification_report(y_true, y_pred, labels=range(len(classes)), target_names=classes, digits=3, zero_division=0))

# Export ONNX : mêmes noms d'entrée/sortie que le modèle factice
out = Path("export")
out.mkdir(exist_ok=True)
model.eval().cpu()
torch.onnx.export(model, torch.randn(1, 3, 224, 224), out / "model.onnx",
                  input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "N"}, "logits": {0: "N"}},
                  opset_version=17, dynamo=False)
(out / "classes.json").write_text(json.dumps(classes, indent=2))
(out / "metrics.json").write_text(json.dumps({
    "arch": "mobilenet_v3_large", "pretrained": PRETRAINED, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
    "lr": LR, "n_train": len(train_ds), "n_test": len(test_ds), "test_accuracy": acc,
    "f1_macro": f1, "history": history}, indent=2))
print("Fichiers exportés :", [p.name for p in out.iterdir()],
      f"| model.onnx = {(out/'model.onnx').stat().st_size/1e6:.1f} Mo")
