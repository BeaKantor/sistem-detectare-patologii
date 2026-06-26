import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, classification_report
import mlflow

from severity_dataset import get_severity_dataloaders
from severity_model import SeverityModel

DATA_DIR   = r"C:\xray-project\data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
OUTPUT_DIR = r"C:\xray-project"

SEV_LABELS = ['N/A', 'mild', 'moderate', 'severe']

EPOCHS        = 20
BATCH_SIZE    = 32
LR_HEAD       = 0.001
LR_FULL       = 0.0001
FREEZE_EPOCHS = 5
NUM_WORKERS   = 0

SEV_WEIGHTS = torch.tensor([1.0, 5.0, 5.0, 10.0], dtype=torch.float32)


def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    weights     = SEV_WEIGHTS.to(device)
    total_loss  = 0.0
    n_batches   = 0
    all_preds   = []
    all_targets = []

    for batch_idx, (images, disease_idx, severity) in enumerate(loader):
        images      = images.to(device)
        disease_idx = disease_idx.to(device)
        severity    = severity.to(device)

        optimizer.zero_grad()
        sev_out = model(images, disease_idx)
        loss    = nn.functional.cross_entropy(sev_out, severity, weight=weights)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

        all_preds.extend(sev_out.argmax(dim=1).cpu().numpy())
        all_targets.extend(severity.cpu().numpy())

        if (batch_idx + 1) % 100 == 0:
            print(
                f"  Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} | "
                f"loss: {loss.item():.4f}"
            )

    acc = accuracy_score(all_targets, all_preds)
    return total_loss / n_batches, acc


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    weights     = SEV_WEIGHTS.to(device)
    total_loss  = 0.0
    n_batches   = 0
    all_preds   = []
    all_targets = []

    for images, disease_idx, severity in loader:
        images      = images.to(device)
        disease_idx = disease_idx.to(device)
        severity    = severity.to(device)

        sev_out = model(images, disease_idx)
        loss    = nn.functional.cross_entropy(sev_out, severity, weight=weights)

        total_loss += loss.item()
        n_batches  += 1

        all_preds.extend(sev_out.argmax(dim=1).cpu().numpy())
        all_targets.extend(severity.cpu().numpy())

    acc_all = accuracy_score(all_targets, all_preds)

   
    non_na_mask      = [t != 0 for t in all_targets]
    filtered_preds   = [p for p, m in zip(all_preds,   non_na_mask) if m]
    filtered_targets = [t for t, m in zip(all_targets, non_na_mask) if m]

    if filtered_targets:
        acc_non_na = accuracy_score(filtered_targets, filtered_preds)
        f1_non_na  = f1_score(filtered_targets, filtered_preds,
                              average='weighted', zero_division=0)
    else:
        acc_non_na = 0.0
        f1_non_na  = 0.0

    return total_loss / n_batches, acc_all, acc_non_na, f1_non_na, all_preds, all_targets


def run_phase(model, train_loader, val_loader, optimizer, scheduler,
              start_epoch, end_epoch, label, device, best_acc, best_path):
    for epoch in range(start_epoch, end_epoch + 1):
        print(f"\nEpoch {epoch}/{EPOCHS} [{label}]")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, epoch
        )
        val_loss, val_acc_all, val_acc_non_na, val_f1, _, _ = evaluate(
            model, val_loader, device
        )

      
        scheduler.step(val_acc_non_na)

        print(f"  train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f}")
        print(f"  val_loss:   {val_loss:.4f} | "
              f"val_acc_all: {val_acc_all:.4f} | "
              f"val_acc_non_na: {val_acc_non_na:.4f} | "
              f"val_f1_non_na: {val_f1:.4f}")

        mlflow.log_metrics({
            "sev_train_loss":    train_loss,
            "sev_train_acc":     train_acc,
            "sev_val_loss":      val_loss,
            "sev_val_acc_all":   val_acc_all,
            "sev_val_acc_non_na": val_acc_non_na,
            "sev_val_f1_non_na": val_f1,
        }, step=epoch)

        if val_acc_non_na > best_acc:
            best_acc = val_acc_non_na
            torch.save(model.state_dict(), best_path)
            print(f"  New best non-N/A accuracy: {best_acc:.4f} — model saved.")

    return best_acc


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading severity datasets...")
    train_loader, val_loader, test_loader = get_severity_dataloaders(
        data_dir    = DATA_DIR,
        images_dir  = IMAGES_DIR,
        batch_size  = BATCH_SIZE,
        num_workers = NUM_WORKERS,
    )

    print("\nBuilding severity model...")
    model = SeverityModel(pretrained=True)
    model.to(device)

    best_acc  = 0.0
    best_path = os.path.join(OUTPUT_DIR, "severity_model.pth")

    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(OUTPUT_DIR, 'mlflow.db')}")
    mlflow.set_experiment("xray_severity_classification")

    with mlflow.start_run(run_name="severity_model_v2"):

        mlflow.log_params({
            "epochs":        EPOCHS,
            "batch_size":    BATCH_SIZE,
            "lr_head":       LR_HEAD,
            "lr_full":       LR_FULL,
            "freeze_epochs": FREEZE_EPOCHS,
            "backbone":      "densenet121",
            "sev_weights":   "1.0/5.0/5.0/10.0",
            "val_metric":    "accuracy_non_na",
            "input":         "1024 features + 13 one-hot disease",
        })

        
        print(f"\nPhase 1: Freezing backbone (epochs 1-{FREEZE_EPOCHS})")
        for param in model.features.parameters():
            param.requires_grad = False

        optimizer = Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR_HEAD,
        )
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

        best_acc = run_phase(
            model, train_loader, val_loader, optimizer, scheduler,
            1, FREEZE_EPOCHS, "HEADS ONLY", device, best_acc, best_path,
        )

        
        print(f"\nPhase 2: Full network unfrozen (epochs {FREEZE_EPOCHS+1}-{EPOCHS})")
        for param in model.features.parameters():
            param.requires_grad = True

        optimizer = Adam(model.parameters(), lr=LR_FULL)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

        best_acc = run_phase(
            model, train_loader, val_loader, optimizer, scheduler,
            FREEZE_EPOCHS + 1, EPOCHS, "FULL NETWORK", device, best_acc, best_path,
        )

    
        print("\nFinal evaluation on test set...")
        model.load_state_dict(torch.load(best_path, map_location=device))
        test_loss, test_acc_all, test_acc_non_na, test_f1, test_preds, test_targets = evaluate(
            model, test_loader, device
        )

    
        print(f"Severity Model - Test Set Results")
        print(f"  Test accuracy (all)    : {test_acc_all:.4f}")
        print(f"  Test accuracy (non-N/A): {test_acc_non_na:.4f}")
        print(f"  Test F1 (non-N/A)      : {test_f1:.4f}")
        print(f"\nPer-class report:")
        print(classification_report(
            test_targets, test_preds,
            target_names=SEV_LABELS,
            zero_division=0,
        ))

        mlflow.log_metrics({
            "sev_test_acc_all":    test_acc_all,
            "sev_test_acc_non_na": test_acc_non_na,
            "sev_test_f1_non_na":  test_f1,
        })
        mlflow.log_artifact(best_path)

        print(f"  Model saved -> {best_path}")
        