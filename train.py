

import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, f1_score
import mlflow
import mlflow.pytorch

from dataset import ChestXrayDataset, get_dataloaders
from model import XrayModel

DATA_DIR   = r"C:\xray-project\data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
OUTPUT_DIR = r"C:\xray-project"

DISEASES = [
    'enlarged cardiomediastinum', 'cardiomegaly', 'atelectasis',
    'consolidation', 'lung edema', 'fracture', 'lung lesion',
    'pleural effusion', 'pneumonia', 'pneumothorax',
    'support device', 'lung opacity', 'pleural other'
]

NUM_CLASSES  = 13
NUM_SEVERITY = 4

EPOCHS        = 30
BATCH_SIZE    = 32
LR_HEAD       = 0.001
LR_FULL       = 0.0001
FREEZE_EPOCHS = 5
NUM_WORKERS   = 0
IMAGE_SIZE    = 224

RESUME_EPOCH = 1
RESUME_PATH  = r"C:\xray-project\xray_model.pth"

DISEASE_RATES = {
    'enlarged cardiomediastinum': 0.035,
    'cardiomegaly':               0.032,
    'atelectasis':                0.101,
    'consolidation':              0.091,
    'lung edema':                 0.013,
    'fracture':                   0.020,
    'lung lesion':                0.104,
    'pleural effusion':           0.090,
    'pneumonia':                  0.018,
    'pneumothorax':               0.034,
    'support device':             0.494,
    'lung opacity':               0.346,
    'pleural other':              0.042,
}

SEVERITY_COUNTS = {
    0: 1007396,
    1: 7652,
    2: 5071,
    3: 459,
}


def compute_prob_loss(pred, target, device):
    mask         = ~torch.isnan(target)
    target_clean = torch.where(mask, target, torch.zeros_like(target))
    weights      = torch.tensor(
        [(1 - r) / r for r in DISEASE_RATES.values()],
        dtype=torch.float32, device=device,
    ).unsqueeze(0).expand_as(pred)
    bce         = nn.functional.binary_cross_entropy(pred, target_clean, reduction='none')
    masked_loss = bce * weights * mask.float()
    n_known     = mask.float().sum()
    if n_known == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return masked_loss.sum() / n_known


def compute_severity_loss(pred, target, labels, device):
    valid_mask = (labels == 1.0) & ~torch.isnan(target)
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    total_sev   = sum(SEVERITY_COUNTS.values())
    sev_weights = torch.tensor(
        [total_sev / (4 * SEVERITY_COUNTS[i]) for i in range(NUM_SEVERITY)],
        dtype=torch.float32, device=device,
    )
    pred_flat    = pred.view(-1, NUM_SEVERITY)
    target_flat  = target.view(-1)
    mask_flat    = valid_mask.view(-1)
    target_clean = torch.where(
        torch.isnan(target_flat),
        torch.zeros_like(target_flat),
        target_flat,
    ).long()
    ce        = nn.functional.cross_entropy(pred_flat, target_clean, weight=sev_weights, reduction='none')
    masked_ce = ce * mask_flat.float()
    return masked_ce.sum() / mask_flat.float().sum()


def compute_metrics(all_probs, all_labels, all_sev_pred, all_sev_true):
    metrics    = {}
    auc_scores = []
    f1_scores  = []
    for i, disease in enumerate(DISEASES):
        mask = ~np.isnan(all_labels[:, i])
        if mask.sum() < 2:
            continue
        y_true = all_labels[mask, i]
        y_pred = all_probs[mask, i]
        if len(np.unique(y_true)) == 2:
            auc = roc_auc_score(y_true, y_pred)
            metrics[f"auc_{disease.replace(' ', '_')}"] = auc
            auc_scores.append(auc)
        y_binary = (y_pred >= 0.5).astype(int)
        f1 = f1_score(y_true, y_binary, zero_division=0)
        metrics[f"f1_{disease.replace(' ', '_')}"] = f1
        f1_scores.append(f1)
    metrics["auc_average"] = np.mean(auc_scores) if auc_scores else 0.0
    metrics["f1_average"]  = np.mean(f1_scores)  if f1_scores  else 0.0
    sev_correct = 0
    sev_total   = 0
    for i in range(NUM_CLASSES):
        mask = (~np.isnan(all_sev_true[:, i])) & (all_labels[:, i] == 1.0)
        if mask.sum() == 0:
            continue
        sev_correct += (all_sev_pred[mask, i] == all_sev_true[mask, i]).sum()
        sev_total   += mask.sum()
    metrics["severity_accuracy"] = sev_correct / sev_total if sev_total > 0 else 0.0
    return metrics


def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_prob_loss = 0.0
    total_sev_loss  = 0.0
    n_batches       = 0
    for batch_idx, (images, labels, probability, severity, _) in enumerate(loader):
        images      = images.to(device)
        labels      = labels.to(device)
        probability = probability.to(device)
        severity    = severity.to(device)
        optimizer.zero_grad()
        prob_out, sev_out = model(images)
        loss_prob = compute_prob_loss(prob_out, probability, device)
        loss_sev  = compute_severity_loss(sev_out, severity, labels, device)
        loss      = loss_prob + loss_sev
        loss.backward()
        optimizer.step()
        total_prob_loss += loss_prob.item()
        total_sev_loss  += loss_sev.item()
        n_batches       += 1
        if (batch_idx + 1) % 100 == 0:
            print(
                f"  Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} | "
                f"prob_loss: {loss_prob.item():.4f} | "
                f"sev_loss: {loss_sev.item():.4f}"
            )
    return total_prob_loss / n_batches, total_sev_loss / n_batches


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_prob_loss = 0.0
    total_sev_loss  = 0.0
    n_batches       = 0
    all_probs    = []
    all_labels   = []
    all_sev_pred = []
    all_sev_true = []
    for images, labels, probability, severity, _ in loader:
        images      = images.to(device)
        labels      = labels.to(device)
        probability = probability.to(device)
        severity    = severity.to(device)
        prob_out, sev_out = model(images)
        loss_prob = compute_prob_loss(prob_out, probability, device)
        loss_sev  = compute_severity_loss(sev_out, severity, labels, device)
        total_prob_loss += loss_prob.item()
        total_sev_loss  += loss_sev.item()
        n_batches       += 1
        all_probs.append(prob_out.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_sev_pred.append(sev_out.argmax(dim=2).cpu().numpy())
        all_sev_true.append(severity.cpu().numpy())
    all_probs    = np.concatenate(all_probs,    axis=0)
    all_labels   = np.concatenate(all_labels,   axis=0)
    all_sev_pred = np.concatenate(all_sev_pred, axis=0)
    all_sev_true = np.concatenate(all_sev_true, axis=0)
    metrics = compute_metrics(all_probs, all_labels, all_sev_pred, all_sev_true)
    return total_prob_loss / n_batches, total_sev_loss / n_batches, metrics


def run_phase(model, train_loader, val_loader, optimizer, scheduler,
              start_epoch, end_epoch, label, device, best_auc, best_path):
    for epoch in range(start_epoch, end_epoch + 1):
        print(f"\nEpoch {epoch}/{EPOCHS} [{label}]")
        train_prob_loss, train_sev_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch
        )
        val_prob_loss, val_sev_loss, metrics = evaluate(model, val_loader, device)
        val_auc = metrics.get("auc_average", 0.0)
        val_f1  = metrics.get("f1_average",  0.0)
        val_sev = metrics.get("severity_accuracy", 0.0)
        scheduler.step(val_auc)
        print(f"  train_prob_loss: {train_prob_loss:.4f} | train_sev_loss: {train_sev_loss:.4f}")
        print(f"  val_prob_loss:   {val_prob_loss:.4f} | val_sev_loss:   {val_sev_loss:.4f}")
        print(f"  val_auc: {val_auc:.4f} | val_f1: {val_f1:.4f} | val_sev_acc: {val_sev:.4f}")
        mlflow.log_metrics({
            "train_prob_loss":  train_prob_loss,
            "train_sev_loss":   train_sev_loss,
            "val_prob_loss":    val_prob_loss,
            "val_sev_loss":     val_sev_loss,
            "val_auc_average":  val_auc,
            "val_f1_average":   val_f1,
            "val_sev_accuracy": val_sev,
        }, step=epoch)
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_path)
            print(f"   New best AUC: {best_auc:.4f} — model saved.")
    return best_auc


if __name__ == "__main__":

    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(OUTPUT_DIR, 'mlflow.db')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, _ = get_dataloaders(
        data_dir    = DATA_DIR,
        images_dir  = IMAGES_DIR,
        batch_size  = BATCH_SIZE,
        num_workers = NUM_WORKERS,
    )

    model = XrayModel(pretrained=True)
    model.to(device)

    best_auc  = 0.0
    best_path = os.path.join(OUTPUT_DIR, "xray_model.pth")

    if RESUME_EPOCH > 1 and os.path.exists(RESUME_PATH):
        print(f"Resuming from epoch {RESUME_EPOCH}, loading {RESUME_PATH}")
        model.load_state_dict(torch.load(RESUME_PATH, map_location=device))
        _, _, metrics = evaluate(model, val_loader, device)
        best_auc = metrics.get("auc_average", 0.0)
        print(f"Checkpoint val AUC: {best_auc:.4f}")

    mlflow.set_experiment("xray_disease_detection")

    with mlflow.start_run(run_name=f"resume_epoch_{RESUME_EPOCH}"):

        mlflow.log_params({
            "epochs":            EPOCHS,
            "batch_size":        BATCH_SIZE,
            "lr_head":           LR_HEAD,
            "lr_full":           LR_FULL,
            "freeze_epochs":     FREEZE_EPOCHS,
            "image_size":        IMAGE_SIZE,
            "resume_epoch":      RESUME_EPOCH,
            "backbone":          "densenet121",
            "pretrained":        True,
            "oversample_factor": 3.0,
        })

        if RESUME_EPOCH <= FREEZE_EPOCHS:
            for param in model.features.parameters():
                param.requires_grad = False
            optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD)
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
            best_auc  = run_phase(
                model, train_loader, val_loader, optimizer, scheduler,
                RESUME_EPOCH, FREEZE_EPOCHS, "HEADS ONLY", device, best_auc, best_path,
            )

        for param in model.features.parameters():
            param.requires_grad = True
        optimizer = Adam(model.parameters(), lr=LR_FULL)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        start_epoch = max(RESUME_EPOCH, FREEZE_EPOCHS + 1)
        best_auc = run_phase(
            model, train_loader, val_loader, optimizer, scheduler,
            start_epoch, EPOCHS, "FULL NETWORK", device, best_auc, best_path,
        )

        mlflow.log_artifact(best_path)
        mlflow.log_param("best_val_auc", best_auc)

      
        print(f"Training complete.")
        print(f"  Best val AUC : {best_auc:.4f}")
        print(f"  Model saved  : {best_path}")
        print(f"  MLflow UI    : run 'mlflow ui' then open http://localhost:5000")
      