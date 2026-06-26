
import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    f1_score, precision_score, recall_score,
)

from dataset import get_dataloaders
from model import load_model

DATA_DIR   = r"C:\xray-project\data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
OUTPUT_DIR = r"C:\xray-project\results"
MODEL_PATH = r"C:\xray-project\xray_model.pth"

DISEASES = [
    'enlarged cardiomediastinum', 'cardiomegaly', 'atelectasis',
    'consolidation', 'lung edema', 'fracture', 'lung lesion',
    'pleural effusion', 'pneumonia', 'pneumothorax',
    'support device', 'lung opacity', 'pleural other'
]

NUM_CLASSES  = 13
NUM_SEVERITY = 4
BATCH_SIZE   = 32
NUM_WORKERS  = 0


def collect_predictions(model, loader, device):
    model.eval()
    all_probs    = []
    all_labels   = []
    all_sev_pred = []
    all_sev_true = []

    with torch.no_grad():
        for images, labels, probability, severity, _ in loader:
            images = images.to(device)
            prob_out, sev_out = model(images)
            all_probs.append(prob_out.cpu().numpy())
            all_labels.append(labels.numpy())
            all_sev_pred.append(sev_out.argmax(dim=2).cpu().numpy())
            all_sev_true.append(severity.numpy())

    return (
        np.concatenate(all_probs,    axis=0),
        np.concatenate(all_labels,   axis=0),
        np.concatenate(all_sev_pred, axis=0),
        np.concatenate(all_sev_true, axis=0),
    )


def compute_disease_metrics(all_probs, all_labels):
    results  = []
    roc_data = {}

    for i, disease in enumerate(DISEASES):
        mask = ~np.isnan(all_labels[:, i])
        if mask.sum() < 2:
            continue

        y_true   = all_labels[mask, i]
        y_pred   = all_probs[mask, i]
        y_binary = (y_pred >= 0.5).astype(int)

        auc  = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) == 2 else None
        f1   = f1_score(y_true, y_binary, zero_division=0)
        prec = precision_score(y_true, y_binary, zero_division=0)
        rec  = recall_score(y_true, y_binary, zero_division=0)

        results.append({
            'disease':   disease,
            'auc_roc':   round(auc, 4) if auc is not None else None,
            'f1':        round(f1,   4),
            'precision': round(prec, 4),
            'recall':    round(rec,  4),
            'n_present': int((y_true == 1).sum()),
            'n_absent':  int((y_true == 0).sum()),
        })

        if auc is not None:
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_data[disease] = (fpr, tpr, auc)

    return results, roc_data


def compute_severity_metrics(all_sev_pred, all_sev_true, all_labels):
    sev_results = []
    for i, disease in enumerate(DISEASES):
        mask = (~np.isnan(all_sev_true[:, i])) & (all_labels[:, i] == 1.0)
        if mask.sum() == 0:
            sev_results.append({'disease': disease, 'severity_accuracy': None, 'n_samples': 0})
            continue
        correct  = (all_sev_pred[mask, i] == all_sev_true[mask, i]).sum()
        accuracy = correct / mask.sum()
        sev_results.append({
            'disease':           disease,
            'severity_accuracy': round(float(accuracy), 4),
            'n_samples':         int(mask.sum()),
        })
    return sev_results


def plot_roc_curves(roc_data, out_path):
    n_cols = 3
    n_rows = int(np.ceil(len(roc_data) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()

    for idx, (disease, (fpr, tpr, auc)) in enumerate(roc_data.items()):
        ax = axes[idx]
        ax.plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC = {auc:.3f}')
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(disease.title(), fontsize=10)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(alpha=0.3)

    for idx in range(len(roc_data), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('ROC Curves — Test Set', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")


def plot_severity_accuracy(sev_results, out_path):
    diseases = [r['disease'] for r in sev_results if r['severity_accuracy'] is not None]
    accs     = [r['severity_accuracy'] for r in sev_results if r['severity_accuracy'] is not None]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(diseases, accs, color='steelblue', alpha=0.8)
    ax.set_xlabel('Severity Accuracy')
    ax.set_title('Severity Classification Accuracy per Disease\n(evaluated on confirmed present cases only)')
    ax.set_xlim([0, 1])
    ax.axvline(x=np.mean(accs), color='red', linestyle='--', label=f'Mean: {np.mean(accs):.3f}')
    ax.legend()
    for bar, acc in zip(bars, accs):
        ax.text(acc + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{acc:.3f}', va='center', fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")


def plot_summary_bar(results, out_path):
    diseases = [r['disease'] for r in results if r['auc_roc'] is not None]
    aucs     = [r['auc_roc'] for r in results if r['auc_roc'] is not None]
    f1s      = [r['f1']     for r in results if r['auc_roc'] is not None]

    x   = np.arange(len(diseases))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - w/2, aucs, w, label='AUC-ROC', color='steelblue', alpha=0.8)
    ax.bar(x + w/2, f1s,  w, label='F1 Score', color='coral',    alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(diseases, rotation=45, ha='right', fontsize=9)
    ax.set_ylim([0, 1])
    ax.set_ylabel('Score')
    ax.set_title('AUC-ROC and F1 Score per Disease — Test Set')
    ax.axhline(y=np.mean(aucs), color='steelblue', linestyle='--', alpha=0.5, label=f'Mean AUC: {np.mean(aucs):.3f}')
    ax.axhline(y=np.mean(f1s),  color='coral',     linestyle='--', alpha=0.5, label=f'Mean F1: {np.mean(f1s):.3f}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")


if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading test set...")
    _, _, test_loader = get_dataloaders(
        data_dir    = DATA_DIR,
        images_dir  = IMAGES_DIR,
        batch_size  = BATCH_SIZE,
        num_workers = NUM_WORKERS,
    )
    print(f"  Test batches: {len(test_loader)}")

    print("Loading model...")
    model = load_model(MODEL_PATH, device)

    print("Running inference on test set...")
    all_probs, all_labels, all_sev_pred, all_sev_true = collect_predictions(
        model, test_loader, device
    )
    print(f"  Collected predictions for {len(all_probs):,} images")

    print("\nComputing disease metrics...")
    results, roc_data = compute_disease_metrics(all_probs, all_labels)

    print("Computing severity metrics...")
    sev_results = compute_severity_metrics(all_sev_pred, all_sev_true, all_labels)

    
   
    print(f"TEST SET RESULTS")
  
    print(f"  {'Disease':<30} {'AUC-ROC':>8} {'F1':>8} {'Precision':>10} {'Recall':>8}")
    
    for r in results:
        auc = f"{r['auc_roc']:.4f}" if r['auc_roc'] is not None else "   N/A"
        print(f"  {r['disease']:<30} {auc:>8} {r['f1']:>8.4f} {r['precision']:>10.4f} {r['recall']:>8.4f}")

    valid_aucs = [r['auc_roc'] for r in results if r['auc_roc'] is not None]
    valid_f1s  = [r['f1']      for r in results]
    valid_prec = [r['precision'] for r in results]
    valid_rec  = [r['recall']    for r in results]

   
    print(f"  {'AVERAGE':<30} {np.mean(valid_aucs):>8.4f} {np.mean(valid_f1s):>8.4f} {np.mean(valid_prec):>10.4f} {np.mean(valid_rec):>8.4f}")
  

    print(f"\nSeverity Accuracy:")
    print(f"  {'Disease':<30} {'Accuracy':>10} {'N Samples':>10}")
  
    valid_sev_accs = []
    for r in sev_results:
        acc = f"{r['severity_accuracy']:.4f}" if r['severity_accuracy'] is not None else "     N/A"
        print(f"  {r['disease']:<30} {acc:>10} {r['n_samples']:>10,}")
        if r['severity_accuracy'] is not None:
            valid_sev_accs.append(r['severity_accuracy'])
  
    print(f"  {'AVERAGE':<30} {np.mean(valid_sev_accs):>10.4f}")


    results_df = pd.DataFrame(results)
    sev_df     = pd.DataFrame(sev_results)
    merged_df  = results_df.merge(sev_df[['disease', 'severity_accuracy']], on='disease', how='left')
    csv_path   = os.path.join(OUTPUT_DIR, "evaluation_results.csv")
    merged_df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")

  
    print("\nGenerating plots...")
    plot_roc_curves(roc_data,     os.path.join(OUTPUT_DIR, "roc_curves.png"))
    plot_severity_accuracy(sev_results, os.path.join(OUTPUT_DIR, "severity_accuracy.png"))
    plot_summary_bar(results,     os.path.join(OUTPUT_DIR, "summary_metrics.png"))

