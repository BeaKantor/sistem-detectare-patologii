

import os
import sys
import argparse
import numpy as np
import pandas as pd

DATA_DIR   = r"D:\xray-project\data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
MODEL_PATH = r"D:\xray-project\xray_model.pth"

TRAIN_LABELS = os.path.join(DATA_DIR, "train_labels_clean.csv")
TEST_LABELS  = os.path.join(DATA_DIR, "test_labels_clean.csv")

DISEASES = [
    'enlarged cardiomediastinum', 'cardiomegaly', 'atelectasis',
    'consolidation', 'lung edema', 'fracture', 'lung lesion',
    'pleural effusion', 'pneumonia', 'pneumothorax',
    'support device', 'lung opacity', 'pleural other'
]

N_PER_DISEASE = 30 

def check_disease_rates():
    print("CHECK 1 — Disease presence rates (training set)")
    

    lab = pd.read_csv(TRAIN_LABELS)
    total = len(lab)
    print(f"Total training images: {total:,}\n")
    print(f"{'Disease':<30} {'Present':>8} {'Rate':>7}")
    

    rates = {}
    for d in DISEASES:
        present = (lab[d] == 1.0).sum()
        rate = 100 * present / total
        rates[d] = rate
        print(f"{d:<30} {present:>8,} {rate:>6.1f}%")

    max_rate = max(rates.values())
    min_rate = min(rates.values())
    print(f"Imbalance ratio (max/min): {max_rate/min_rate:.0f}:1")


def check_support_device_imbalance():
    print("CHECK 2 — Support device imbalance (test set)")
    

    lab = pd.read_csv(TEST_LABELS)
    sd = lab['support device']
    present = (sd == 1.0).sum()
    absent  = (sd == 0.0).sum()

    print(f"Support device — present: {present:,}")
    print(f"Support device — absent:  {absent:,}")
    print(f"\nOnly {absent} absent cases means the model cannot learn")
    print("to recognise 'absent' — justification for excluding it from the app.")



def check_negatives():
    print("CHECK 3 — Negative-case verification (all 13 diseases)")

    import torch
    import cv2
    from preprocess import get_val_transforms
    from model import load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(MODEL_PATH, device)
    transform = get_val_transforms()
    lab = pd.read_csv(TEST_LABELS)

    print(f"{'Disease':<28} {'Tested':>7} {'Said no':>9} {'Acc':>6}")
 

    for d_idx, disease in enumerate(DISEASES):
        absent = lab[lab[disease] == 0.0]
        if len(absent) == 0:
            print(f"{disease:<28} {'0':>7} {'-- no absent --':>9}")
            continue

        checked = 0
        correct = 0
        for i in range(len(absent)):
            fname = str(absent.iloc[i, 0]).split('/')[-1].split('\\')[-1]
            if not fname.endswith('.png'):
                fname += '.png'
            img = cv2.imread(os.path.join(IMAGES_DIR, fname))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = transform(image=img_rgb)['image'].unsqueeze(0).to(device)
            with torch.no_grad():
                prob_out, _ = model(tensor)
            prob = float(prob_out[0][d_idx])
            if prob < 0.5:
                correct += 1
            checked += 1
            if checked >= N_PER_DISEASE:
                break

        acc = 100 * correct / checked if checked else 0
        print(f"{disease:<28} {checked:>7} {correct:>9} {acc:>5.0f}%")

    print("High accuracy = the model correctly recognises absent diseases.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rates-only', action='store_true',
                        help='Run only the fast CSV-based checks (skip the model).')
    args = parser.parse_args()

    
    check_disease_rates()
    check_support_device_imbalance()

    
    if not args.rates_only:
        check_negatives()
    else:
        print("\n(Skipped CHECK 3 — model-based verification — due to --rates-only)")