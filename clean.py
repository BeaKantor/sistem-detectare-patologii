

import os
import numpy as np
import pandas as pd

DATA_DIR = r"C:\xray-project\data"

DISEASES = [
    'enlarged cardiomediastinum', 'cardiomegaly', 'atelectasis',
    'consolidation', 'lung edema', 'fracture', 'lung lesion',
    'pleural effusion', 'pneumonia', 'pneumothorax',
    'support device', 'lung opacity', 'pleural other'
]


UNCERTAIN_VALUES = {-1, -2, -3}


def load_annotation(csv_path: str, annotation_type: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df['type_annotation'] == annotation_type].copy()
    df = df.reset_index(drop=True)
    return df[['subjectid_studyid'] + DISEASES]


def build_presence_mask(labels_df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame()
    result['subjectid_studyid'] = labels_df['subjectid_studyid']
    for disease in DISEASES:
        col = pd.to_numeric(labels_df[disease], errors='coerce')
        result[disease] = np.where(
            col.isin(UNCERTAIN_VALUES) | col.isna(), np.nan,
            np.where(col == 1, 1.0, 0.0)
        )
    return result


def build_probability(probability_df: pd.DataFrame, presence_mask: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame()
    result['subjectid_studyid'] = probability_df['subjectid_studyid']
    for disease in DISEASES:
        prob = pd.to_numeric(probability_df[disease], errors='coerce')
        pres = presence_mask[disease].values
        prob_clean = np.where(prob == 101, np.nan, prob / 100.0)
        prob_clean = np.where(np.isnan(pres.astype(float)), np.nan, prob_clean)
        result[disease] = prob_clean
    return result


def build_severity(severity_df: pd.DataFrame, presence_mask: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame()
    result['subjectid_studyid'] = severity_df['subjectid_studyid']
    for disease in DISEASES:
        sev  = pd.to_numeric(severity_df[disease], errors='coerce')
        pres = presence_mask[disease].values
        sev_clean = np.where(sev == -1, 0, sev).astype(float)
        sev_clean = np.where(np.isnan(pres.astype(float)), np.nan, sev_clean)
        sev_clean = np.where(pres == 0.0, 0.0, sev_clean)
        result[disease] = sev_clean
    return result


def build_location(location_df: pd.DataFrame, presence_mask: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame()
    result['subjectid_studyid'] = location_df['subjectid_studyid']
    for disease in DISEASES:
        loc  = location_df[disease].astype(str).str.strip()
        pres = presence_mask[disease].values
        loc_clean = loc.replace('-1', '').values.astype(object)
        loc_clean = np.where(np.isnan(pres.astype(float)), np.nan, loc_clean)
        loc_clean = np.where(pres == 0.0, '', loc_clean)
        result[disease] = loc_clean
    return result


def clean_split(raw_path: str, out_prefix: str) -> None:
    print(f"Processing: {os.path.basename(raw_path)}")

    labels_df      = load_annotation(raw_path, 'labels')
    probability_df = load_annotation(raw_path, 'probability')
    severity_df    = load_annotation(raw_path, 'severity')
    location_df    = load_annotation(raw_path, 'location')

    presence    = build_presence_mask(labels_df)
    probability = build_probability(probability_df, presence)
    severity    = build_severity(severity_df, presence)
    location    = build_location(location_df, presence)

    files = {
        f"{out_prefix}_labels_clean.csv":      presence,
        f"{out_prefix}_probability_clean.csv": probability,
        f"{out_prefix}_severity_clean.csv":    severity,
        f"{out_prefix}_location_clean.csv":    location,
    }

    for filename, df in files.items():
        df.to_csv(os.path.join(DATA_DIR, filename), index=False)
        print(f"  Saved → {filename}")


if __name__ == "__main__":

    splits = [
        (os.path.join(DATA_DIR, "train_labels.csv"), "train"),
        (os.path.join(DATA_DIR, "val_labels.csv"),   "val"),
        (os.path.join(DATA_DIR, "test_labels.csv"),  "test"),
    ]

    for raw_path, prefix in splits:
        clean_split(raw_path, prefix)

