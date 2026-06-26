
import os
import pandas as pd

DATA_DIR = r"C:\xray-project\data"

DISEASES = [
    'enlarged cardiomediastinum', 'cardiomegaly', 'atelectasis',
    'consolidation', 'lung edema', 'fracture', 'lung lesion',
    'pleural effusion', 'pneumonia', 'pneumothorax',
    'support device', 'lung opacity', 'pleural other'
]

train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_labels.csv'))
val_df   = pd.read_csv(os.path.join(DATA_DIR, 'val_labels.csv'))
test_df  = pd.read_csv(os.path.join(DATA_DIR, 'test_labels.csv'))


print(f"\nDataset sizes (rows before filtering):")
print(f"  Train : {len(train_df):>8,}")
print(f"  Val   : {len(val_df):>8,}")
print(f"  Test  : {len(test_df):>8,}")

print(f"\nAnnotation types:")
print(f"  {train_df['type_annotation'].unique().tolist()}")

prob_train = train_df[train_df['type_annotation'] == 'probability']
prob_val   = val_df[val_df['type_annotation'] == 'probability']
prob_test  = test_df[test_df['type_annotation'] == 'probability']

print(f"\nUnique images per split:")
print(f"  Train : {len(prob_train):>8,}")
print(f"  Val   : {len(prob_val):>8,}")
print(f"  Test  : {len(prob_test):>8,}")
print(f"  Total : {len(prob_train) + len(prob_val) + len(prob_test):>8,}")

print(f"\nDisease distribution in training set (probability annotation):")
print(f"  {'Disease':<30} {'Present':>8} {'Absent':>8} {'Uncertain':>10} {'% Present':>10}")


total = len(prob_train)
for disease in DISEASES:
    col       = pd.to_numeric(prob_train[disease], errors='coerce')
    present   = (col >= 50).sum()
    absent    = (col < 50).sum()
    uncertain = (col == 101).sum()
    pct       = 100 * present / total
    print(f"  {disease:<30} {present:>8,} {absent:>8,} {uncertain:>10,} {pct:>9.1f}%")

