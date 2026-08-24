import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

METADATA = "data/metadata/selected_subjects.csv"
PROCESSED_DIR = "data/processed"
OUTPUT_DIR = "data/dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load metadata
df = pd.read_csv(METADATA)

# Remove invalid IDs
df = df[
    df["FILE_ID"].notna() &
    (df["FILE_ID"].astype(str).str.strip() != "")
]

df["FILE_ID"] = df["FILE_ID"].astype(str).str.strip()
df["LABEL"] = df["LABEL"].astype(int)

print("Total subjects:", len(df))
print("\nClass distribution:")
print(df["LABEL"].value_counts())

# --------------------------------------------------
# Train = 70%, Validation = 15%, Test = 15%
# --------------------------------------------------

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["LABEL"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["LABEL"],
    random_state=42
)

# Save split metadata
train_df.to_csv(
    os.path.join(OUTPUT_DIR, "train.csv"),
    index=False
)

val_df.to_csv(
    os.path.join(OUTPUT_DIR, "validation.csv"),
    index=False
)

test_df.to_csv(
    os.path.join(OUTPUT_DIR, "test.csv"),
    index=False
)

print("\nDataset split:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

print("\nTrain labels:")
print(train_df["LABEL"].value_counts())

print("\nValidation labels:")
print(val_df["LABEL"].value_counts())

print("\nTest labels:")
print(test_df["LABEL"].value_counts())

print("\nDataset preparation completed.")
