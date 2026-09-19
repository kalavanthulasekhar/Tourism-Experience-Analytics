import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ============================================================
# PATHS
# ============================================================

DATA_PATH = r".\data\processed\tourism_master_cleaned.csv"

ORIGINAL_MODEL_PATH = (
    r".\models\visitmode_classification_model.pkl"
)

DEPLOYMENT_MODEL_PATH = (
    r".\models\visitmode_classification_deployment.pkl"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


TARGET = "visitmode"

mode_mapping = {
    1: "Business",
    2: "Couples",
    3: "Family",
    4: "Friends",
    5: "Solo"
}

df["visitmode_name"] = df[TARGET].map(mode_mapping)


# ============================================================
# EXACT ORIGINAL TRAIN/TEST SPLIT
# ============================================================

train_idx, test_idx = train_test_split(
    df.index,
    test_size=0.20,
    random_state=42,
    stratify=df["visitmode_name"]
)

train_data = df.loc[train_idx].copy()
test_data = df.loc[test_idx].copy()

modes = [
    "Business",
    "Couples",
    "Family",
    "Friends",
    "Solo"
]

for mode in modes:
    train_data[f"mode_{mode}"] = (
        train_data["visitmode_name"] == mode
    ).astype(int)

global_mode_distribution = train_data[
    "visitmode_name"
].value_counts(normalize=True)

# Reproduce notebook 09's leave-one-out user history features for training.
user_counts = pd.crosstab(
    train_data["userid"],
    train_data["visitmode_name"]
).reindex(columns=modes, fill_value=0).astype(float)

user_history_count = (
    train_data["userid"].map(user_counts.sum(axis=1)) - 1
)

for mode in modes:
    column = f"user_{mode.lower()}_ratio"
    train_data[column] = (
        train_data["userid"].map(user_counts[mode])
        - (train_data["visitmode_name"] == mode).astype(int)
    ) / user_history_count.replace(0, np.nan)
    train_data[column] = train_data[column].fillna(
        global_mode_distribution.get(mode, 0)
    )
    test_data[column] = (
        test_data["userid"].map(user_counts[mode])
        / user_counts.sum(axis=1).reindex(
            test_data["userid"]
        ).values
    )
    test_data[column] = test_data[column].fillna(
        global_mode_distribution.get(mode, 0)
    )

for group_column, prefix in [
    ("attractionid", "attraction"),
    ("attractiontypeid", "type")
]:
    group_counts = pd.crosstab(
        train_data[group_column],
        train_data["visitmode_name"]
    ).reindex(columns=modes, fill_value=0).astype(float)
    group_totals = group_counts.sum(axis=1)

    for mode in modes:
        column = f"{prefix}_{mode.lower()}_ratio"
        ratios = group_counts[mode] / group_totals.replace(0, np.nan)
        train_data[column] = train_data[group_column].map(ratios).fillna(
            global_mode_distribution.get(mode, 0)
        )
        test_data[column] = test_data[group_column].map(ratios).fillna(0)

feature_columns = list(
    joblib.load(ORIGINAL_MODEL_PATH).feature_names_in_
)
X_test = test_data[feature_columns].copy()
y_test = test_data["visitmode_name"].copy()

print("\nTraining samples:", len(train_data))
print("Testing samples :", len(X_test))


# ============================================================
# LOAD MODELS
# ============================================================

print("\n" + "=" * 70)
print("LOADING MODELS")
print("=" * 70)

print("\nLoading original model...")
original_model = joblib.load(
    ORIGINAL_MODEL_PATH
)

print("Original model loaded.")

print("\nLoading deployment model...")
deployment_model = joblib.load(
    DEPLOYMENT_MODEL_PATH
)

print("Deployment model loaded.")


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, name):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    print("\nAccuracy     :", round(accuracy, 4))
    print("Precision    :", round(precision, 4))
    print("Recall       :", round(recall, 4))
    print("Weighted F1  :", round(weighted_f1, 4))
    print("Macro F1     :", round(macro_f1, 4))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Weighted_F1": weighted_f1,
        "Macro_F1": macro_f1
    }


# ============================================================
# EVALUATE BOTH MODELS
# ============================================================

original_results = evaluate_model(
    original_model,
    "ORIGINAL 908 MB MODEL"
)

deployment_results = evaluate_model(
    deployment_model,
    "DEPLOYMENT 4.51 MB MODEL"
)


# ============================================================
# COMPARISON
# ============================================================

results = pd.DataFrame([
    original_results,
    deployment_results
])

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    results.to_string(
        index=False
    )
)


# ============================================================
# SIZE COMPARISON
# ============================================================

original_size = (
    os.path.getsize(
        ORIGINAL_MODEL_PATH
    ) / (1024 * 1024)
)

deployment_size = (
    os.path.getsize(
        DEPLOYMENT_MODEL_PATH
    ) / (1024 * 1024)
)

reduction = (
    (original_size - deployment_size)
    / original_size
) * 100


print("\n" + "=" * 70)
print("MODEL SIZE COMPARISON")
print("=" * 70)

print(
    "Original size    :",
    round(original_size, 2),
    "MB"
)

print(
    "Deployment size  :",
    round(deployment_size, 2),
    "MB"
)

print(
    "Size reduction   :",
    round(reduction, 2),
    "%"
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_path = (
    r".\reports\classification_model_comparison.csv"
)

results.to_csv(
    output_path,
    index=False
)

print("\nComparison saved to:")
print(output_path)

print("\nDone.")