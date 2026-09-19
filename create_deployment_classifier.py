import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ============================================================
# PATHS
# ============================================================

MODEL_PATH = r".\models\visitmode_classification_model.pkl"
DATA_PATH = r".\data\processed\tourism_master_cleaned.csv"
OUTPUT_PATH = r".\models\visitmode_classification_deployment.pkl"

# ============================================================
# LOAD EXISTING PIPELINE
# ============================================================

print("Loading existing classification pipeline...")

pipeline = joblib.load(MODEL_PATH)
preprocessor = pipeline.named_steps["preprocessor"]
old_model = pipeline.named_steps["model"]

print("Existing model loaded.")
print("Existing trees:", len(old_model.estimators_))
print("Existing features:", old_model.n_features_in_)
print("Classes:", old_model.classes_)

# ============================================================
# LOAD FEATURE DATA
# ============================================================

print("\nLoading feature dataset...")
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)

# ============================================================
# RECREATE THE ORIGINAL ENHANCED TRAINING FEATURES
# ============================================================

mode_mapping = {
    1: "Business",
    2: "Couples",
    3: "Family",
    4: "Friends",
    5: "Solo"
}

df["visitmode_name"] = df["visitmode"].map(mode_mapping)

train_idx, _ = train_test_split(
    df.index,
    test_size=0.20,
    random_state=42,
    stratify=df["visitmode_name"]
)

train_data = df.loc[train_idx].copy()
modes = list(old_model.classes_)

for mode in modes:
    train_data[f"mode_{mode}"] = (
        train_data["visitmode_name"] == mode
    ).astype(int)

global_mode_distribution = train_data[
    "visitmode_name"
].value_counts(normalize=True)

user_mode_stats = train_data.groupby("userid")[
    [f"mode_{mode}" for mode in modes]
].sum()
user_total = user_mode_stats.sum(axis=1)

attraction_mode_stats = train_data.groupby("attractionid")[
    [f"mode_{mode}" for mode in modes]
].sum()
attraction_total = attraction_mode_stats.sum(axis=1)

type_mode_stats = train_data.groupby("attractiontypeid")[
    [f"mode_{mode}" for mode in modes]
].sum()
type_total = type_mode_stats.sum(axis=1)

for mode in modes:
    suffix = mode.lower()
    user_mode_stats[f"user_{suffix}_ratio"] = (
        user_mode_stats[f"mode_{mode}"]
        / user_total.replace(0, np.nan)
    )
    attraction_mode_stats[f"attraction_{suffix}_ratio"] = (
        attraction_mode_stats[f"mode_{mode}"]
        / attraction_total.replace(0, np.nan)
    )
    type_mode_stats[f"type_{suffix}_ratio"] = (
        type_mode_stats[f"mode_{mode}"]
        / type_total.replace(0, np.nan)
    )

user_ratios = user_mode_stats[
    [f"user_{mode.lower()}_ratio" for mode in modes]
].reset_index()
attraction_ratios = attraction_mode_stats[
    [f"attraction_{mode.lower()}_ratio" for mode in modes]
].reset_index()
type_ratios = type_mode_stats[
    [f"type_{mode.lower()}_ratio" for mode in modes]
].reset_index()

train_data = train_data.merge(user_ratios, on="userid", how="left")
train_data = train_data.merge(
    attraction_ratios,
    on="attractionid",
    how="left"
)
train_data = train_data.merge(
    type_ratios,
    on="attractiontypeid",
    how="left"
)

for mode in modes:
    for prefix in ["user", "attraction", "type"]:
        column = f"{prefix}_{mode.lower()}_ratio"
        train_data[column] = train_data[column].fillna(
            global_mode_distribution.get(mode, 0)
        )

feature_columns = list(pipeline.feature_names_in_)
print("\nExpected input columns:", len(feature_columns))

missing_columns = [
    col for col in feature_columns
    if col not in train_data.columns
]

if missing_columns:
    print("\nMissing columns:")
    for col in missing_columns:
        print("-", col)
    raise ValueError(
        "The deployment dataset does not contain all "
        "features required by the existing model."
    )

X = train_data[feature_columns].copy()

# ============================================================
# TARGET
# ============================================================

TARGET = "visitmode"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' was not found in dataset."
    )

y = train_data["visitmode_name"].copy()

# ============================================================
# TRANSFORM FEATURES
# ============================================================

print("\nTransforming features...")
X_transformed = preprocessor.transform(X)
print("Transformed feature shape:", X_transformed.shape)

# ============================================================
# TRAIN SMALLER RANDOM FOREST
# ============================================================

print("\nTraining deployment model...")

deployment_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=15,
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=42
)

deployment_model.fit(X_transformed, y)

# ============================================================
# TRAINING PERFORMANCE CHECK
# ============================================================

print("\nEvaluating deployment model...")
predictions = deployment_model.predict(X_transformed)
accuracy = accuracy_score(y, predictions)
weighted_f1 = f1_score(y, predictions, average="weighted")
macro_f1 = f1_score(y, predictions, average="macro")

print("\nDeployment model training results:")
print("Accuracy:", round(accuracy, 4))
print("Weighted F1:", round(weighted_f1, 4))
print("Macro F1:", round(macro_f1, 4))
print("\nClassification Report:")
print(classification_report(y, predictions))

# ============================================================
# CREATE PIPELINE
# ============================================================

deployment_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", deployment_model)
    ]
)

# ============================================================
# SAVE
# ============================================================

print("\nSaving deployment model...")
joblib.dump(deployment_pipeline, OUTPUT_PATH, compress=3)

# ============================================================
# FILE SIZE
# ============================================================

file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)

print("\n========================================")
print("DEPLOYMENT MODEL CREATED")
print("========================================")
print("File:")
print(OUTPUT_PATH)
print("Size:", round(file_size_mb, 2), "MB")
print("========================================")
