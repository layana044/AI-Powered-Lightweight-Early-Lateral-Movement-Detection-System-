# ============================================================
# Lateral Movement Detection System
# Step 4: Machine Learning Model Training (Isolation Forest)
#
# Purpose:
# Train an unsupervised Isolation Forest model and evaluate it
# properly using a train/test split to avoid data leakage.
# ============================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import joblib

# ============================================================
# Step 1: Load Dataset (WITH labels)
# ============================================================

print("================================================")
print("              LOADING DATASET                  ")
print("================================================")

dataset = pd.read_csv("preprocessed_dataset_with_label.csv")

# Separate features and labels
X = dataset.drop(columns=['Label'])
y = dataset['Label']

print(f"Dataset loaded: {X.shape[0]} rows, {X.shape[1]} features")

# ============================================================
# Step 2: Split Data (IMPORTANT)
# ============================================================

print("\n================================================")
print("            SPLITTING DATASET                  ")
print("================================================")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,        # 30% for testing
    random_state=42,
    stratify=y            # maintain class distribution
)

print(f"Training set: {X_train.shape[0]} rows")
print(f"Testing set : {X_test.shape[0]} rows")

# ============================================================
# Step 3: Train Isolation Forest Model
# ============================================================

print("\n================================================")
print("        TRAINING ISOLATION FOREST MODEL        ")
print("================================================")

model = IsolationForest(
    n_estimators=100,
    contamination=0.08,   # approx. 8% anomalies
    random_state=42,
    n_jobs=-1
)

print("Training model... please wait.")
model.fit(X_train)
print("Model training complete!")

# ============================================================
# Step 4: Make Predictions
# ============================================================

print("\n================================================")
print("              MAKING PREDICTIONS               ")
print("================================================")

predictions = model.predict(X_test)

# Convert predictions:
# 1  → 0 (Normal)
# -1 → 1 (Attack)
predictions_mapped = np.where(predictions == 1, 0, 1)

# ============================================================
# Step 5: Evaluate Model Performance
# ============================================================

print("\n================================================")
print("         EVALUATING MODEL PERFORMANCE          ")
print("================================================")

accuracy = accuracy_score(y_test, predictions_mapped)
precision = precision_score(y_test, predictions_mapped, zero_division=0)
recall = recall_score(y_test, predictions_mapped, zero_division=0)
f1 = f1_score(y_test, predictions_mapped, zero_division=0)
conf_matrix = confusion_matrix(y_test, predictions_mapped)

print("Evaluation Metrics:")
print(f"  Accuracy  : {accuracy:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1-Score  : {f1:.4f}")

print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(classification_report(y_test, predictions_mapped))

# ============================================================
# Step 6: Save the Model
# ============================================================

print("\n================================================")
print("                 SAVING MODEL                  ")
print("================================================")

model_filename = "isolation_forest_model.pkl"
joblib.dump(model, model_filename)

print(f"Model successfully saved as: {model_filename}")

print("\n================================================")
print("               PIPELINE COMPLETE               ")
print("================================================")
