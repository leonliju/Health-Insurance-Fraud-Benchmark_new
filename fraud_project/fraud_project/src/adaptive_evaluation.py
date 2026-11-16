
"""
Module 3B: Evaluate the impact of adaptive fraud.

Compares the original XGBoost model on:
1. Original test claims
2. Adapted test claims

The model is NOT retrained in this experiment.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_DIR = "data_v3"
ADAPTIVE_DIR = "data_v3/adaptive/5pct"
TAG = "5pct"
VARIANT = "with_district"
DETECTOR = "xgboost"

MODEL_PATH = (
    f"{DATA_DIR}/models/"
    f"{DETECTOR}_{TAG}_{VARIANT}.joblib"
)

BASE_FEATURES = f"{DATA_DIR}/features_{TAG}.csv"
BASE_LABELS = f"{DATA_DIR}/labels_{TAG}.csv"
BASE_FEATURE_LIST = f"{DATA_DIR}/feature_list_{TAG}.json"

ADAPTIVE_FEATURES = f"{ADAPTIVE_DIR}/features_{TAG}_round1.csv"
ADAPTIVE_LABELS = f"{ADAPTIVE_DIR}/labels_{TAG}_round1.csv"
ADAPTIVE_FEATURE_LIST = (
    f"{ADAPTIVE_DIR}/feature_list_{TAG}_round1.json"
)

OUTPUT_DIR = ADAPTIVE_DIR
CAPACITY = 0.05


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("\nLoading original and adapted datasets...")

X_base = pd.read_csv(BASE_FEATURES)
Y_base = pd.read_csv(BASE_LABELS)

X_adapt = pd.read_csv(ADAPTIVE_FEATURES)
Y_adapt = pd.read_csv(ADAPTIVE_LABELS)

with open(BASE_FEATURE_LIST) as f:
    base_list = json.load(f)

with open(ADAPTIVE_FEATURE_LIST) as f:
    adapt_list = json.load(f)

# Select features used by the original model
feature_cols = base_list["features"] + base_list["optional"]

if VARIANT == "no_district":
    feature_cols = base_list["features"]

# Verify feature compatibility
adapt_feature_cols = (
    adapt_list["features"] + adapt_list["optional"]
)

if VARIANT == "no_district":
    adapt_feature_cols = adapt_list["features"]

if feature_cols != adapt_feature_cols:
    raise ValueError(
        "Original and adapted feature lists do not match."
    )

# Verify claim IDs and labels
if not X_base["claim_id"].equals(Y_base["claim_id"]):
    raise ValueError("Original claim ID order does not match.")

if not X_adapt["claim_id"].equals(Y_adapt["claim_id"]):
    raise ValueError("Adapted claim ID order does not match.")

# Only evaluate test claims
base_test = X_base["split"].eq("test")
adapt_test = X_adapt["split"].eq("test")

base_ids = X_base.loc[base_test, "claim_id"].reset_index(drop=True)
adapt_ids = X_adapt.loc[adapt_test, "claim_id"].reset_index(drop=True)

if not base_ids.equals(adapt_ids):
    raise ValueError("Original and adapted test claim IDs differ.")

base_y = Y_base.loc[base_test].reset_index(drop=True)
adapt_y = Y_adapt.loc[adapt_test].reset_index(drop=True)

if not base_y["is_fraud"].equals(adapt_y["is_fraud"]):
    raise ValueError("Fraud labels changed during adaptation.")

print("Original test claims:", len(base_ids))
print("Adapted test claims:", len(adapt_ids))


# --------------------------------------------------
# LOAD ORIGINAL MODEL
# --------------------------------------------------

print("\nLoading original XGBoost model...")

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# GENERATE FRAUD SCORES
# --------------------------------------------------

print("\nScoring original test claims...")

X_base_test = X_base.loc[base_test, feature_cols]
base_scores = model.predict_proba(X_base_test)[:, 1]

print("Scoring adapted test claims...")

X_adapt_test = X_adapt.loc[adapt_test, feature_cols]
adapt_scores = model.predict_proba(X_adapt_test)[:, 1]


# --------------------------------------------------
# EVALUATION FUNCTION
# --------------------------------------------------

def evaluate(y, scores, threshold=None):

    y = np.asarray(y)
    scores = np.asarray(scores)

    K = max(1, int(round(CAPACITY * len(y))))

    # Select the K highest-scoring claims
    top_k = np.argsort(-scores)[:K]

    flagged = np.zeros(len(y), dtype=bool)
    flagged[top_k] = True

    precision_k = y[flagged].mean()
    recall_k = y[flagged].sum() / max(y.sum(), 1)

    result = {
        "PR_AUC": average_precision_score(y, scores),
        "Precision_at_K": precision_k,
        "Recall_at_K": recall_k,
        "K": K
    }

    if threshold is not None:
        threshold_flags = scores >= threshold

        result["Recall_at_baseline_threshold"] = (
            y[threshold_flags].sum() / max(y.sum(), 1)
        )

        result["Flagged_at_baseline_threshold"] = (
            threshold_flags.sum()
        )

    return result, flagged


# --------------------------------------------------
# BASELINE THRESHOLD
# --------------------------------------------------

K = max(1, int(round(CAPACITY * len(base_scores))))

# Score cutoff for the original model's top-K claims
baseline_threshold = np.sort(base_scores)[-K]


# --------------------------------------------------
# EVALUATE BOTH DATASETS
# --------------------------------------------------

print("\nEvaluating baseline performance...")

baseline_results, baseline_flags = evaluate(
    base_y["is_fraud"],
    base_scores
)

print("Evaluating adapted performance...")

adapted_results, adapted_flags = evaluate(
    adapt_y["is_fraud"],
    adapt_scores,
    threshold=baseline_threshold
)

baseline_results["Recall_at_baseline_threshold"] = (
    base_y["is_fraud"].to_numpy()[base_scores >= baseline_threshold].sum()
    / max(base_y["is_fraud"].sum(), 1)
)

baseline_results["Flagged_at_baseline_threshold"] = (
    base_scores >= baseline_threshold
).sum()


# --------------------------------------------------
# DISPLAY COMPARISON
# --------------------------------------------------

comparison = pd.DataFrame(
    [baseline_results, adapted_results],
    index=["Baseline", "After Adaptation"]
)

print("\n==============================================")
print("ADAPTIVE FRAUD EVALUATION")
print("==============================================")
print(comparison.round(4).to_string())


# --------------------------------------------------
# FRAUD TYPE AND SEVERITY ANALYSIS
# --------------------------------------------------

group_rows = []

for (fraud_type, severity), indices in (
    adapt_y[adapt_y["is_fraud"] == 1]
    .groupby(["fraud_type", "severity_tier"]).groups.items()
):

    idx = np.array(list(indices))

    group_rows.append({
        "fraud_type": fraud_type,
        "severity_tier": severity,
        "total_claims": len(idx),
        "baseline_recall_at_K": baseline_flags[idx].mean(),
        "adapted_recall_at_K": adapted_flags[idx].mean()
    })

group_results = pd.DataFrame(group_rows)

print("\nFRAUD TYPE AND SEVERITY COMPARISON")
print(group_results.round(4).to_string(index=False))


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

comparison.to_csv(
    f"{OUTPUT_DIR}/adaptive_evaluation_metrics.csv"
)

group_results.to_csv(
    f"{OUTPUT_DIR}/adaptive_group_evaluation.csv",
    index=False
)

print("\nEvaluation completed successfully!")
print("Results saved in:", OUTPUT_DIR)
