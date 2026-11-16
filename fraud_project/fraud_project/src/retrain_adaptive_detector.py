"""
Retrain XGBoost on adaptively modified TRAIN claims and compare it with
the original XGBoost model on the SAME adapted-build TEST features.

Usage:
python src/retrain_adaptive_detector.py --dir data_v3 --tag 5pct \
  --adaptive-dir data_v3/adaptive/5pct_train_adaptation \
  --variant with_district --capacity 0.05
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier


def evaluate_scores(name, scores, y_true, labels_test, k):
    """Calculate ranking metrics and fraud-type/severity Recall@K."""
    order = np.argsort(-np.asarray(scores))
    flagged = np.zeros(len(y_true), dtype=bool)
    flagged[order[:k]] = True

    result = {
        "model": name,
        "pr_auc": average_precision_score(y_true, scores),
        "precision_at_K": float(np.mean(y_true[flagged])) if flagged.any() else 0.0,
        "recall_at_K": (
            float(np.sum(y_true[flagged]) / np.sum(y_true))
            if np.sum(y_true) else 0.0
        ),
        "K": int(k),
    }

    group_frame = labels_test.reset_index(drop=True).copy()
    group_frame["flagged"] = flagged
    group_frame = group_frame[group_frame["is_fraud"] == 1]

    group_rows = []
    for (fraud_type, tier), group in group_frame.groupby(
        ["fraud_type", "severity_tier"], dropna=False
    ):
        group_rows.append({
            "model": name,
            "fraud_type": fraud_type,
            "severity_tier": tier,
            "total_claims": len(group),
            "recall_at_K": float(group["flagged"].mean()),
        })

    return result, group_rows, flagged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data_v3")
    parser.add_argument("--tag", default="5pct")
    parser.add_argument(
        "--adaptive-dir",
        default="data_v3/adaptive/5pct_train_adaptation",
        help="Folder created by adaptive_fraudster.py with --target-split train",
    )
    parser.add_argument(
        "--variant",
        choices=["with_district", "no_district"],
        default="with_district",
    )
    parser.add_argument("--capacity", type=float, default=0.05)
    args = parser.parse_args()

    if not 0 < args.capacity <= 1:
        raise ValueError("--capacity must be greater than 0 and at most 1")

    # Original feature metadata defines the exact feature columns and order.
    with open(os.path.join(args.dir, f"feature_list_{args.tag}.json")) as f:
        feature_list = json.load(f)

    if args.variant == "with_district":
        cols = feature_list["features"] + feature_list.get("optional", [])
    else:
        cols = feature_list["features"]

    # Original data and model.
    X_original = pd.read_csv(os.path.join(args.dir, f"features_{args.tag}.csv"))
    Y_original = pd.read_csv(os.path.join(args.dir, f"labels_{args.tag}.csv"))
    original_model_path = os.path.join(
        args.dir, "models", f"xgboost_{args.tag}_{args.variant}.joblib"
    )
    if not os.path.exists(original_model_path):
        raise FileNotFoundError(
            f"Original model not found: {original_model_path}. "
            "Run train_detectors.py first."
        )
    original_model = joblib.load(original_model_path)

    # Adapted feature/label tables created from adapted TRAIN raw claims.
    X_adapted = pd.read_csv(
        os.path.join(args.adaptive_dir, f"features_{args.tag}_round1.csv")
    )
    Y_adapted = pd.read_csv(
        os.path.join(args.adaptive_dir, f"labels_{args.tag}_round1.csv")
    )

    # Validate row alignment and ensure the held-out labels were not changed.
    if not X_adapted["claim_id"].astype(str).equals(
        Y_adapted["claim_id"].astype(str)
    ):
        raise ValueError("Adapted feature and label claim_id order does not match.")

    original_test = Y_original[Y_original["split"] == "test"].copy()
    adapted_test = Y_adapted[Y_adapted["split"] == "test"].copy()
    original_test = original_test.sort_values("claim_id").reset_index(drop=True)
    adapted_test = adapted_test.sort_values("claim_id").reset_index(drop=True)

    if not original_test["claim_id"].astype(str).equals(
        adapted_test["claim_id"].astype(str)
    ):
        raise ValueError("Original and adapted TEST claim IDs do not match.")

    for col in ["is_fraud", "fraud_type", "severity_tier"]:
        if col in original_test and col in adapted_test:
            left = original_test[col].fillna("").astype(str)
            right = adapted_test[col].fillna("").astype(str)
            if not left.equals(right):
                raise ValueError(f"TEST labels changed unexpectedly in column: {col}")

    missing = [c for c in cols if c not in X_adapted.columns]
    if missing:
        raise ValueError(f"Adapted feature table is missing columns: {missing}")

    train_mask = X_adapted["split"].eq("train")
    test_mask = X_adapted["split"].eq("test")

    # Align labels to feature rows by claim_id rather than relying on row position.
    y_by_id = Y_adapted.set_index("claim_id")["is_fraud"]
    y_all = X_adapted["claim_id"].map(y_by_id)
    if y_all.isna().any():
        raise ValueError("Some adapted feature rows have no matching labels.")

    X_train = X_adapted.loc[train_mask, cols]
    y_train = y_all.loc[train_mask].astype(int).to_numpy()

    # Use the same adapted-build test feature rows for both models.
    X_test_frame = X_adapted.loc[test_mask].copy()
    X_test = X_test_frame[cols]
    test_labels = Y_adapted.set_index("claim_id").loc[
        X_test_frame["claim_id"], :
    ].reset_index()
    y_test = test_labels["is_fraud"].astype(int).to_numpy()

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Training or test split is empty.")
    if len(np.unique(y_train)) < 2:
        raise ValueError("Adapted training split must contain both classes.")

    # Match the XGBoost configuration in train_detectors.py.
    positive_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    retrained_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=positive_weight,
        tree_method="hist",
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )
    retrained_model.fit(X_train, y_train)

    # Score both models on precisely the same held-out test feature matrix.
    baseline_scores = original_model.predict_proba(X_test)[:, 1]
    retrained_scores = retrained_model.predict_proba(X_test)[:, 1]

    k = max(1, int(round(args.capacity * len(y_test))))
    baseline_metrics, baseline_groups, _ = evaluate_scores(
        "Original XGBoost", baseline_scores, y_test, test_labels, k
    )
    retrained_metrics, retrained_groups, _ = evaluate_scores(
        "Adaptive-retrained XGBoost", retrained_scores, y_test, test_labels, k
    )

    output_dir = os.path.join(args.adaptive_dir, "retrained")
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)

    metrics = pd.DataFrame([baseline_metrics, retrained_metrics])
    metrics.to_csv(os.path.join(output_dir, "retraining_comparison_metrics.csv"), index=False)

    group_metrics = pd.DataFrame(baseline_groups + retrained_groups)
    group_metrics.to_csv(
        os.path.join(output_dir, "retraining_group_comparison.csv"), index=False
    )

    score_output = pd.DataFrame({
        "claim_id": X_test_frame["claim_id"].to_numpy(),
        "split": "test",
        "is_fraud": y_test,
        "original_xgboost_score": baseline_scores,
        "adaptive_retrained_xgboost_score": retrained_scores,
    })
    score_output.to_csv(
        os.path.join(output_dir, "retraining_test_scores.csv"), index=False
    )

    model_output = os.path.join(
        output_dir, "models", f"xgboost_{args.tag}_{args.variant}_adaptive_retrained.joblib"
    )
    joblib.dump(retrained_model, model_output)

    print("\nADAPTIVE RETRAINING COMPARISON")
    print(f"Training rows: {len(X_train):,}")
    print(f"Training fraud claims: {int(y_train.sum()):,}")
    print(f"Test rows: {len(X_test):,}")
    print(f"Review capacity: {args.capacity:.1%} (K={k:,})")
    print(metrics.round(4).to_string(index=False))

    print("\nFRAUD TYPE AND SEVERITY RECALL@K")
    print(group_metrics.round(4).to_string(index=False))

    print("\nSaved outputs:")
    print(os.path.join(output_dir, "retraining_comparison_metrics.csv"))
    print(os.path.join(output_dir, "retraining_group_comparison.csv"))
    print(os.path.join(output_dir, "retraining_test_scores.csv"))
    print(model_output)
    print(
        "\nNote: both models were evaluated on the same test features rebuilt "
        "after TRAIN adaptation. The original model was not retrained."
    )


if __name__ == "__main__":
    main()
