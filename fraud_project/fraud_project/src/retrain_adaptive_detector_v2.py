"""
Module 3C (corrected): adversarial retraining of XGBoost.

Fixes three problems in retrain_adaptive_detector.py:
  1. REPLACE vs AUGMENT. The old script trained on the rebuilt train table in which every adapted fraud claim
     had REPLACED its original. The retrained model therefore never saw strongly inflated upcoding. Here the
     default (--mode augment) keeps every original training claim and ADDS the adapted copies of the fraud claims.
  2. FEATURE SHIFT. build_features.py recomputes hospital-level features (hosp_mean_adj_ratio, hosp_volume) from the
     whole train split. After TRAIN adaptation these change for almost every test row, so the old "original model"
     was scored on shifted test features. Here every model is scored on features in the ORIGINAL feature space,
     and the adapted training rows get hospital features from the original train statistics.
  3. WRONG TEST SET. The point of adversarial retraining is robustness, so both models are scored on
        A) the clean test claims (does retraining hurt normal performance?)
        B) the test claims adapted against the ORIGINAL model (does retraining recover robustness?)

Usage:
  python src/retrain_adaptive_detector_v2.py --dir data_v3 --tag 5pct --mode augment
  python src/retrain_adaptive_detector_v2.py --dir data_v3 --tag 5pct --mode replace   # reproduces the old behaviour
Needs the outputs of: build_features.py, train_detectors.py, and adaptive_fraudster.py run twice
(--target-split test -> data_v3/adaptive/5pct ; --target-split train -> data_v3/adaptive/5pct_train_adaptation).
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier


def evaluate(name, test_set, scores, y, lab, k):
    order = np.argsort(-scores)
    flag = np.zeros(len(y), dtype=bool)
    flag[order[:k]] = True
    row = {"model": name, "test_set": test_set, "K": k,
           "pr_auc": average_precision_score(y, scores),
           "precision_at_K": float(y[flag].mean()),
           "recall_at_K": float(y[flag].sum() / y.sum())}
    groups = []
    f = lab.assign(flagged=flag)
    f = f[f["is_fraud"] == 1]
    for (ft, tier), g in f.groupby(["fraud_type", "severity_tier"]):
        groups.append({"model": name, "test_set": test_set, "fraud_type": ft, "severity_tier": tier,
                       "total_claims": len(g), "recall_at_K": float(g["flagged"].mean())})
    return row, groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data_v3")
    ap.add_argument("--tag", default="5pct")
    ap.add_argument("--variant", choices=["with_district", "no_district"], default="with_district")
    ap.add_argument("--capacity", type=float, default=0.05)
    ap.add_argument("--mode", choices=["augment", "replace"], default="augment")
    ap.add_argument("--train-adaptive-dir", default=None)
    ap.add_argument("--test-adaptive-dir", default=None)
    ap.add_argument("--round2-adaptive-dir", default=None,
                    help="test claims adapted against the RETRAINED model (adaptive_fraudster.py --model-path ...); "
                         "adds test set C")
    ap.add_argument("--reuse-retrained", action="store_true",
                    help="load the saved retrained model from the output folder instead of training again")
    a = ap.parse_args()
    d, t = a.dir, a.tag
    tr_dir = a.train_adaptive_dir or f"{d}/adaptive/{t}_train_adaptation"
    te_dir = a.test_adaptive_dir or f"{d}/adaptive/{t}"
    out_dir = f"{tr_dir}/retrained_v2_{a.mode}"
    os.makedirs(out_dir, exist_ok=True)

    fl = json.load(open(f"{d}/feature_list_{t}.json"))
    cols = fl["features"] + (fl.get("optional", []) if a.variant == "with_district" else [])

    X = pd.read_csv(f"{d}/features_{t}.csv")
    Y = pd.read_csv(f"{d}/labels_{t}.csv")
    assert X["claim_id"].equals(Y["claim_id"])
    Xta = pd.read_csv(f"{tr_dir}/features_{t}_round1.csv")
    assert Xta["claim_id"].equals(X["claim_id"]), "train-adapted feature rows are not in the same order"
    Xte = pd.read_csv(f"{te_dir}/features_{t}_round1.csv")
    Yte = pd.read_csv(f"{te_dir}/labels_{t}_round1.csv")
    assert Xte["claim_id"].equals(X["claim_id"]) and Yte["claim_id"].equals(Y["claim_id"])

    # ---- original-space hospital statistics (same formulas as build_features.py) ----
    hos = pd.read_csv(f"{d}/hospitals.csv")
    mult = hos.set_index("hospital_id")[next(c for c in hos.columns if "multiplier" in c.lower())]
    orig_claims = pd.read_csv(f"{d}/claims_fraud_{t}.csv").set_index("claim_id")
    ada_claims = pd.read_csv(f"{tr_dir}/adaptive_claims_round_1.csv").set_index("claim_id")
    split = X.set_index("claim_id")["split"]
    adj = orig_claims["claimed_amount"] / orig_claims["reference_package_rate"] / orig_claims["hospital_id"].map(mult)
    is_train = split.reindex(orig_claims.index).eq("train")
    hosp_mean = adj[is_train].groupby(orig_claims.loc[is_train, "hospital_id"]).mean()
    overall_mean = float(adj[is_train].mean())
    hosp_count = orig_claims["hospital_id"].value_counts()

    # ---- which train fraud claims were really changed ----
    summ = pd.read_csv(f"{tr_dir}/adaptation_summary.csv", dtype={"claim_id": str})
    changed_ids = set(summ.loc[summ["changed"].astype(bool), "claim_id"])
    xi = X.set_index("claim_id")
    adapted_rows = Xta.set_index("claim_id").loc[sorted(changed_ids), cols].copy()
    new_h = ada_claims.loc[adapted_rows.index, "hospital_id"]
    old_h = orig_claims.loc[adapted_rows.index, "hospital_id"]
    adapted_rows["hosp_mean_adj_ratio"] = new_h.map(hosp_mean).fillna(overall_mean).to_numpy()
    adapted_rows["hosp_volume"] = (new_h.map(hosp_count).fillna(0) + (new_h != old_h).astype(int)).to_numpy()
    same_h = (new_h == old_h).to_numpy()
    for c in ["hosp_mean_adj_ratio", "hosp_volume"]:   # sanity: unchanged hospital -> identical to the original feature
        ok = np.isclose(adapted_rows.loc[same_h, c].to_numpy(), xi.loc[adapted_rows.index[same_h], c].to_numpy()).all()
        assert ok, f"{c}: adapted rows with an unchanged hospital do not match the original feature"

    # ---- training sets ----
    tr_mask = X["split"].eq("train").to_numpy()
    base = X.loc[tr_mask].set_index("claim_id")[cols]
    y_base = Y.loc[tr_mask].set_index("claim_id")["is_fraud"].astype(int)
    if a.mode == "augment":
        X_train = pd.concat([base, adapted_rows])
        y_train = np.concatenate([y_base.to_numpy(), np.ones(len(adapted_rows), dtype=int)])
    else:
        keep = base.drop(index=adapted_rows.index)
        X_train = pd.concat([keep, adapted_rows])
        y_train = np.concatenate([y_base.loc[keep.index].to_numpy(), np.ones(len(adapted_rows), dtype=int)])
    pos_w = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    saved = f"{out_dir}/xgboost_{t}_{a.variant}_retrained_{a.mode}.joblib"
    if a.reuse_retrained and os.path.exists(saved):
        new = joblib.load(saved)
    else:
        new = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                            scale_pos_weight=pos_w, tree_method="hist", eval_metric="aucpr", random_state=42, n_jobs=-1)
        new.fit(X_train, y_train)
    orig = joblib.load(f"{d}/models/xgboost_{t}_{a.variant}.joblib")

    # ---- evaluation on A (clean test) and B (test adapted against the original model) ----
    te = X["split"].eq("test").to_numpy()
    lab = Y.loc[te].reset_index(drop=True)
    y = lab["is_fraud"].astype(int).to_numpy()
    assert Yte.loc[te, "is_fraud"].astype(int).to_numpy().tolist() == y.tolist(), "test labels changed"
    k = max(1, int(round(a.capacity * te.sum())))
    sets = {"A_clean_test": X.loc[te, cols], "B_adapted_vs_original": Xte.loc[te, cols]}
    if a.round2_adaptive_dir:
        X2 = pd.read_csv(f"{a.round2_adaptive_dir}/features_{t}_round1.csv")
        Y2 = pd.read_csv(f"{a.round2_adaptive_dir}/labels_{t}_round1.csv")
        assert X2["claim_id"].equals(X["claim_id"]) and Y2["is_fraud"].equals(Y["is_fraud"]), "round-2 rows do not line up"
        sets["C_adapted_vs_retrained"] = X2.loc[te, cols]
    rows, groups = [], []
    for sname, M in sets.items():
        for mname, mdl in [("Original XGBoost", orig), (f"Retrained ({a.mode})", new)]:
            r, g = evaluate(mname, sname, mdl.predict_proba(M)[:, 1], y, lab, k)
            rows.append(r)
            groups += g
    metrics, gm = pd.DataFrame(rows), pd.DataFrame(groups)
    metrics.to_csv(f"{out_dir}/retraining_v2_metrics.csv", index=False)
    gm.to_csv(f"{out_dir}/retraining_v2_group_recall.csv", index=False)
    joblib.dump(new, saved)

    print(f"\nMODE: {a.mode}")
    print(f"Training rows: {len(X_train):,}   fraud rows: {int(y_train.sum()):,}   (adapted copies: {len(adapted_rows):,})")
    print(f"Test rows: {te.sum():,}   K = {k:,}")
    print(metrics.round(4).to_string(index=False))
    pv = gm.pivot_table(index=["fraud_type", "severity_tier"], columns=["test_set", "model"], values="recall_at_K")
    print("\nRecall@K by fraud type and tier:")
    print(pv.round(3).to_string())
    print("\nSaved to", out_dir)


if __name__ == "__main__":
    main()
