"""Module 2, step 2: train Isolation Forest, Random Forest, XGBoost; evaluate on the test split.
Usage: python src/train_detectors.py --dir data_v3 --tag 5pct [--capacity 0.05]
Needs: pip install scikit-learn xgboost joblib
Outputs in --dir: scores_<tag>_<model>_<variant>.csv, models/<...>.joblib, metrics_<tag>.csv
"""
import argparse, json, os, joblib, numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

def main():
    a = argparse.ArgumentParser()
    a.add_argument("--dir", default="data_v3"); a.add_argument("--tag", default="5pct")
    a.add_argument("--capacity", type=float, default=0.05, help="review capacity as share of test claims")
    a = a.parse_args()
    d, t = a.dir, a.tag
    X = pd.read_csv(f"{d}/features_{t}.csv"); Y = pd.read_csv(f"{d}/labels_{t}.csv")
    fl = json.load(open(f"{d}/feature_list_{t}.json"))
    assert (X.claim_id == Y.claim_id).all()
    os.makedirs(f"{d}/models", exist_ok=True)

    tr, te = (X.split == "train").values, (X.split == "test").values
    y = Y.is_fraud.values
    pos_w = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
    variants = {"with_district": fl["features"] + fl["optional"], "no_district": fl["features"]}
    if not fl["optional"]: variants = {"no_district": fl["features"]}

    def make(name):
        if name == "isolation_forest": return IsolationForest(n_estimators=300, random_state=42, n_jobs=-1)
        if name == "random_forest":
            return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
                                          random_state=42, n_jobs=-1)
        return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                             scale_pos_weight=pos_w, tree_method="hist", eval_metric="aucpr", random_state=42, n_jobs=-1)

    K = int(round(a.capacity * te.sum()))
    rows = []
    for vname, cols in variants.items():
        for mname in ["isolation_forest", "random_forest", "xgboost"]:
            m = make(mname)
            if mname == "isolation_forest": m.fit(X.loc[tr, cols]); s = -m.score_samples(X[cols])   # higher = more anomalous
            else: m.fit(X.loc[tr, cols], y[tr]); s = m.predict_proba(X[cols])[:, 1]
            pd.DataFrame({"claim_id": X.claim_id, "split": X.split, "score": s}).to_csv(
                f"{d}/scores_{t}_{mname}_{vname}.csv", index=False)
            joblib.dump(m, f"{d}/models/{mname}_{t}_{vname}.joblib")

            st, yt = s[te], y[te]
            flag = np.zeros(len(st), bool); flag[np.argsort(-st)[:K]] = True
            r = {"dataset": t, "detector": mname, "variant": vname, "K": K,
                 "pr_auc": average_precision_score(yt, st),
                 "precision_at_K": yt[flag].mean(), "recall_at_K": yt[flag].sum() / yt.sum()}
            lab = Y[te]
            for (ft, tier), idx in lab[lab.is_fraud == 1].groupby(["fraud_type", "severity_tier"]).groups.items():
                pos = lab.index.get_indexer(idx)
                r[f"recall_{ft}_{tier}"] = flag[pos].mean()
            rows.append(r)
            print(f"{vname:14s} {mname:17s} PR-AUC={r['pr_auc']:.3f}  P@K={r['precision_at_K']:.3f}  R@K={r['recall_at_K']:.3f}")
    out = pd.DataFrame(rows)
    out.to_csv(f"{d}/metrics_{t}.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print("\nPer type/tier recall at K:\n", out.filter(regex="^(detector|variant|recall_[a-z]+_)").round(2).to_string(index=False))

if __name__ == "__main__":
    main()
