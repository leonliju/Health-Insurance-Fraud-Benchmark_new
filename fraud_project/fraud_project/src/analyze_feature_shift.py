"""
Feature-distribution analysis for the adaptive fraudster.

Shows which features the fraudster changed, by fraud type and tier, and draws the amount-ratio histogram
(genuine claims vs fraud before adaptation vs fraud after adaptation).

Usage:
  python src/analyze_feature_shift.py --dir data_v3 --tag 5pct --split test
  (--split test uses data_v3/adaptive/5pct ; --split train uses data_v3/adaptive/5pct_train_adaptation)

Outputs (in the adaptive folder):
  feature_shift_summary.csv   mean and median of every feature before/after, plus the share of claims that changed
  adj_ratio_histogram.png     the amount-ratio histogram, with a zoom on 0.9 to 1.3
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data_v3")
    ap.add_argument("--tag", default="5pct")
    ap.add_argument("--split", choices=["test", "train"], default="test")
    a = ap.parse_args()
    d, t = a.dir, a.tag
    ad = f"{d}/adaptive/{t}" if a.split == "test" else f"{d}/adaptive/{t}_train_adaptation"

    fl = json.load(open(f"{d}/feature_list_{t}.json"))
    cols = fl["features"] + fl.get("optional", [])
    X0 = pd.read_csv(f"{d}/features_{t}.csv")
    Y = pd.read_csv(f"{d}/labels_{t}.csv")
    X1 = pd.read_csv(f"{ad}/features_{t}_round1.csv")
    assert X0["claim_id"].equals(X1["claim_id"]) and X0["claim_id"].equals(Y["claim_id"])
    summ = pd.read_csv(f"{ad}/adaptation_summary.csv", dtype={"claim_id": str})
    changed = set(summ.loc[summ["changed"].astype(bool), "claim_id"])

    target = Y["split"].eq(a.split) & Y["is_fraud"].eq(1)
    X0, X1 = X0.set_index("claim_id"), X1.set_index("claim_id")
    ids = Y.loc[target, "claim_id"]
    lab = Y.loc[target].set_index("claim_id")

    # ---- table: before / after for every feature, by fraud type and tier ----
    rows = []
    for (ft, tier), g in lab.groupby(["fraud_type", "severity_tier"]):
        gid = g.index
        mod = gid.isin(changed)
        for c in cols:
            b, aft = X0.loc[gid, c].astype(float), X1.loc[gid, c].astype(float)
            moved = ~np.isclose(b.to_numpy(), aft.to_numpy())
            rows.append({"fraud_type": ft, "severity_tier": tier, "feature": c, "claims": len(gid),
                         "claims_modified": int(mod.sum()),
                         "share_feature_changed": round(float(moved.mean()), 3),
                         "mean_before": round(float(b.mean()), 4), "mean_after": round(float(aft.mean()), 4),
                         "median_before": round(float(b.median()), 4), "median_after": round(float(aft.median()), 4)})
    table = pd.DataFrame(rows)
    table.to_csv(f"{ad}/feature_shift_summary.csv", index=False)
    changed_feats = table[table["share_feature_changed"] > 0.05]
    print(f"\nFeatures that changed for more than 5% of claims in at least one fraud group ({a.split} split):")
    print(changed_feats[["fraud_type", "severity_tier", "feature", "share_feature_changed",
                         "median_before", "median_after"]].to_string(index=False))

    # ---- amount-ratio histogram ----
    gen = X0.loc[Y.loc[Y["split"].eq(a.split) & Y["is_fraud"].eq(0), "claim_id"], "adj_ratio"]
    up = lab[lab["fraud_type"] == "upcoding"].index
    before, after = X0.loc[up, "adj_ratio"], X1.loc[up, "adj_ratio"]
    spike = ((after > 1.059) & (after < 1.061)).mean()
    spike_gen = ((gen > 1.059) & (gen < 1.061)).mean()
    print(f"\nUpcoding claims ({len(up):,}): median adjusted ratio {before.median():.3f} -> {after.median():.3f}")
    print(f"Share within 1.059 to 1.061 after adaptation: {spike:.1%}  (genuine claims: {spike_gen:.1%})")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, rng_, bins, title in [(axes[0], (0.8, 2.6), 80, "Full range"), (axes[1], (0.9, 1.3), 80, "Zoom: 0.9 to 1.3")]:
        ax.hist(gen, bins=bins, range=rng_, density=True, alpha=0.45, color="grey", label="Genuine claims")
        ax.hist(before, bins=bins, range=rng_, density=True, alpha=0.55, color="tab:red", label="Upcoding before adaptation")
        ax.hist(after, bins=bins, range=rng_, density=True, alpha=0.55, color="tab:blue", label="Upcoding after adaptation")
        ax.set_title(title)
        ax.set_xlabel("Hospital-adjusted amount ratio (adj_ratio)")
    axes[0].set_ylabel("Density")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Amount ratio before and after adaptation ({a.split} fraud claims, {t})")
    fig.tight_layout()
    fig.savefig(f"{ad}/adj_ratio_histogram.png", dpi=150)
    print(f"\nSaved: {ad}/feature_shift_summary.csv and {ad}/adj_ratio_histogram.png")


if __name__ == "__main__":
    main()
