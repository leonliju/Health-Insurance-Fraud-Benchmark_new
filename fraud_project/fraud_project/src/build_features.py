"""Module 2, step 1: build features + patient-level split.
Usage: python src/build_features.py --claims data_v3/claims_fraud_5pct.csv \
        --hospitals data_v3/hospitals.csv --params config/params.json --out data_v3 --tag 5pct
"""
import argparse, json, numpy as np, pandas as pd

ANSWER_KEY = {"is_fraud", "fraud_type", "severity_tier",
              "mechanism", "clean_claim_id", "linked_claim_id"}

def main():
    a = argparse.ArgumentParser()
    a.add_argument("--claims", required=True); a.add_argument("--hospitals", required=True)
    a.add_argument("--params", default="config/params.json")
    a.add_argument("--out", default="."); a.add_argument("--tag", default="5pct")
    a = a.parse_args()

    p = json.load(open(a.params)); seed = p["seed"]
    pk = pd.DataFrame(p["packages"]).set_index("package_code")
    df = pd.read_csv(a.claims, parse_dates=["admission_date", "discharge_date"])
    hos = pd.read_csv(a.hospitals)
    print("hospitals.csv columns:", list(hos.columns))

    # ---- hospital attributes (no labels) ----
    mcol = next(c for c in hos.columns if "multiplier" in c.lower())
    dcol = next((c for c in hos.columns if "district" in c.lower() and "aspir" not in c.lower()), None)
    h = hos.rename(columns={mcol: "hosp_multiplier"})
    hfeat = ["hosp_multiplier"]
    for c in h.columns:
        if c in ("hospital_id", "hosp_multiplier", dcol, "specialties", "size_weight") or "name" in c.lower(): continue
        if h[c].dtype == object or str(h[c].dtype) == "str":
            d = pd.get_dummies(h[c], prefix="h_" + c).astype(int); h = pd.concat([h, d], axis=1); hfeat += list(d.columns)
        else:
            h[c] = h[c].astype(float); hfeat.append(c)
    df = df.merge(h[["hospital_id"] + hfeat + ([dcol] if dcol else [])], on="hospital_id", how="left")
    assert df.hosp_multiplier.notna().all(), "claims reference unknown hospitals"

    # ---- amount ----
    df["raw_ratio"] = df.claimed_amount / df.reference_package_rate
    df["adj_ratio"] = df.raw_ratio / df.hosp_multiplier

    # ---- plausibility ----
    code = df.procedure_package_code
    psex = code.map(pk.sex)
    df["sex_violation"] = ((psex != "any") & (psex != df.patient_sex)).astype(int)
    df["age_dist"] = (df.patient_age - code.map(pk.age_mean)).abs() / code.map(pk.age_sd)
    df["repeat_pkg"] = (code.map(pk.repeat_max) > 0).astype(int)

    # ---- timing (same patient + same package) ----
    df = df.sort_values(["patient_id", "procedure_package_code", "admission_date", "claim_id"])
    g = df.groupby(["patient_id", "procedure_package_code"])
    BIG = 999
    df["gap_prev"] = (df.admission_date - g.admission_date.shift(1)).dt.days.fillna(BIG)
    df["gap_next"] = (g.admission_date.shift(-1) - df.admission_date).dt.days.fillna(BIG)
    df["same_pkg_count"] = g.claim_id.transform("count")
    df["claims_per_patient"] = df.groupby("patient_id").claim_id.transform("count")

    # ---- overlap with another stay of the same patient ----
    s = df[["claim_id", "patient_id", "hospital_id", "admission_date", "discharge_date"]]
    m = s.merge(s, on="patient_id", suffixes=("", "_o"))
    m = m[(m.claim_id != m.claim_id_o) & (m.admission_date <= m.discharge_date_o)
          & (m.discharge_date >= m.admission_date_o)]
    ov = m.groupby("claim_id").size().rename("n_overlap")
    ovd = m[m.hospital_id != m.hospital_id_o].groupby("claim_id").size().rename("n_overlap_diff_hosp")
    df = df.merge(ov, on="claim_id", how="left").merge(ovd, on="claim_id", how="left")
    df[["n_overlap", "n_overlap_diff_hosp"]] = df[["n_overlap", "n_overlap_diff_hosp"]].fillna(0)
    df["overlap_any"] = (df.n_overlap > 0).astype(int)
    df["overlap_diff_hosp"] = (df.n_overlap_diff_hosp > 0).astype(int)

    # ---- optional clue: hospital in patient's own district ----
    optional = []
    if dcol:
        df["in_district"] = (df[dcol] == df.patient_district).astype(int); optional = ["in_district"]

    # ---- patient-level split 60/20/20 ----
    pats = np.array(sorted(df.patient_id.unique()))
    rng = np.random.default_rng(seed); rng.shuffle(pats)
    n = len(pats); cut1, cut2 = int(.6 * n), int(.8 * n)
    smap = {**{x: "train" for x in pats[:cut1]}, **{x: "calib" for x in pats[cut1:cut2]},
            **{x: "test" for x in pats[cut2:]}}
    df["split"] = df.patient_id.map(smap)
    assert df.groupby("patient_id").split.nunique().max() == 1, "patient in >1 split"

    # ---- hospital-level features (volume all claims; mean ratio from TRAIN only) ----
    df["hosp_volume"] = df.groupby("hospital_id").claim_id.transform("count")
    tr = df[df.split == "train"].groupby("hospital_id").adj_ratio.mean()
    df["hosp_mean_adj_ratio"] = df.hospital_id.map(tr).fillna(df[df.split == "train"].adj_ratio.mean())

    num = ["raw_ratio", "adj_ratio", "sex_violation", "age_dist", "repeat_pkg", "gap_prev", "gap_next",
           "same_pkg_count", "overlap_any", "overlap_diff_hosp", "n_overlap", "claims_per_patient",
           "hosp_volume", "hosp_mean_adj_ratio"]
    features = num + hfeat
    assert not (set(features + optional) & ANSWER_KEY), "answer-key column leaked into features"

    df = df.sort_values("claim_id")
    cols = ["claim_id", "split"] + features + optional
    df[cols].to_csv(f"{a.out}/features_{a.tag}.csv", index=False)
    df[["claim_id", "split", "is_fraud", "fraud_type", "severity_tier"]].to_csv(f"{a.out}/labels_{a.tag}.csv", index=False)
    json.dump({"features": features, "optional": optional}, open(f"{a.out}/feature_list_{a.tag}.json", "w"), indent=1)
    print(df.groupby("split").agg(rows=("claim_id", "size"), fraud=("is_fraud", "mean")).round(4))
    print(len(features), "features +", optional, "optional; saved to", a.out)

if __name__ == "__main__":
    main()
