#!/usr/bin/env python3
"""
Sanity checks for the clean synthetic claims dataset.

Usage:
    python src/validate_claims.py --config config/params.json --data data
Exits with status 1 if any hard check fails.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

EXPECTED_COLUMNS = [
    "claim_id", "patient_id", "hospital_id", "doctor_id",
    "admission_date", "discharge_date",
    "patient_age", "patient_sex", "patient_district",
    "diagnosis_code", "procedure_package_code",
    "claimed_amount", "reference_package_rate", "length_of_stay",
    "is_fraud", "fraud_type", "severity_tier",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/params.json")
    ap.add_argument("--data", default="data")
    a = ap.parse_args()

    cfg = json.load(open(a.config, encoding="utf-8"))
    claims = pd.read_csv(os.path.join(a.data, "claims_clean.csv"))
    hospitals = pd.read_csv(os.path.join(a.data, "hospitals.csv"))
    doctors = pd.read_csv(os.path.join(a.data, "doctors.csv"))
    packages = pd.read_csv(os.path.join(a.data, "packages.csv"))
    inc = cfg["hospital_attributes"]["incentive"]
    day_rates = {v["day_rate"] for k, v in cfg["bed_categories"].items() if not k.startswith("_")}
    clip = cfg["amount_ratio_clip"]

    fails = []

    def check(name, ok):
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            fails.append(name)

    adm = pd.to_datetime(claims["admission_date"])
    dis = pd.to_datetime(claims["discharge_date"])

    check("columns match the shared schema (17, in order)", list(claims.columns) == EXPECTED_COLUMNS)
    check("no missing values", int(claims.isna().sum().sum()) == 0)
    check("claim_id is unique", claims["claim_id"].is_unique)
    check("admission dates inside the window",
          adm.min() >= pd.Timestamp(cfg["window_start"]) and adm.max() <= pd.Timestamp(cfg["window_end"]))
    check("discharge is after admission", bool((dis > adm).all()))
    check("length_of_stay equals discharge minus admission", bool(((dis - adm).dt.days == claims["length_of_stay"]).all()))
    check("length_of_stay >= 1", bool((claims["length_of_stay"] >= 1).all()))
    check("ages between 0 and 95", bool(claims["patient_age"].between(0, 95).all()))
    check("claimed_amount > 0", bool((claims["claimed_amount"] > 0).all()))
    check("clean file: is_fraud is all 0", bool((claims["is_fraud"] == 0).all()))
    check("clean file: fraud_type and severity_tier are 'none'",
          bool((claims["fraud_type"] == "none").all() and (claims["severity_tier"] == "none").all()))

    m = claims.merge(packages, left_on="procedure_package_code", right_on="package_code", how="left", suffixes=("", "_pkg"))
    check("every package code exists in packages.csv", bool(m["package_name"].notna().all()))
    check("diagnosis code matches the package", bool((m["diagnosis_code"] == m["diagnosis_code_pkg"]).all()))
    check("sex restrictions respected (e.g. obstetrics = female)", bool(((m["sex"] == "any") | (m["sex"] == m["patient_sex"])).all()))

    # ---- pricing ----
    fixed = m["pricing"] == "fixed"
    check("fixed-price packages: reference rate equals the HBP 2.2 rate",
          bool((m.loc[fixed, "reference_package_rate"] == m.loc[fixed, "reference_package_rate_pkg"]).all()))
    per_day = ~fixed
    daily = (m.loc[per_day, "reference_package_rate"] / m.loc[per_day, "length_of_stay"]).round(2)
    check("medical packages: reference rate is bed-days x an HBP 2.2 day rate",
          bool(daily.isin(day_rates).all()))
    hm = hospitals.set_index("hospital_id")
    expected = (np.where(hm["nabh_level"] == "full", 1 + inc["nabh_full"], np.where(hm["nabh_level"] == "entry", 1 + inc["nabh_entry"], 1.0))
                * np.where(hm["is_metro"] == 1, 1 + inc["metro"], 1.0) * np.where(hm["is_aspirational"] == 1, 1 + inc["aspirational"], 1.0)
                * np.where(hm["has_pg_dnb"] == 1, 1 + inc["pg_dnb"], 1.0))
    check("hospital incentive multiplier equals the compounded NHA incentives",
          bool(np.allclose(hm["incentive_multiplier"].to_numpy(), expected, atol=1e-3)))
    mult = hm["incentive_multiplier"].reindex(claims["hospital_id"]).to_numpy()
    ratio = claims["claimed_amount"] / (claims["reference_package_rate"] * mult)
    check(f"genuine claimed amount is within {clip[0]:.2f} to {clip[1]:.2f} of reference x hospital multiplier (allowing rounding)",
          bool(ratio.between(clip[0] - 0.01, clip[1] + 0.01).all()))

    # ---- official-rule realism ----
    pub = hm["hospital_type"].reindex(claims["hospital_id"]).to_numpy() == "public"
    res = m["public_reserved"].to_numpy().astype(bool)
    share_pub = pub[res].mean() if res.any() else 1.0
    check(f"caesarean and high risk delivery are mostly at public hospitals ({share_pub:.0%}; target about {1 - cfg['private_referral_share']:.0%})",
          share_pub >= 0.85)
    ok_rules = True
    c2 = claims.assign(adm=adm).merge(packages[["package_code", "max_per_patient", "min_gap_days", "repeat_mean"]],
                                      left_on="procedure_package_code", right_on="package_code")
    c2 = c2.sort_values(["patient_id", "procedure_package_code", "adm"])
    for (pid, code), g in c2.groupby(["patient_id", "procedure_package_code"]):
        if len(g) == 1 or g["repeat_mean"].iloc[0] > 0:   # dialysis and chemotherapy repeat by design
            continue
        mp, mg = int(g["max_per_patient"].iloc[0]), int(g["min_gap_days"].iloc[0])
        if mp > 0 and len(g) > mp:
            ok_rules = False
            break
        if mg > 0 and (g["adm"].diff().dt.days.dropna() < mg).any():
            ok_rules = False
            break
    check("one-time surgeries are not repeated and minimum gaps are respected (dialysis and chemotherapy series are exempt)",
          ok_rules)

    d = claims.join(doctors.set_index("doctor_id")[["hospital_id", "specialty"]].rename(columns={"hospital_id": "doc_hosp", "specialty": "doc_spec"}), on="doctor_id")
    check("every doctor works at the claim's hospital", bool((d["hospital_id"] == d["doc_hosp"]).all()))
    spec = packages.set_index("package_code")["specialty"].reindex(claims["procedure_package_code"]).to_numpy()
    check("doctor specialty matches the package specialty", bool((d["doc_spec"].to_numpy() == spec).all()))
    offers = hospitals.set_index("hospital_id")["specialties"]
    check("hospital offers the package specialty", all(s in offers[h].split("|") for h, s in zip(claims["hospital_id"], spec)))

    # ---- summary ----
    print("\n--- Summary ---")
    cr = claims["claimed_amount"] / claims["reference_package_rate"]
    print(f"claims: {len(claims):,}   patients: {claims['patient_id'].nunique():,}   "
          f"hospitals used: {claims['hospital_id'].nunique():,}   doctors used: {claims['doctor_id'].nunique():,}")
    print(f"claims per patient (mean): {len(claims) / claims['patient_id'].nunique():.2f}")
    print(f"claimed / reference ratio: mean {cr.mean():.3f}, range {cr.min():.2f} to {cr.max():.2f} (hospital uplift lifts it above 1)")
    print(f"hospital multiplier: mean {hospitals['incentive_multiplier'].mean():.3f}, max {hospitals['incentive_multiplier'].max():.3f}")
    print(f"length of stay (mean): {claims['length_of_stay'].mean():.2f} days")
    same = claims.duplicated(subset=["patient_id", "procedure_package_code"], keep=False)
    print(f"claims sharing patient+package with another claim (genuine repeats): {same.mean():.1%}")
    mix = (claims["procedure_package_code"].value_counts(normalize=True)
           .rename_axis("package_code").reset_index(name="share")
           .merge(packages[["package_code", "package_name"]], on="package_code"))
    mix["share"] = (mix["share"] * 100).round(1).astype(str) + "%"
    print("\nPackage mix:")
    print(mix.to_string(index=False))
    print("\nResult:", "ALL CHECKS PASSED" if not fails else f"{len(fails)} CHECK(S) FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
