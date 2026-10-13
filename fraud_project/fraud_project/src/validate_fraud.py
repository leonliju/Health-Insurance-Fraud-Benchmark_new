#!/usr/bin/env python3
"""
Checks that an injected fraud dataset is consistent with its labels and with the clean data.

Usage:
  python src/validate_fraud.py --config config/params.json --data data_full --rate 0.05
Exits with status 1 if any hard check fails.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from inject_fraud import COLS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/params.json")
    ap.add_argument("--data", default="data")
    ap.add_argument("--rate", type=float, required=True)
    a = ap.parse_args()

    cfg = json.load(open(a.config, encoding="utf-8"))
    fc = cfg["fraud"]
    pk = pd.DataFrame(cfg["packages"]).set_index("package_code")
    tag = f"{int(round(a.rate * 100))}pct"
    cl = pd.read_csv(os.path.join(a.data, "claims_clean.csv")).set_index("claim_id")
    fr = pd.read_csv(os.path.join(a.data, f"claims_fraud_{tag}.csv"))
    lab = pd.read_csv(os.path.join(a.data, f"fraud_labels_{tag}.csv"), keep_default_na=False)
    hosp = pd.read_csv(os.path.join(a.data, "hospitals.csv"))
    docs = pd.read_csv(os.path.join(a.data, "doctors.csv")).set_index("doctor_id")
    pat = pd.read_csv(os.path.join(a.data, "patients.csv"))
    pat["date_of_birth"] = pd.to_datetime(pat["date_of_birth"])
    dob = pat.set_index("patient_id")["date_of_birth"]

    mult = hosp.set_index("hospital_id")["incentive_multiplier"]
    day_rates = {v["day_rate"] for k, v in cfg["bed_categories"].items() if not k.startswith("_")}
    clip = cfg["amount_ratio_clip"]

    fails = []

    def check(name, ok):
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            fails.append(name)

    fr["adm"] = pd.to_datetime(fr["admission_date"])
    fr["dis"] = pd.to_datetime(fr["discharge_date"])
    m = fr.merge(lab, on="claim_id", suffixes=("", "_lab"))
    check("columns match the shared 17-column schema", list(fr.columns[:17]) == COLS)
    check("labels file lines up with claims file", len(m) == len(fr) and fr["claim_id"].is_unique)
    check("label columns agree between the two files",
          bool((m["is_fraud"] == m["is_fraud_lab"]).all() and (m["fraud_type"] == m["fraud_type_lab"]).all()
               and (m["severity_tier"] == m["severity_tier_lab"]).all()))
    f = m[m["is_fraud"] == 1]
    g = m[m["is_fraud"] == 0]
    check(f"fraud share is {a.rate:.0%} (within 0.1 point)", abs(len(f) / len(m) - a.rate) < 0.001)
    check("is_fraud, fraud_type, severity_tier are consistent",
          bool(((f["fraud_type"] != "none") & (f["severity_tier"] != "none")).all()
               and ((g["fraud_type"] == "none") & (g["severity_tier"] == "none")).all()))
    check("fraud types and tiers are the intended ones",
          set(f["fraud_type"]) == {"upcoding", "duplicate", "phantom"} and set(f["severity_tier"]) == {"obvious", "subtle"})

    # ---- structural rules that must still hold for every row ----
    check("claim_id unique and no missing values", bool(fr["claim_id"].is_unique and fr[COLS].isna().sum().sum() == 0))
    check("dates inside the window and discharge after admission",
          bool((fr["adm"] >= cfg["window_start"]).all() and (fr["dis"] <= pd.Timestamp(cfg["window_end"]) + pd.Timedelta(days=60)).all()
               and (fr["dis"] > fr["adm"]).all()))
    check("admission dates inside the window", bool((fr["adm"] <= cfg["window_end"]).all()))
    check("length_of_stay equals discharge minus admission", bool(((fr["dis"] - fr["adm"]).dt.days == fr["length_of_stay"]).all()))
    check("ages between 0 and 95", bool(fr["patient_age"].between(0, 95).all()))
    age_calc = [int((a_ - dob[p]).days / 365.25) for p, a_ in zip(fr["patient_id"], fr["adm"])]
    check("patient_age matches date of birth at admission", bool((np.array(age_calc) == fr["patient_age"].to_numpy()).all()))
    d = fr.join(docs[["hospital_id", "specialty"]].rename(columns={"hospital_id": "doc_hosp", "specialty": "doc_spec"}), on="doctor_id")
    check("every doctor works at the claim's hospital", bool((d["hospital_id"] == d["doc_hosp"]).all()))
    spec = pk["specialty"].reindex(fr["procedure_package_code"]).to_numpy()
    check("doctor specialty matches the package specialty", bool((d["doc_spec"].to_numpy() == spec).all()))
    offers = hosp.set_index("hospital_id")["specialties"]
    check("hospital offers the package specialty",
          all(s in offers[h].split("|") for h, s in zip(fr["hospital_id"], spec)))

    # ---- pricing rules still hold for every row, fraud or not ----
    fx = (pk["pricing"].reindex(fr["procedure_package_code"]).to_numpy() == "fixed")
    rate_cfg = pk["reference_rate"].reindex(fr["procedure_package_code"]).to_numpy()
    check("fixed-price packages carry the HBP 2.2 reference rate on every row",
          bool((fr["reference_package_rate"].to_numpy()[fx] == rate_cfg[fx]).all()))
    daily = (fr["reference_package_rate"].to_numpy()[~fx] / fr["length_of_stay"].to_numpy()[~fx]).round(2)
    check("medical packages carry bed-days x an HBP 2.2 day rate on every row", bool(pd.Series(daily).isin(day_rates).all()))
    row_mult = mult.reindex(fr["hospital_id"]).to_numpy()
    ph_rows = (m["fraud_type"] == "phantom").to_numpy()
    ph_ratio = fr["claimed_amount"].to_numpy()[ph_rows] / (fr["reference_package_rate"].to_numpy()[ph_rows] * row_mult[ph_rows])
    check("phantom claims look like genuine billing (within 0.90 to 1.10 of reference x hospital multiplier)",
          bool(pd.Series(ph_ratio).between(clip[0] - 0.01, clip[1] + 0.01).all()))

    # ---- genuine rows are untouched ----
    clean_rows = m[m["clean_claim_id"] != ""]
    check("every clean claim appears exactly once", len(clean_rows) == len(cl) and clean_rows["clean_claim_id"].is_unique)
    gj = g.join(cl, on="clean_claim_id", rsuffix="_clean")
    same = all((gj[c] == gj[c + "_clean"]).all() for c in COLS if c not in ("claim_id",))
    check("genuine rows are identical to the clean file", bool(same))

    # ---- upcoding ----
    u = f[f["fraud_type"] == "upcoding"].join(cl, on="clean_claim_id", rsuffix="_clean")
    check("upcoding rows come from clean claims and keep patient, hospital, doctor, dates",
          bool(all((u[c] == u[c + "_clean"]).all() for c in ["patient_id", "hospital_id", "doctor_id", "admission_date", "discharge_date", "diagnosis_code"])))
    ratio = u["claimed_amount"] / (u["reference_package_rate_clean"] * mult.reindex(u["hospital_id"]).to_numpy())
    ai = u["mechanism"] == "amount_inflation"
    lo_o, hi_o = fc["upcoding"]["obvious_ratio"]
    lo_s, hi_s = fc["upcoding"]["subtle_ratio"]
    tol = 0.003
    check("obvious amount inflation is 1.6 to 2.5 times the hospital's legitimate price",
          bool(ratio[ai & (u["severity_tier"] == "obvious")].between(lo_o - tol, hi_o + tol).all()))
    check("subtle amount inflation is 1.10 to 1.25 times the hospital's legitimate price",
          bool(ratio[ai & (u["severity_tier"] == "subtle")].between(lo_s - tol, hi_s + tol).all()))
    sw = u[u["mechanism"] == "package_swap"]
    check("package swaps go to a pricier fixed-price package of the same specialty",
          bool(all(pk.loc[n, "specialty"] == pk.loc[o, "specialty"] and pk.loc[n, "reference_rate"] > pk.loc[o, "reference_rate"]
                   for n, o in zip(sw["procedure_package_code"], sw["procedure_package_code_clean"]))))
    check("subtle upcoding never changes the package", bool((u[u["severity_tier"] == "subtle"]["procedure_package_code"]
                                                           == u[u["severity_tier"] == "subtle"]["procedure_package_code_clean"]).all()))

    # ---- duplicates ----
    dup = f[f["fraud_type"] == "duplicate"]
    src = m.set_index("claim_id").loc[dup["linked_claim_id"]]
    src.index = dup.index
    check("duplicates link to a genuine claim of the same patient, diagnosis and package",
          bool(((src["is_fraud"] == 0).all() and (src["patient_id"] == dup["patient_id"]).all()
                and (src["diagnosis_code"] == dup["diagnosis_code"]).all()
                and (src["procedure_package_code"] == dup["procedure_package_code"]).all())))
    check("duplicates never copy a repeat-treatment package (dialysis, chemotherapy)",
          bool((pk["repeat_mean"].reindex(dup["procedure_package_code"]).to_numpy() == 0).all()))
    shift = (dup["adm"] - pd.to_datetime(src["admission_date"])).dt.days.abs()
    ob = dup["severity_tier"] == "obvious"
    check("obvious duplicates: other hospital, within 2 days, identical amount",
          bool(((dup[ob]["hospital_id"] != src[ob]["hospital_id"]).all()) and (shift[ob] <= fc["duplicate"]["obvious_shift_days"][1]).all()
               and (dup[ob]["claimed_amount"] == src[ob]["claimed_amount"]).all()))
    sb = ~ob
    amt_ratio = dup[sb]["claimed_amount"] / src[sb]["claimed_amount"]
    clipped = (shift[sb] < fc["duplicate"]["subtle_shift_days"][0])  # shifted back/clipped at window edges
    check("subtle duplicates: 3 to 14 days apart (except at window edges), amount within 3.5%",
          bool(((shift[sb] <= fc["duplicate"]["subtle_shift_days"][1]).all() and amt_ratio.between(0.965, 1.035).all()
                and clipped.mean() < 0.02)))

    # ---- phantom ----
    ph = f[f["fraud_type"] == "phantom"]
    po = ph[ph["severity_tier"] == "obvious"]
    sex_rule = pk["sex"].reindex(po["procedure_package_code"]).to_numpy()
    z = (po["patient_age"].to_numpy() - pk["age_mean"].reindex(po["procedure_package_code"]).to_numpy()) \
        / pk["age_sd"].reindex(po["procedure_package_code"]).to_numpy()
    sm = po["mechanism"] == "sex_mismatch"
    check("obvious phantom (sex): package restricted to the other sex",
          bool(all((r not in ("any", s_)) for r, s_ in zip(sex_rule[sm.to_numpy()], po[sm]["patient_sex"]))))
    check("obvious phantom (age): age is more than 3.5 spreads from the package's typical age",
          bool((np.abs(z[~sm.to_numpy()]) > fc["phantom"]["age_z_threshold"]).all()))
    ps = ph[ph["severity_tier"] == "subtle"]
    s2 = m.set_index("claim_id").loc[ps["linked_claim_id"]]
    s2.index = ps.index
    s_adm, s_dis = pd.to_datetime(s2["admission_date"]), pd.to_datetime(s2["discharge_date"])
    check("subtle phantom: same patient, other hospital, starts inside the other stay",
          bool(((s2["patient_id"] == ps["patient_id"]).all() and (s2["hospital_id"] != ps["hospital_id"]).all()
                and (ps["adm"] >= s_adm).all() and (ps["adm"] < s_dis).all())))
    plaus_sex = all(r in ("any", s_) for r, s_ in zip(pk["sex"].reindex(ps["procedure_package_code"]), ps["patient_sex"]))
    check("subtle phantom: package is allowed for the patient's sex", bool(plaus_sex))

    # ---- information only ----
    print("\n--- Summary ---")
    print(f"claims: {len(fr):,}   fraud: {len(f):,} ({len(f) / len(fr):.2%})")
    print(pd.crosstab(f["fraud_type"], f["severity_tier"], margins=True).to_string())
    print("\nMechanisms:")
    print(f.groupby(["fraud_type", "mechanism"]).size().to_string())
    per_h = f["hospital_id"].value_counts()
    top = max(1, int(round(0.10 * fr["hospital_id"].nunique())))
    print(f"\nShare of fraud at the top 10% of hospitals by fraud count: {per_h.head(top).sum() / len(f):.0%} "
          f"(top 10% by claim volume would be about 10% to 25%)")
    gg = g.sort_values(["patient_id", "adm"]).copy()
    prev_dis = gg.groupby("patient_id")["dis"].shift()
    overlap = int((gg["adm"] < prev_dis).sum())
    print(f"Genuine claims that start before the same patient's previous genuine stay ended (accidental overlaps): "
          f"{overlap:,} of {len(g):,} ({overlap / len(g):.2%})")
    print("\nResult:", "ALL CHECKS PASSED" if not fails else f"{len(fails)} CHECK(S) FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
