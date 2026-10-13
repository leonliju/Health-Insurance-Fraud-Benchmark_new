#!/usr/bin/env python3
"""
Builds data/parameter_sources.csv: every parameter, its value, where it comes from and whether it is
sourced from an official document or is an assumption. Also checks that each package rate in the
config appears in the extracted HBP 2.2 master (reference/hbp_2_2_master_extracted.csv).

Usage: python src/make_parameter_table.py --config config/params.json --hbp reference/hbp_2_2_master_extracted.csv --out data
"""
import argparse
import json
import os
import re
import sys

import pandas as pd

HBP = "AB PM-JAY Health Benefit Package master HBP 2.2 (National Health Authority), master PDF"
GUIDE = "NHA, National Health Benefit Package 2.2 User Guidelines (November 2021)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/params.json")
    ap.add_argument("--hbp", default="reference/hbp_2_2_master_extracted.csv")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    cfg = json.load(open(a.config, encoding="utf-8"))
    hbp = pd.read_csv(a.hbp, dtype=str).fillna("")
    rows, problems = [], []

    def add(group, parameter, value, source, status, note=""):
        rows.append(dict(group=group, parameter=parameter, value=value, source=source, status=status, note=note))

    # ---- package rates: verified against the extracted master ----
    for p in cfg["packages"]:
        code = p["hbp_code"].split("/")[0]
        r = hbp[hbp["procedure_code"] == code]
        if r.empty:
            problems.append(f"{p['package_code']}: HBP code {code} not found")
            continue
        r = r.iloc[0]
        rate = str(int(p["reference_rate"]))
        text = (r["total_price"] + " " + r["procedure_price"] + " " + r["stratification"]).replace(",", "")
        found = re.search(rf"(?<!\d){rate}(?!\d)", text) is not None
        if not found:
            problems.append(f"{p['package_code']}: rate {rate} not found for {code} in the master")
        unit = "per bed-day (routine ward)" if p["pricing"] == "per_day" else "per package"
        add("package rate", f"{p['package_code']} {p['package_name']}: reference rate ({unit})", int(p["reference_rate"]),
            f"{HBP}, code {p['hbp_code']}, PDF page {r['page']}", "sourced",
            "verified against extracted master" if found else "NOT VERIFIED")
    for k, v in cfg["bed_categories"].items():
        if k.startswith("_"):
            continue
        add("bed-day rate", f"{k} day rate", v["day_rate"], f"{HBP}, medical package stratification (e.g. MG016A pneumonia)", "sourced")
        add("bed-day mix", f"{k} share of medical admissions", v["share"], "", "assumption")
    inc = cfg["hospital_attributes"]["incentive"]
    for k, v in inc.items():
        add("hospital incentive", f"{k} uplift", v, f"{GUIDE}, FAQ 26 (compounded; state discretion)", "sourced",
            "assumed to apply at every qualifying hospital")
    for k, v in cfg["hospital_attributes"].items():
        if k.endswith("_share"):
            add("hospital attributes", k, v, "", "assumption")
    add("official rule", "caesarean and high risk delivery mainly at public hospitals", "yes", f"{GUIDE}, 3.13 and 4.1", "sourced")
    add("official rule", "share of those claims at private hospitals (by referral)", cfg["private_referral_share"], "", "assumption")
    add("official rule", "IT system limits repeats (minimum gaps, maximum bookings) for e.g. appendicectomy, cataract", "exists", f"{GUIDE}, FAQ 34c-d", "sourced",
        "the exact numbers (max per patient, minimum gap) are assumptions")
    for p in cfg["packages"]:
        if p["max_per_patient"] or p["min_gap_days"]:
            add("repeat limits", f"{p['package_code']} max per patient / minimum gap (days)", f"{p['max_per_patient']} / {p['min_gap_days']}", "", "assumption")
    # ---- assumptions ----
    for p in cfg["packages"]:
        add("package frequency", f"{p['package_code']} relative weight", p["weight"], "", "assumption")
        add("patient profile", f"{p['package_code']} typical age (mean, sd) and sex rule", f"{p['age_mean']}, {p['age_sd']}; {p['sex']}", "", "assumption")
        add("length of stay", f"{p['package_code']} mean, sd (days)", f"{p['los_mean']}, {p['los_sd']}", "", "assumption",
            "HBP sets no minimum or maximum stay (user guidelines FAQ 41)")
    add("repeat treatment", "dialysis: ~every 3 days (2-4), mean repeats 110", "", "checked against a 1,164-patient Synthea run (US-based simulator)", "checked against a simulator")
    add("repeat treatment", "breast-cancer chemotherapy: ~22-day cycles (20-24), mean 7 repeats", "", "checked against the same Synthea run", "checked against a simulator")
    for k in ["n_patients", "n_hospitals", "n_districts", "extra_claims_lambda", "female_share", "share_public_hospitals",
              "local_hospital_share", "doctor_extra_lambda", "amount_ratio_sigma"]:
        add("scale and mix", k, cfg[k], "", "assumption")
    add("scale and mix", "age buckets", json.dumps(cfg["age_buckets"]), "", "assumption")
    fc = cfg["fraud"]
    add("fraud design", "type shares", json.dumps(fc["type_shares"]), "", "assumption", "definitions follow NHA's own list of abuse concerns (guidelines 2.2.4, 2.2.10, FAQ 34)")
    add("fraud design", "share obvious", fc["obvious_share"], "", "assumption")
    add("fraud design", "clustering (share of hospitals prone, share of fraud there)", json.dumps(fc["cluster"]), "", "assumption")
    for k in ["upcoding", "duplicate", "phantom"]:
        add("fraud design", f"{k} settings", json.dumps(fc[k]), "", "assumption")

    out = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    out.to_csv(os.path.join(a.out, "parameter_sources.csv"), index=False)
    n_src = int(out["status"].str.startswith("sourced").sum())
    print(f"{len(out)} parameters: {n_src} sourced from official documents, "
          f"{int((out['status'] == 'assumption').sum())} assumptions, {int(out['status'].str.startswith('checked').sum())} checked against a simulator")
    if problems:
        print("PROBLEMS:")
        for x in problems:
            print("  -", x)
        sys.exit(1)
    print("All 16 package rates were found in the extracted HBP 2.2 master.")


if __name__ == "__main__":
    main()
