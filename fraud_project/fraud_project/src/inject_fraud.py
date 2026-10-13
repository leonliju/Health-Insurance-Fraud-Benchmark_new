#!/usr/bin/env python3
"""
Inject labelled fraud into the clean claims table.

Three fraud types, each with an 'obvious' and a 'subtle' tier:
  upcoding   in place: inflate the amount (relative to the hospital's legitimate price), or swap to a pricier package
  duplicate  new row: copy of a genuine claim (other hospital / resubmission)
  phantom    new row: obvious = package impossible for the patient (sex/age);
             subtle  = plausible package but overlapping another stay of the same patient

Fraud is clustered: about 10% of hospitals ("fraud-prone") receive about 60% of fraud.
All settings are in config/params.json under "fraud".

Outputs per fraud rate (e.g. 5%):
  claims_fraud_5pct.csv   the 17-column shared table (answer key filled in)
  fraud_labels_5pct.csv   extra detail: mechanism, source clean claim, linked claim

Usage:
  python src/inject_fraud.py --config config/params.json --data data_full --out data_full --rates 0.02 0.05 0.08
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

COLS = [
    "claim_id", "patient_id", "hospital_id", "doctor_id", "admission_date", "discharge_date",
    "patient_age", "patient_sex", "patient_district", "diagnosis_code", "procedure_package_code",
    "claimed_amount", "reference_package_rate", "length_of_stay", "is_fraud", "fraud_type", "severity_tier",
]


def age_at(dob, date):
    return int((date - dob).days / 365.25)


def inject(clean, patients, hospitals, doctors, cfg, rate):
    fc = cfg["fraud"]
    rng = np.random.default_rng(fc["seed"] + int(round(rate * 1000)))
    pk = pd.DataFrame(cfg["packages"]).set_index("package_code")
    start, end = pd.Timestamp(cfg["window_start"]), pd.Timestamp(cfg["window_end"])
    sigma, clip = cfg["amount_ratio_sigma"], cfg["amount_ratio_clip"]

    df = clean.copy()
    df["admission_date"] = pd.to_datetime(df["admission_date"])
    df["discharge_date"] = pd.to_datetime(df["discharge_date"])
    df["clean_claim_id"] = df["claim_id"]
    df["mechanism"] = ""
    df["linked_clean_id"] = ""
    n_clean = len(df)

    pat = patients.copy()
    pat["date_of_birth"] = pd.to_datetime(pat["date_of_birth"])
    dob = pat.set_index("patient_id")["date_of_birth"]
    pat_info = pat.set_index("patient_id")

    hosp_ids = hospitals["hospital_id"].to_numpy()
    mult_h = dict(zip(hospitals["hospital_id"], hospitals["incentive_multiplier"]))
    hosp_by_spec = {}
    for hid, specs in zip(hospitals["hospital_id"], hospitals["specialties"]):
        for s in specs.split("|"):
            hosp_by_spec.setdefault(s, []).append(hid)
    doc_by = {k: g["doctor_id"].to_numpy() for k, g in doctors.groupby(["hospital_id", "specialty"])}

    # ---- fraud-prone hospitals -------------------------------------------------
    n_prone = max(1, int(round(fc["cluster"]["prone_hospital_share"] * len(hosp_ids))))
    prone = set(rng.choice(hosp_ids, size=n_prone, replace=False).tolist())
    p_cl = fc["cluster"]["share_of_fraud_at_prone"]
    s = df["hospital_id"].isin(prone).mean()
    W = (p_cl / (1 - p_cl)) * ((1 - s) / s) if 0 < s < 1 else 1.0
    row_w = np.where(df["hospital_id"].isin(prone), W, 1.0)

    def pick_hospital(spec, exclude=None):
        cands = [h for h in hosp_by_spec[spec] if h != exclude]
        pr = [h for h in cands if h in prone]
        other = [h for h in cands if h not in prone]
        if pr and (rng.random() < p_cl or not other):
            return rng.choice(pr)
        return rng.choice(other if other else cands)

    def pick_doctor(h, spec):
        return rng.choice(doc_by[(h, spec)])

    def draw_los(code):
        m, sd = pk.loc[code, "los_mean"], pk.loc[code, "los_sd"]
        return int(np.clip(round(rng.normal(m, sd)), 1, np.ceil(m + 3 * sd)))

    bed_names = [k for k in cfg["bed_categories"] if not k.startswith("_")]
    bed_p = np.array([cfg["bed_categories"][k]["share"] for k in bed_names], dtype=float)
    bed_p /= bed_p.sum()
    bed_r = np.array([cfg["bed_categories"][k]["day_rate"] for k in bed_names], dtype=float)

    def draw_ref(code, los):
        """Reference rate for a new claim: fixed package price, or bed-days x day rate for medical packages."""
        if pk.loc[code, "pricing"] == "per_day":
            return int(los * bed_r[rng.choice(len(bed_r), p=bed_p)])
        return int(pk.loc[code, "reference_rate"])

    def draw_amount(ref, h):
        """Genuine-looking amount: reference x the hospital's incentive multiplier x small noise."""
        r = float(np.clip(rng.lognormal(0, sigma), clip[0], clip[1]))
        return int(round(ref * mult_h[h] * r / 10) * 10)

    # ---- how many of each ------------------------------------------------------
    sh = fc["type_shares"]
    n_fraud = int(round(rate * n_clean / (1 - rate * (1 - sh["upcoding"]))))
    n_up = int(round(n_fraud * sh["upcoding"]))
    n_dup = int(round(n_fraud * sh["duplicate"]))
    n_ph = n_fraud - n_up - n_dup
    ob = fc["obvious_share"]

    def tiers(n):
        n_o = int(round(n * ob))
        t = np.array(["obvious"] * n_o + ["subtle"] * (n - n_o))
        rng.shuffle(t)
        return t

    used = np.zeros(n_clean, dtype=bool)
    is_repeat = pk.loc[df["procedure_package_code"], "repeat_mean"].to_numpy() > 0

    def sample(pool_mask, n):
        idx = np.flatnonzero(pool_mask & ~used)
        w = row_w[idx]
        chosen = rng.choice(idx, size=n, replace=False, p=w / w.sum())
        used[chosen] = True
        return chosen

    # ---- 1. upcoding (in place) ------------------------------------------------
    uc = fc["upcoding"]
    for ridx, tier in zip(sample(np.ones(n_clean, bool), n_up), tiers(n_up)):
        code = df.at[ridx, "procedure_package_code"]
        ref = df.at[ridx, "reference_package_rate"]
        sex = df.at[ridx, "patient_sex"]
        spec = pk.loc[code, "specialty"]
        mech = "amount_inflation"
        hm_ = mult_h[df.at[ridx, "hospital_id"]]   # inflation is measured against the hospital's legitimate price
        if tier == "subtle":
            r = rng.uniform(*uc["subtle_ratio"])
            df.at[ridx, "claimed_amount"] = int(round(ref * hm_ * r / 10) * 10)
        else:
            cands = [c for c in pk.index
                     if pk.loc[c, "specialty"] == spec and pk.loc[c, "pricing"] == "fixed"
                     and pk.loc[c, "reference_rate"] > ref and pk.loc[c, "sex"] in ("any", sex)]
            if cands and rng.random() < uc["swap_probability"]:
                new = rng.choice(cands)
                df.at[ridx, "procedure_package_code"] = new
                df.at[ridx, "reference_package_rate"] = int(pk.loc[new, "reference_rate"])
                df.at[ridx, "claimed_amount"] = int(round(pk.loc[new, "reference_rate"] * hm_
                                                          * rng.uniform(*uc["swap_amount_ratio"]) / 10) * 10)
                mech = "package_swap"
            else:
                r = rng.uniform(*uc["obvious_ratio"])
                df.at[ridx, "claimed_amount"] = int(round(ref * hm_ * r / 10) * 10)
        df.at[ridx, "is_fraud"] = 1
        df.at[ridx, "fraud_type"] = "upcoding"
        df.at[ridx, "severity_tier"] = tier
        df.at[ridx, "mechanism"] = mech

    new_rows = []

    # ---- 2. duplicate claims (new rows) ----------------------------------------
    dc = fc["duplicate"]
    for sidx, tier in zip(sample(~is_repeat, n_dup), tiers(n_dup)):
        src = df.loc[sidx]
        code = src["procedure_package_code"]
        spec = pk.loc[code, "specialty"]
        los = int(src["length_of_stay"])
        if tier == "obvious":
            shift = int(rng.integers(dc["obvious_shift_days"][0], dc["obvious_shift_days"][1] + 1))
            same = False
            amount = int(src["claimed_amount"])
            mech = "copy_other_hospital"
        else:
            shift = int(rng.integers(dc["subtle_shift_days"][0], dc["subtle_shift_days"][1] + 1))
            same = rng.random() < dc["subtle_same_hospital_prob"]
            amount = int(round(src["claimed_amount"] * rng.uniform(1 - dc["subtle_amount_noise"],
                                                                    1 + dc["subtle_amount_noise"]) / 10) * 10)
            mech = "resubmit_same_hospital" if same else "copy_other_hospital"
        # same rule as the clean data: admission must fall inside the window (a stay may run past the end)
        adm = src["admission_date"] + pd.Timedelta(days=shift)
        if adm > end:
            adm = src["admission_date"] - pd.Timedelta(days=shift)
        adm = max(adm, start)
        if same:
            h, d = src["hospital_id"], src["doctor_id"]
        else:
            h = pick_hospital(spec, exclude=src["hospital_id"])
            d = pick_doctor(h, spec)
        r = src.to_dict()
        r.update(hospital_id=h, doctor_id=d, admission_date=adm, discharge_date=adm + pd.Timedelta(days=los),
                 patient_age=age_at(dob[src["patient_id"]], adm), claimed_amount=amount,
                 is_fraud=1, fraud_type="duplicate", severity_tier=tier, mechanism=mech,
                 clean_claim_id="", linked_clean_id=src["clean_claim_id"])
        new_rows.append(r)

    # ---- 3. phantom billing (new rows) -----------------------------------------
    pc = fc["phantom"]
    n_ph_tiers = tiers(n_ph)
    n_sub = int((n_ph_tiers == "subtle").sum())
    nonrep = pk[pk["repeat_mean"] == 0]
    # subtle: plausible package, overlapping stay at another hospital
    for sidx in sample(~is_repeat, n_sub):
        src = df.loc[sidx]
        days_left = (end - src["admission_date"]).days + 1
        adm = src["admission_date"] + pd.Timedelta(days=int(rng.integers(0, max(1, min(int(src["length_of_stay"]), days_left)))))
        pid = src["patient_id"]
        age = age_at(dob[pid], adm)
        sex = src["patient_sex"]
        w = np.array([nonrep.loc[c, "weight"]
                      * np.exp(-0.5 * ((age - nonrep.loc[c, "age_mean"]) / nonrep.loc[c, "age_sd"]) ** 2)
                      / nonrep.loc[c, "age_sd"]
                      * (1.0 if nonrep.loc[c, "sex"] in ("any", sex) else 0.0) for c in nonrep.index]) + 1e-12
        code = rng.choice(nonrep.index.to_numpy(), p=w / w.sum())
        spec = pk.loc[code, "specialty"]
        h = pick_hospital(spec, exclude=src["hospital_id"])
        los = draw_los(code)
        ref_new = draw_ref(code, los)
        r = src.to_dict()
        r.update(hospital_id=h, doctor_id=pick_doctor(h, spec), admission_date=adm,
                 discharge_date=adm + pd.Timedelta(days=los), patient_age=age,
                 diagnosis_code=pk.loc[code, "diagnosis_code"], procedure_package_code=code,
                 claimed_amount=draw_amount(ref_new, h), reference_package_rate=ref_new,
                 length_of_stay=los, is_fraud=1, fraud_type="phantom", severity_tier="subtle",
                 mechanism="overlapping_stay", clean_claim_id="", linked_clean_id=src["clean_claim_id"])
        new_rows.append(r)
    # obvious: package impossible for the patient
    for _ in range(n_ph - n_sub):
        for _try in range(50):
            pid = pat["patient_id"].iloc[int(rng.integers(0, len(pat)))]
            adm = start + pd.Timedelta(days=int(rng.integers(0, (end - start).days - 10)))
            age = age_at(dob[pid], adm)
            sex = pat_info.at[pid, "patient_sex"]
            use_sex = rng.random() < pc["obvious_sex_violation_prob"]
            sex_bad = [c for c in nonrep.index if nonrep.loc[c, "sex"] not in ("any", sex)]
            age_bad = [c for c in nonrep.index
                       if abs(age - nonrep.loc[c, "age_mean"]) / nonrep.loc[c, "age_sd"] > pc["age_z_threshold"]
                       and nonrep.loc[c, "sex"] in ("any", sex)]
            if use_sex and sex_bad:
                code, mech = rng.choice(sex_bad), "sex_mismatch"
            elif age_bad:
                code, mech = rng.choice(age_bad), "age_mismatch"
            elif sex_bad:
                code, mech = rng.choice(sex_bad), "sex_mismatch"
            else:
                continue
            break
        spec = pk.loc[code, "specialty"]
        h = pick_hospital(spec)
        los = draw_los(code)
        ref_new = draw_ref(code, los)
        new_rows.append(dict(
            claim_id="", patient_id=pid, hospital_id=h, doctor_id=pick_doctor(h, spec),
            admission_date=adm, discharge_date=adm + pd.Timedelta(days=los), patient_age=age,
            patient_sex=sex, patient_district=pat_info.at[pid, "patient_district"],
            diagnosis_code=pk.loc[code, "diagnosis_code"], procedure_package_code=code,
            claimed_amount=draw_amount(ref_new, h), reference_package_rate=ref_new,
            length_of_stay=los, is_fraud=1, fraud_type="phantom", severity_tier="obvious",
            mechanism=mech, clean_claim_id="", linked_clean_id=""))

    # ---- assemble, shuffle ties, re-number ids (so ids and order cannot leak labels) ------
    allr = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    allr["_tie"] = rng.random(len(allr))
    allr = allr.sort_values(["admission_date", "_tie"], kind="stable").reset_index(drop=True)
    allr["claim_id"] = [f"C{i + 1:06d}" for i in range(len(allr))]
    to_final = dict(zip(allr.loc[allr["clean_claim_id"] != "", "clean_claim_id"],
                        allr.loc[allr["clean_claim_id"] != "", "claim_id"]))
    allr["linked_claim_id"] = allr["linked_clean_id"].map(to_final).fillna("")
    allr["admission_date"] = allr["admission_date"].dt.strftime("%Y-%m-%d")
    allr["discharge_date"] = allr["discharge_date"].dt.strftime("%Y-%m-%d")
    for c in ["patient_age", "claimed_amount", "reference_package_rate", "length_of_stay", "is_fraud"]:
        allr[c] = allr[c].astype(int)
    labels = allr[["claim_id", "is_fraud", "fraud_type", "severity_tier", "mechanism",
                   "clean_claim_id", "linked_claim_id"]]
    return allr[COLS], labels, prone


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/params.json")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default=None)
    ap.add_argument("--rates", type=float, nargs="+", default=[0.02, 0.05, 0.08])
    args = ap.parse_args()
    out = args.out or args.data
    cfg = json.load(open(args.config, encoding="utf-8"))
    clean = pd.read_csv(os.path.join(args.data, "claims_clean.csv"))
    patients = pd.read_csv(os.path.join(args.data, "patients.csv"))
    hospitals = pd.read_csv(os.path.join(args.data, "hospitals.csv"))
    doctors = pd.read_csv(os.path.join(args.data, "doctors.csv"))
    os.makedirs(out, exist_ok=True)
    for rate in args.rates:
        claims, labels, prone = inject(clean, patients, hospitals, doctors, cfg, rate)
        tag = f"{int(round(rate * 100))}pct"
        claims.to_csv(os.path.join(out, f"claims_fraud_{tag}.csv"), index=False)
        labels.to_csv(os.path.join(out, f"fraud_labels_{tag}.csv"), index=False)
        f = claims[claims["is_fraud"] == 1]
        print(f"rate {rate:.0%}: {len(claims):,} claims, {len(f):,} fraud ({len(f) / len(claims):.2%})")
        print(pd.crosstab(f["fraud_type"], f["severity_tier"]).to_string())
        print()


if __name__ == "__main__":
    main()
