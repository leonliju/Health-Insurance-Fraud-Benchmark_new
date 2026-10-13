#!/usr/bin/env python3
"""
Generate a CLEAN (fraud-free) synthetic health-insurance claims dataset.

Rule-based simulator (no machine learning). Prices come from the AB PM-JAY Health Benefit
Package master HBP 2.2 (National Health Authority); everything else that is not sourced is an
ASSUMPTION and is listed in data/parameter_sources.csv.

Realism rules built in (sources in the README):
  * medical packages are priced per bed-day (routine ward, HDU, ICU) x length of stay
  * hospital incentives (NABH, metro, aspirational district, PG/DNB) compound on the base rate
  * caesarean and high risk delivery happen mostly at public hospitals (private by referral)
  * one-time surgeries (appendicectomy, cholecystectomy, TURP ...) are not repeated for a patient
  * dialysis and chemotherapy genuinely repeat (patterns checked against a Synthea run)

Usage:
    python src/generate_claims.py --config config/params.json --out data_full
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

CLAIM_COLUMNS = [
    "claim_id", "patient_id", "hospital_id", "doctor_id",
    "admission_date", "discharge_date",
    "patient_age", "patient_sex", "patient_district",
    "diagnosis_code", "procedure_package_code",
    "claimed_amount", "reference_package_rate", "length_of_stay",
    "is_fraud", "fraud_type", "severity_tier",
]


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def bed_tables(cfg):
    names = [k for k in cfg["bed_categories"] if not k.startswith("_")]
    shares = np.array([cfg["bed_categories"][k]["share"] for k in names], dtype=float)
    rates = np.array([cfg["bed_categories"][k]["day_rate"] for k in names], dtype=float)
    return names, shares / shares.sum(), rates


# ----------------------------------------------------------------------
# Reference entities
# ----------------------------------------------------------------------
def build_hospitals(cfg, rng, specialties):
    n = cfg["n_hospitals"]
    n_dist = cfg["n_districts"]
    district_weights = rng.lognormal(0, 0.5, n_dist)
    district_weights /= district_weights.sum()
    district = rng.choice(n_dist, size=n, p=district_weights)
    h_type = np.where(rng.random(n) < cfg["share_public_hospitals"], "public", "private")
    size = rng.lognormal(0, 0.8, n)
    size_norm = size / size.mean()

    offered = []
    for i in range(n):
        k = int(np.clip(1 + rng.poisson(1 + 1.5 * size_norm[i]), 1, len(specialties)))
        others = [s for s in specialties if s != "General Medicine"]
        chosen = list(rng.choice(others, size=min(k - 1, len(others)), replace=False)) if k > 1 else []
        offered.append(["General Medicine"] + chosen)
    biggest = int(np.argmax(size))
    for s in specialties:
        if not any(s in o for o in offered):
            offered[biggest].append(s)

    # hospital attributes that drive the legitimate price uplift (NHA incentives, compounded)
    ha = cfg["hospital_attributes"]
    inc = ha["incentive"]
    p_full, p_entry = ha["nabh_full_share"], ha["nabh_entry_share"]
    nabh = rng.choice(["none", "entry", "full"], size=n, p=[1 - p_full - p_entry, p_entry, p_full])
    metro = rng.random(n) < ha["metro_share"]
    aspirational = (~metro) & (rng.random(n) < ha["aspirational_share"] / (1 - ha["metro_share"]))
    pg_dnb = rng.random(n) < ha["pg_dnb_share"]
    mult = (np.where(nabh == "full", 1 + inc["nabh_full"], np.where(nabh == "entry", 1 + inc["nabh_entry"], 1.0))
            * np.where(metro, 1 + inc["metro"], 1.0)
            * np.where(aspirational, 1 + inc["aspirational"], 1.0)
            * np.where(pg_dnb, 1 + inc["pg_dnb"], 1.0))

    hospitals = pd.DataFrame({
        "hospital_id": [f"H{i + 1:03d}" for i in range(n)],
        "district_idx": district,
        "district": [f"District_{d + 1:03d}" for d in district],
        "hospital_type": h_type,
        "nabh_level": nabh,
        "is_metro": metro.astype(int),
        "is_aspirational": aspirational.astype(int),
        "has_pg_dnb": pg_dnb.astype(int),
        "incentive_multiplier": np.round(mult, 4),
        "size_weight": np.round(size, 4),
        "specialties": ["|".join(o) for o in offered],
    })
    return hospitals, district_weights


def build_doctors(cfg, rng, hospitals):
    rows = []
    d = 0
    size = hospitals["size_weight"].to_numpy()
    size_norm = size / size.mean()
    for i, h in hospitals.iterrows():
        for spec in h["specialties"].split("|"):
            n_doc = 1 + rng.poisson(cfg["doctor_extra_lambda"] * size_norm[i])
            for _ in range(int(n_doc)):
                d += 1
                rows.append((f"D{d:04d}", h["hospital_id"], i, spec, rng.lognormal(0, 0.5)))
    return pd.DataFrame(rows, columns=["doctor_id", "hospital_id", "hospital_idx", "specialty", "workload_weight"])


def build_patients(cfg, rng, district_weights):
    n = cfg["n_patients"]
    buckets = cfg["age_buckets"]
    shares = np.array([b["share"] for b in buckets], dtype=float)
    shares /= shares.sum()
    bucket_idx = rng.choice(len(buckets), size=n, p=shares)
    lo = np.array([b["min"] for b in buckets])[bucket_idx]
    hi = np.array([b["max"] for b in buckets])[bucket_idx]
    age_at_start = rng.integers(lo, hi + 1)
    start = pd.Timestamp(cfg["window_start"])
    dob = start - pd.to_timedelta((age_at_start * 365.25 + rng.integers(0, 365, n)).astype(int), unit="D")
    sex = np.where(rng.random(n) < cfg["female_share"], "F", "M")
    district_idx = rng.choice(cfg["n_districts"], size=n, p=district_weights)
    return pd.DataFrame({
        "patient_id": [f"P{i + 1:05d}" for i in range(n)],
        "dob": dob,
        "patient_sex": sex,
        "district_idx": district_idx,
        "patient_district": [f"District_{d + 1:03d}" for d in district_idx],
    })


# ----------------------------------------------------------------------
# Claim generation
# ----------------------------------------------------------------------
def package_probabilities(cfg, patients, packages):
    """Prevalence weight x age fit x sex rule, for every patient and package."""
    start = pd.Timestamp(cfg["window_start"])
    end = pd.Timestamp(cfg["window_end"])
    mid = start + (end - start) / 2
    age_mid = ((mid - patients["dob"]).dt.days / 365.25).to_numpy()
    sex = patients["patient_sex"].to_numpy()
    probs = np.zeros((len(patients), len(packages)))
    for j, p in packages.iterrows():
        fit = np.exp(-0.5 * ((age_mid - p["age_mean"]) / p["age_sd"]) ** 2) / p["age_sd"]
        mask = np.ones(len(patients)) if p["sex"] == "any" else (sex == p["sex"]).astype(float)
        probs[:, j] = p["weight"] * fit * mask
    probs += 1e-12
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def enforce_one_time_rules(rng, claims, probs, packages):
    """No repeated one-time surgeries; minimum gaps between repeats (NHA configures these in its IT system)."""
    maxp = packages["max_per_patient"].to_numpy()
    ming = packages["min_gap_days"].to_numpy()
    claims = claims.sort_values(["patient_idx", "admission_date"], kind="stable").reset_index(drop=True)
    pkg = claims["pkg_idx"].to_numpy().copy()
    dates = claims["admission_date"].to_numpy()
    positions = claims.groupby("patient_idx").indices
    sizes = claims.groupby("patient_idx").size()
    changed = 0
    for pid in sizes[sizes >= 2].index:
        count, last = {}, {}
        for k in positions[pid]:
            bad = set()
            while True:
                j = int(pkg[k])
                gap_bad = ming[j] > 0 and j in last and int((dates[k] - last[j]).astype("timedelta64[D]").astype(int)) < ming[j]
                if not ((maxp[j] > 0 and count.get(j, 0) >= maxp[j]) or gap_bad):
                    break
                bad.add(j)
                p = probs[pid].copy()
                p[list(bad)] = 0
                if p.sum() <= 1e-9:
                    break
                pkg[k] = rng.choice(len(p), p=p / p.sum())
                changed += 1
            j = int(pkg[k])
            count[j] = count.get(j, 0) + 1
            last[j] = dates[k]
    claims["pkg_idx"] = pkg
    return claims, changed


def generate(cfg, seed=None):
    seed = cfg["seed"] if seed is None else seed
    rng = np.random.default_rng(seed)
    packages = pd.DataFrame(cfg["packages"]).reset_index(drop=True)
    specialties = list(dict.fromkeys(packages["specialty"].tolist()))
    start = pd.Timestamp(cfg["window_start"])
    end = pd.Timestamp(cfg["window_end"])
    window_days = (end - start).days

    hospitals, district_weights = build_hospitals(cfg, rng, specialties)
    doctors = build_doctors(cfg, rng, hospitals)
    patients = build_patients(cfg, rng, district_weights)

    # --- admissions per patient and package choice ---
    n_claims = 1 + rng.poisson(cfg["extra_claims_lambda"], len(patients))
    pat_idx = np.repeat(np.arange(len(patients)), n_claims)
    probs = package_probabilities(cfg, patients, packages)
    cum = probs[pat_idx].cumsum(axis=1)
    u = rng.random((len(pat_idx), 1))
    pkg_idx = np.minimum((cum < u).sum(axis=1), len(packages) - 1)
    n = len(pat_idx)
    claims = pd.DataFrame({
        "patient_idx": pat_idx,
        "pkg_idx": pkg_idx,
        "admission_date": start + pd.to_timedelta(rng.integers(0, window_days, n), unit="D"),
    })
    claims, n_redrawn = enforce_one_time_rules(rng, claims, probs, packages)
    claims["specialty"] = packages["specialty"].to_numpy()[claims["pkg_idx"]]
    claims["reserved"] = packages["public_reserved"].to_numpy()[claims["pkg_idx"]]
    claims["district_idx"] = patients["district_idx"].to_numpy()[claims["patient_idx"]]

    # --- hospital choice: local first; public-reserved packages mostly at public hospitals ---
    is_public = (hospitals["hospital_type"] == "public").to_numpy()
    hosp_size = hospitals["size_weight"].to_numpy()
    hosp_district = hospitals["district_idx"].to_numpy()
    hosp_offers = {s: hospitals.index[hospitals["specialties"].str.contains(s, regex=False)].to_numpy() for s in specialties}

    def sample_from(pool, m):
        w = hosp_size[pool]
        return rng.choice(pool, size=m, p=w / w.sum())

    claims["hospital_idx"] = -1
    for (dist, spec, res), grp in claims.groupby(["district_idx", "specialty", "reserved"]):
        idx = grp.index.to_numpy()
        m = len(idx)
        glob = hosp_offers[spec]
        pub_only = (rng.random(m) >= cfg["private_referral_share"]) if res else np.zeros(m, dtype=bool)
        chosen = np.empty(m, dtype=int)
        for flag in (True, False):
            sel = np.flatnonzero(pub_only == flag)
            if len(sel) == 0:
                continue
            pool = glob[is_public[glob]] if flag else glob
            if len(pool) == 0:
                pool = glob
            loc = pool[hosp_district[pool] == dist]
            use_loc = (rng.random(len(sel)) < cfg["local_hospital_share"]) & (len(loc) > 0)
            out = np.empty(len(sel), dtype=int)
            if use_loc.any():
                out[use_loc] = sample_from(loc, int(use_loc.sum()))
            if (~use_loc).any():
                out[~use_loc] = sample_from(pool, int((~use_loc).sum()))
            chosen[sel] = out
        claims.loc[idx, "hospital_idx"] = chosen

    # --- doctor choice: within hospital and specialty ---
    claims["doctor_pos"] = -1
    doc_lookup = {k: (g.index.to_numpy(), g["workload_weight"].to_numpy())
                  for k, g in doctors.groupby(["hospital_idx", "specialty"])}
    for (h, spec), grp in claims.groupby(["hospital_idx", "specialty"]):
        ids, w = doc_lookup[(h, spec)]
        claims.loc[grp.index, "doctor_pos"] = rng.choice(ids, size=len(grp), p=w / w.sum())

    # --- length of stay ---
    los_mean = packages["los_mean"].to_numpy()[claims["pkg_idx"]]
    los_sd = packages["los_sd"].to_numpy()[claims["pkg_idx"]]
    los = np.rint(rng.normal(los_mean, los_sd)).astype(int)
    claims["length_of_stay"] = np.clip(los, 1, np.ceil(los_mean + 3 * los_sd).astype(int))

    # --- repeat claims (dialysis sessions, chemotherapy cycles) ---
    base = claims.copy()
    extra_frames = []
    rep_mean = packages["repeat_mean"].to_numpy()[base["pkg_idx"]]
    rep_max = packages["repeat_max"].to_numpy()[base["pkg_idx"]]
    k = np.minimum(rng.poisson(rep_mean), rep_max)
    prev = base.loc[k > 0].copy()
    prev["_k"] = k[k > 0]
    j = 0
    while len(prev) > 0:
        j += 1
        prev = prev[prev["_k"] >= j].copy()
        if prev.empty:
            break
        gmin = packages["gap_min"].to_numpy()[prev["pkg_idx"]]
        gmax = packages["gap_max"].to_numpy()[prev["pkg_idx"]]
        gap = rng.integers(gmin, gmax + 1)
        nxt = prev.copy()
        nxt["admission_date"] = prev["admission_date"] + pd.to_timedelta(prev["length_of_stay"] + gap, unit="D")
        lm = packages["los_mean"].to_numpy()[nxt["pkg_idx"]]
        ls = packages["los_sd"].to_numpy()[nxt["pkg_idx"]]
        nxt["length_of_stay"] = np.clip(np.rint(rng.normal(lm, ls)).astype(int), 1, np.ceil(lm + 3 * ls).astype(int))
        nxt = nxt[nxt["admission_date"] <= end]
        extra_frames.append(nxt)
        prev = nxt
    if extra_frames:
        claims = pd.concat([claims] + [f.drop(columns="_k") for f in extra_frames], ignore_index=True)

    # --- discharge, reference rate (fixed or per bed-day), amount with hospital uplift, age ---
    claims["discharge_date"] = claims["admission_date"] + pd.to_timedelta(claims["length_of_stay"], unit="D")
    _, bed_shares, bed_rates = bed_tables(cfg)
    bed = rng.choice(len(bed_shares), size=len(claims), p=bed_shares)
    per_day = (packages["pricing"].to_numpy()[claims["pkg_idx"]] == "per_day")
    fixed_ref = packages["reference_rate"].to_numpy()[claims["pkg_idx"]]
    ref = np.where(per_day, claims["length_of_stay"].to_numpy() * bed_rates[bed], fixed_ref).astype(int)
    mult = hospitals["incentive_multiplier"].to_numpy()[claims["hospital_idx"]]
    ratio = np.clip(rng.lognormal(0, cfg["amount_ratio_sigma"], len(claims)),
                    cfg["amount_ratio_clip"][0], cfg["amount_ratio_clip"][1])
    claims["reference_package_rate"] = ref
    claims["claimed_amount"] = (np.rint(ref * mult * ratio / 10) * 10).astype(int)
    dob = patients["dob"].to_numpy()[claims["patient_idx"]]
    claims["patient_age"] = ((claims["admission_date"].to_numpy() - dob).astype("timedelta64[D]").astype(int) / 365.25).astype(int)

    # --- final ids, labels, column order ---
    claims = claims.sort_values(["admission_date", "patient_idx"], kind="stable").reset_index(drop=True)
    claims["claim_id"] = [f"C{i + 1:06d}" for i in range(len(claims))]
    claims["patient_id"] = patients["patient_id"].to_numpy()[claims["patient_idx"]]
    claims["patient_sex"] = patients["patient_sex"].to_numpy()[claims["patient_idx"]]
    claims["patient_district"] = patients["patient_district"].to_numpy()[claims["patient_idx"]]
    claims["hospital_id"] = hospitals["hospital_id"].to_numpy()[claims["hospital_idx"]]
    claims["doctor_id"] = doctors["doctor_id"].to_numpy()[claims["doctor_pos"]]
    claims["diagnosis_code"] = packages["diagnosis_code"].to_numpy()[claims["pkg_idx"]]
    claims["procedure_package_code"] = packages["package_code"].to_numpy()[claims["pkg_idx"]]
    claims["is_fraud"] = 0
    claims["fraud_type"] = "none"
    claims["severity_tier"] = "none"
    claims["admission_date"] = claims["admission_date"].dt.strftime("%Y-%m-%d")
    claims["discharge_date"] = claims["discharge_date"].dt.strftime("%Y-%m-%d")
    claims = claims[CLAIM_COLUMNS]

    patients_out = patients[["patient_id", "patient_sex", "patient_district"]].copy()
    patients_out["date_of_birth"] = patients["dob"].dt.strftime("%Y-%m-%d")
    hospitals_out = hospitals[["hospital_id", "district", "hospital_type", "nabh_level", "is_metro", "is_aspirational",
                               "has_pg_dnb", "incentive_multiplier", "size_weight", "specialties"]]
    doctors_out = doctors[["doctor_id", "hospital_id", "specialty", "workload_weight"]]
    packages_out = packages[["package_code", "package_name", "hbp_code", "diagnosis_code", "specialty", "pricing",
                             "reference_rate", "sex", "public_reserved", "max_per_patient", "min_gap_days",
                             "los_mean", "repeat_mean"]].rename(columns={"reference_rate": "reference_package_rate"})
    return claims, patients_out, hospitals_out, doctors_out, packages_out, n_redrawn


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/params.json")
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-patients", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.n_patients:
        cfg["n_patients"] = args.n_patients
    claims, patients, hospitals, doctors, packages, n_redrawn = generate(cfg, args.seed)

    os.makedirs(args.out, exist_ok=True)
    claims.to_csv(os.path.join(args.out, "claims_clean.csv"), index=False)
    patients.to_csv(os.path.join(args.out, "patients.csv"), index=False)
    hospitals.to_csv(os.path.join(args.out, "hospitals.csv"), index=False)
    doctors.to_csv(os.path.join(args.out, "doctors.csv"), index=False)
    packages.to_csv(os.path.join(args.out, "packages.csv"), index=False)
    print(f"Wrote {len(claims):,} claims, {len(patients):,} patients, {len(hospitals):,} hospitals, "
          f"{len(doctors):,} doctors to {args.out}/ (one-time-surgery rules re-drew {n_redrawn:,} treatments)")


if __name__ == "__main__":
    main()
