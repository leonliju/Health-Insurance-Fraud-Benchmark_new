#!/usr/bin/env python3
"""
Module 3, step 1: black-box, rule-based adaptive fraudster.

The fraudster queries a saved detector's fraud scores and tests constrained
changes to fraudulent claims in the selected split (TEST by default). It never
changes genuine claims, fraud labels, or the original input files.

Supported adaptations:
  * upcoding: reduce the inflated amount while keeping the claim upcoded;
  * duplicate: move the resubmission date within the configured duplicate window;
  * phantom: try another hospital that supports the same specialty, while
    preserving the impossible package or overlapping stay.

Usage:
  python src/adaptive_fraudster.py --dir data_v3 --tag 5pct \
      --detector xgboost --variant with_district

Inputs expected in --dir:
  claims_fraud_<tag>.csv
  fraud_labels_<tag>.csv
  features_<tag>.csv
  labels_<tag>.csv
  feature_list_<tag>.json
  hospitals.csv
  doctors.csv
  models/<detector>_<tag>_<variant>.joblib

The script writes a separate adaptive/ directory. Original files are untouched.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def make_hospital_features(hospitals):
    """Reproduce the hospital-feature encoding used by build_features.py."""
    h = hospitals.copy()
    multiplier_col = next(
        (c for c in h.columns if "multiplier" in c.lower()), None
    )
    if multiplier_col is None:
        raise ValueError("No hospital multiplier column found in hospitals.csv")

    district_col = next(
        (
            c for c in h.columns
            if "district" in c.lower() and "aspir" not in c.lower()
        ),
        None,
    )

    h = h.rename(columns={multiplier_col: "hosp_multiplier"})
    feature_cols = ["hosp_multiplier"]

    for col in h.columns:
        if col in (
            "hospital_id", "hosp_multiplier", district_col,
            "specialties", "size_weight"
        ) or "name" in col.lower():
            continue

        if h[col].dtype == object or str(h[col].dtype) == "str":
            dummies = pd.get_dummies(h[col], prefix="h_" + col).astype(int)
            h = pd.concat([h, dummies], axis=1)
            feature_cols.extend(dummies.columns.tolist())
        else:
            h[col] = pd.to_numeric(h[col], errors="coerce").astype(float)
            feature_cols.append(col)

    h = h.drop_duplicates("hospital_id").set_index("hospital_id")
    return h, feature_cols, district_col


def temporal_features_for_claim(candidate, claims, feature_row):
    """
    Recalculate timing and overlap features for one candidate claim.
    Other claims belonging to the same patient are treated as unchanged.
    """
    pid = candidate["patient_id"]
    cid = candidate["claim_id"]

    others = claims[
        (claims["patient_id"] == pid) & (claims["claim_id"] != cid)
    ].copy()

    current = pd.DataFrame([candidate])
    group = pd.concat([others, current], ignore_index=True)
    group["admission_date"] = pd.to_datetime(group["admission_date"])
    group["discharge_date"] = pd.to_datetime(group["discharge_date"])
    group = group.sort_values(
        ["procedure_package_code", "admission_date", "claim_id"]
    )

    same_package = group[
        group["procedure_package_code"] == candidate["procedure_package_code"]
    ].copy()
    same_package = same_package.sort_values(["admission_date", "claim_id"])
    positions = np.flatnonzero(same_package["claim_id"].to_numpy() == cid)

    if len(positions) != 1:
        raise ValueError(f"Could not uniquely locate claim {cid} in patient group")

    pos = int(positions[0])
    dates = same_package["admission_date"].tolist()
    feature_row["gap_prev"] = (
        (dates[pos] - dates[pos - 1]).days if pos > 0 else 999
    )
    feature_row["gap_next"] = (
        (dates[pos + 1] - dates[pos]).days
        if pos < len(dates) - 1 else 999
    )
    feature_row["same_pkg_count"] = len(same_package)
    feature_row["claims_per_patient"] = len(group)

    start = pd.Timestamp(candidate["admission_date"])
    end = pd.Timestamp(candidate["discharge_date"])
    overlap_count = 0
    overlap_different_hospital = 0

    for _, other in others.iterrows():
        other_start = pd.Timestamp(other["admission_date"])
        other_end = pd.Timestamp(other["discharge_date"])
        overlaps = start <= other_end and end >= other_start
        if overlaps:
            overlap_count += 1
            if other["hospital_id"] != candidate["hospital_id"]:
                overlap_different_hospital += 1

    feature_row["n_overlap"] = overlap_count
    feature_row["n_overlap_diff_hosp"] = overlap_different_hospital
    feature_row["overlap_any"] = int(overlap_count > 0)
    feature_row["overlap_diff_hosp"] = int(overlap_different_hospital > 0)
    return feature_row


def set_hospital_features(
    feature_row,
    candidate,
    old_hospital_id,
    claims,
    hospital_features,
    hospital_feature_cols,
    district_col,
    optional_cols,
    train_hospital_mean,
    overall_train_mean,
):
    """Update hospital-derived features after a candidate hospital change."""
    new_hospital_id = candidate["hospital_id"]
    if new_hospital_id not in hospital_features.index:
        raise ValueError(f"Unknown hospital_id: {new_hospital_id}")

    for col in hospital_feature_cols:
        if col in feature_row.index:
            feature_row[col] = hospital_features.at[new_hospital_id, col]

    # hosp_volume in build_features.py is the count across the full dataset.
    counts = claims["hospital_id"].value_counts()
    feature_row["hosp_volume"] = int(counts.get(new_hospital_id, 0))
    if new_hospital_id != old_hospital_id:
        feature_row["hosp_volume"] += 1

    # The hospital mean adjusted ratio is based on TRAIN claims only.
    feature_row["hosp_mean_adj_ratio"] = float(
        train_hospital_mean.get(new_hospital_id, overall_train_mean)
    )

    if "in_district" in optional_cols and district_col:
        patient_district = candidate.get("patient_district")
        hospital_district = (
            hospital_features.at[new_hospital_id, district_col]
            if district_col in hospital_features.columns else None
        )
        feature_row["in_district"] = int(
            pd.notna(patient_district)
            and pd.notna(hospital_district)
            and patient_district == hospital_district
        )

    return feature_row


def choose_doctor(doctors, hospital_id, specialty, fallback):
    """Choose a doctor belonging to the selected hospital and specialty."""
    if doctors is None or doctors.empty:
        return fallback
    matches = doctors[
        (doctors["hospital_id"] == hospital_id)
        & (doctors["specialty"] == specialty)
    ]
    if matches.empty:
        return fallback
    return str(matches.iloc[0]["doctor_id"])



def detector_scores(model, matrix, detector_name):
    """Return scores where larger values always mean more suspicious."""
    if detector_name == "isolation_forest":
        return -model.score_samples(matrix)
    return model.predict_proba(matrix)[:, 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="data_v3")
    parser.add_argument("--tag", default="5pct")
    parser.add_argument(
        "--detector",
        choices=["xgboost", "random_forest", "isolation_forest"],
        default="xgboost",
    )
    parser.add_argument(
        "--variant",
        choices=["with_district", "no_district"],
        default="with_district",
    )
    parser.add_argument("--params", default="config/params.json")
    parser.add_argument("--max-hospital-candidates", type=int, default=8)

    # Select which dataset split the fraudster should target.
    parser.add_argument(
        "--target-split",
        choices=["test", "train"],
        default="test",
        help="Split containing fraudulent claims to adapt",
    )

    args = parser.parse_args()
    data_dir = Path(args.dir)

    # Keep TEST and TRAIN adaptation experiments separate.
    if args.target_split == "test":
        output_dir = data_dir / "adaptive" / args.tag
    else:
        output_dir = (
            data_dir / "adaptive" / f"{args.tag}_train_adaptation"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Load the original data and saved detector ----------
    claims_path = data_dir / f"claims_fraud_{args.tag}.csv"
    detail_path = data_dir / f"fraud_labels_{args.tag}.csv"
    features_path = data_dir / f"features_{args.tag}.csv"
    labels_path = data_dir / f"labels_{args.tag}.csv"
    feature_list_path = data_dir / f"feature_list_{args.tag}.json"
    model_path = (
        data_dir / "models"
        / f"{args.detector}_{args.tag}_{args.variant}.joblib"
    )

    required = [
        claims_path, detail_path, features_path, labels_path,
        feature_list_path, data_dir / "hospitals.csv", model_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required input file(s) not found:\n" + "\n".join(missing)
        )

    params = json.loads(Path(args.params).read_text(encoding="utf-8"))
    claims = pd.read_csv(
        claims_path, parse_dates=["admission_date", "discharge_date"]
    )
    fraud_details = pd.read_csv(detail_path, dtype={"claim_id": str})
    X = pd.read_csv(features_path)
    Y = pd.read_csv(labels_path)
    feature_list = json.loads(feature_list_path.read_text(encoding="utf-8"))
    hospitals = pd.read_csv(data_dir / "hospitals.csv")

    doctors_path = data_dir / "doctors.csv"
    doctors = pd.read_csv(doctors_path) if doctors_path.exists() else None

    if not X["claim_id"].equals(Y["claim_id"]):
        raise ValueError("features and labels claim_id order does not match")
    if claims["claim_id"].duplicated().any():
        raise ValueError("claims_fraud contains duplicate claim_id values")
    if fraud_details["claim_id"].duplicated().any():
        raise ValueError("fraud_labels contains duplicate claim_id values")

    base_features = feature_list["features"]
    optional_features = feature_list.get("optional", [])
    model_features = (
        base_features + optional_features
        if args.variant == "with_district"
        else base_features
    )
    missing_features = sorted(set(model_features) - set(X.columns))
    if missing_features:
        raise ValueError(f"Feature columns missing from features CSV: {missing_features}")

    model = joblib.load(model_path)
    hospital_features, hospital_feature_cols, district_col = (
        make_hospital_features(hospitals)
    )
    hospital_feature_cols = [
        c for c in hospital_feature_cols if c in model_features
    ]

    # Attach the patient-level split to the raw claims for internal calculations.
    split_map = X.set_index("claim_id")["split"]
    claims = claims.copy()
    claims["_split"] = claims["claim_id"].map(split_map)
    if claims["_split"].isna().any():
        raise ValueError("Some raw claims are missing from the feature split file")

    # Train-only hospital means, matching build_features.py.
    multiplier_col = next(
        c for c in hospitals.columns if "multiplier" in c.lower()
    )
    multiplier_map = hospitals.set_index("hospital_id")[multiplier_col]
    train_claims = claims[claims["_split"] == "train"].copy()
    train_claims["hosp_multiplier"] = train_claims["hospital_id"].map(
        multiplier_map
    )
    train_claims["adj_ratio"] = (
        train_claims["claimed_amount"]
        / train_claims["reference_package_rate"]
        / train_claims["hosp_multiplier"]
    )
    train_hospital_mean = train_claims.groupby("hospital_id")[
        "adj_ratio"
    ].mean()
    overall_train_mean = float(train_claims["adj_ratio"].mean())

    # Feature rows indexed by claim_id; retain only model input columns.
    feature_index = X.set_index("claim_id")
    model_X = feature_index[model_features].copy()
    model_X = model_X.apply(pd.to_numeric, errors="raise")

    label_index = Y.set_index("claim_id")
    fraud_detail_index = fraud_details.set_index("claim_id")
    claim_index = claims.set_index("claim_id", drop=False)

    # Adapt only fraudulent claims from the requested split.
    target_ids = Y.loc[
        (Y["split"] == args.target_split) & (Y["is_fraud"] == 1),
        "claim_id",
    ].astype(str).tolist()

    if not target_ids:
        raise ValueError(
            f"No fraudulent {args.target_split.upper()} claims found"
        )
    baseline_scores = detector_scores(model, model_X.loc[target_ids], args.detector)
    baseline_score_map = dict(zip(target_ids, baseline_scores))

    fraud_cfg = params["fraud"]
    upcoding_cfg = fraud_cfg["upcoding"]
    duplicate_cfg = fraud_cfg["duplicate"]
    window_start = pd.Timestamp(params["window_start"])
    window_end = pd.Timestamp(params["window_end"])

    rng = np.random.default_rng(int(params.get("seed", 42)) + 901)
    candidates = []
    candidate_counts = {}

    # ---------- Generate valid candidate modifications ----------
    for claim_id in target_ids:
        claim = claim_index.loc[claim_id].to_dict()
        detail = fraud_detail_index.loc[claim_id]
        fraud_type = str(label_index.at[claim_id, "fraud_type"])
        tier = str(label_index.at[claim_id, "severity_tier"])
        mechanism = str(detail.get("mechanism", ""))
        original_score = float(baseline_score_map[claim_id])
        base_row = model_X.loc[claim_id].copy()
        generated = 0

        def add_candidate(candidate_claim, candidate_features, description):
            nonlocal generated
            candidate_features = candidate_features[model_features].copy()
            candidates.append({
                "claim_id": claim_id,
                "claim": candidate_claim,
                "features": candidate_features,
                "description": description,
                "score_before": original_score,
            })
            generated += 1

        # ----- Upcoding: reduce the amount but retain an inflated bill/package swap.
        if fraud_type == "upcoding":
            old_hospital = claim["hospital_id"]
            hospital_multiplier = float(multiplier_map.loc[old_hospital])
            reference_rate = float(claim["reference_package_rate"])
            current_ratio = (
                float(claim["claimed_amount"]) / reference_rate / hospital_multiplier
            )

            if mechanism == "package_swap":
                minimum_ratio = float(upcoding_cfg["swap_amount_ratio"][0])
            else:
                minimum_ratio = max(
                    1.01, float(upcoding_cfg["subtle_ratio"][0])
                )

            proposed_ratios = sorted({
                round(minimum_ratio, 4),
                round(max(minimum_ratio, current_ratio * 0.95), 4),
                round(max(minimum_ratio, current_ratio * 0.85), 4),
            })

            for ratio in proposed_ratios:
                if ratio >= current_ratio - 1e-5:
                    continue
                new_claim = claim.copy()
                new_amount = int(round(
                    reference_rate * hospital_multiplier * ratio / 10
                ) * 10)
                actual_ratio = (
                    new_amount / reference_rate / hospital_multiplier
                )
                if mechanism != "package_swap" and actual_ratio < 1.0:
                    continue

                new_claim["claimed_amount"] = new_amount
                new_features = base_row.copy()
                new_features["raw_ratio"] = new_amount / reference_rate
                new_features["adj_ratio"] = actual_ratio
                add_candidate(
                    new_claim, new_features,
                    f"reduced_upcoding_amount_ratio_{actual_ratio:.3f}",
                )

        # ----- Duplicate: change resubmission timing within its configured tier.
        elif fraud_type == "duplicate":
            linked_id = str(detail.get("linked_claim_id", ""))
            if linked_id and linked_id in claim_index.index:
                source = claim_index.loc[linked_id]
                source_date = pd.Timestamp(source["admission_date"])
                current_date = pd.Timestamp(claim["admission_date"])
                direction = 1 if current_date >= source_date else -1

                shift_range = (
                    duplicate_cfg["obvious_shift_days"]
                    if tier == "obvious"
                    else duplicate_cfg["subtle_shift_days"]
                )
                lo, hi = int(shift_range[0]), int(shift_range[1])
                shifts = sorted({
                    lo, hi, int(round((lo + hi) / 2))
                })

                for shift in shifts:
                    new_date = source_date + pd.Timedelta(days=direction * shift)
                    if not (window_start <= new_date <= window_end):
                        continue
                    if new_date == current_date:
                        continue

                    new_claim = claim.copy()
                    los = int(claim["length_of_stay"])
                    new_claim["admission_date"] = new_date
                    new_claim["discharge_date"] = (
                        new_date + pd.Timedelta(days=los)
                    )
                    new_features = base_row.copy()
                    new_features = temporal_features_for_claim(
                        new_claim, claims, new_features
                    )
                    add_candidate(
                        new_claim, new_features,
                        f"duplicate_resubmission_shift_{shift}_days",
                    )

        # ----- Phantom: change hospital only; retain the impossible package or
        # overlapping stay that makes the claim phantom.
        elif fraud_type == "phantom":
            package_code = claim["procedure_package_code"]
            package_rows = [
                p for p in params["packages"]
                if p["package_code"] == package_code
            ]
            if package_rows:
                specialty = package_rows[0]["specialty"]
                eligible = hospitals[
                    hospitals["specialties"].fillna("").apply(
                        lambda value: specialty in str(value).split("|")
                    )
                    & (hospitals["hospital_id"] != claim["hospital_id"])
                ].copy()

                if not eligible.empty:
                    sample_n = min(
                        args.max_hospital_candidates, len(eligible)
                    )
                    sampled = eligible.iloc[
                        rng.choice(len(eligible), size=sample_n, replace=False)
                    ]

                    for _, hospital in sampled.iterrows():
                        new_hospital_id = hospital["hospital_id"]
                        new_claim = claim.copy()
                        old_hospital_id = claim["hospital_id"]
                        new_claim["hospital_id"] = new_hospital_id
                        new_claim["doctor_id"] = choose_doctor(
                            doctors, new_hospital_id, specialty,
                            claim.get("doctor_id"),
                        )

                        new_features = base_row.copy()
                        new_features = set_hospital_features(
                            new_features,
                            new_claim,
                            old_hospital_id,
                            claims,
                            hospital_features,
                            hospital_feature_cols,
                            district_col,
                            optional_features,
                            train_hospital_mean,
                            overall_train_mean,
                        )
                        add_candidate(
                            new_claim, new_features,
                            f"phantom_hospital_{new_hospital_id}",
                        )

        candidate_counts[claim_id] = generated

    # ---------- Score all candidate rows in one batch ----------
    selected_by_claim = {}
    if candidates:
        candidate_matrix = pd.DataFrame(
            [c["features"] for c in candidates],
            columns=model_features,
        )
        candidate_scores = detector_scores(model, candidate_matrix, args.detector)

        for candidate, score in zip(candidates, candidate_scores):
            cid = candidate["claim_id"]
            score = float(score)
            current_best = selected_by_claim.get(cid)

            if (
                current_best is None
                or score < current_best["score_after"]
            ):
                selected_by_claim[cid] = {
                    **candidate,
                    "score_after": score,
                }

    # ---------- Apply only score-reducing candidates to a copy ----------
    adaptive_claims = claims.drop(columns=["_split"]).copy()
    adaptive_claims = adaptive_claims.set_index("claim_id", drop=False)
    summary_rows = []

    for claim_id in target_ids:
        before = float(baseline_score_map[claim_id])
        selected = selected_by_claim.get(claim_id)

        if selected is not None and selected["score_after"] < before:
            for col, value in selected["claim"].items():
                if col in adaptive_claims.columns:
                    adaptive_claims.at[claim_id, col] = value
            after = float(selected["score_after"])
            action = selected["description"]
            changed = True
        else:
            after = before
            action = "no_valid_score_reducing_candidate"
            changed = False

        summary_rows.append({
            "claim_id": claim_id,
            "fraud_type": label_index.at[claim_id, "fraud_type"],
            "severity_tier": label_index.at[claim_id, "severity_tier"],
            "score_before": before,
            "score_after_candidate_search": after,
            "score_reduction": before - after,
            "changed": changed,
            "candidate_count": candidate_counts.get(claim_id, 0),
            "adaptation": action,
        })

    adaptive_claims = adaptive_claims.reset_index(drop=True)
    summary = pd.DataFrame(summary_rows)

    # Re-score the modified target-split fraud claims using the selected candidate
    # feature rows. This is the black-box score after adaptation.
    after_scores = dict(zip(
        summary["claim_id"],
        summary["score_after_candidate_search"],
    ))
    summary["score_after"] = summary["claim_id"].map(after_scores)

    # Preserve the original detailed labels and linkage metadata.
    adaptive_labels = fraud_details.copy()

    claims_out = output_dir / "adaptive_claims_round_1.csv"
    labels_out = output_dir / "adaptive_labels_round_1.csv"
    summary_out = output_dir / "adaptation_summary.csv"

    adaptive_claims.to_csv(claims_out, index=False)
    adaptive_labels.to_csv(labels_out, index=False)

    # Rebuild the complete feature table after all selected edits. This captures
    # interactions between multiple changed claims (timing, overlaps, volumes).
    build_script = Path(__file__).with_name("build_features.py")
    rebuilt_tag = f"{args.tag}_round1"
    subprocess.run(
        [
            sys.executable,
            str(build_script),
            "--claims", str(claims_out),
            "--hospitals", str(data_dir / "hospitals.csv"),
            "--params", args.params,
            "--out", str(output_dir),
            "--tag", rebuilt_tag,
        ],
        check=True,
    )

    rebuilt_features_path = output_dir / f"features_{rebuilt_tag}.csv"
    rebuilt_labels_path = output_dir / f"labels_{rebuilt_tag}.csv"
    rebuilt_X = pd.read_csv(rebuilt_features_path).set_index("claim_id")
    rebuilt_Y = pd.read_csv(rebuilt_labels_path).set_index("claim_id")

    # Verify the same patient-level test fraud claims are still present.
    if not set(target_ids).issubset(rebuilt_X.index):
        raise ValueError("Some target fraud claims are missing after feature rebuild")
    if not (rebuilt_Y.loc[target_ids, "is_fraud"] == 1).all():
        raise ValueError("Fraud labels changed during adaptive feature rebuild")

    final_matrix = rebuilt_X.loc[target_ids, model_features].apply(
        pd.to_numeric, errors="raise"
    )
    final_scores = detector_scores(model, final_matrix, args.detector)
    final_score_map = dict(zip(target_ids, final_scores))

    summary["score_after_candidate_search"] = summary["score_after"]
    summary["score_after"] = summary["claim_id"].map(final_score_map)
    summary["score_reduction"] = (
        summary["score_before"] - summary["score_after"]
    )
    summary["reduced_after_full_rebuild"] = (
        summary["score_reduction"] > 0
    )
    summary.to_csv(summary_out, index=False)

    changed_count = int(summary["changed"].sum())
    mean_before = float(summary["score_before"].mean())
    mean_after = float(summary["score_after"].mean())

    print("\nAdaptive fraudster completed.")
    print(f"Dataset:              {args.tag}")
    print(f"Detector:             {args.detector}")
    print(f"Feature variant:      {args.variant}")
    print(
        f"Fraud {args.target_split.upper()} claims: "
        f"{len(target_ids):,}"
    )
    print(f"Claims modified:      {changed_count:,}")
    print(f"Mean score before:    {mean_before:.4f}")
    print(f"Mean score after:     {mean_after:.4f}")
    print(f"Mean score reduction: {mean_before - mean_after:.4f}")
    print("\nSaved:")
    print(f"  {claims_out}")
    print(f"  {labels_out}")
    print(f"  {summary_out}")
    print(f"  {rebuilt_features_path}")
    print(f"  {rebuilt_labels_path}")
    print("\nNote: only fraudulent claims in the selected split were targeted; other splits were not edited in the raw claims file.")


if __name__ == "__main__":
    main()
