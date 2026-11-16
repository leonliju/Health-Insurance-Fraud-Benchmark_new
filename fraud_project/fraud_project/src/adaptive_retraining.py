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