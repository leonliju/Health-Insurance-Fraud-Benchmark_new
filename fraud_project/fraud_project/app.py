from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------
# Health Insurance Fraud Detector Robustness Dashboard
# Run from project root: streamlit run app.py
# ------------------------------------------------------------
st.set_page_config(
    page_title="Health Insurance Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data_v3"
ADAPTIVE = DATA / "adaptive" / "5pct"
TRAIN_ADAPT = DATA / "adaptive" / "5pct_train_adaptation"
RETRAINED = TRAIN_ADAPT / "retrained"
AUGMENT = TRAIN_ADAPT / "retrained_v2_augment"
REPLACE = TRAIN_ADAPT / "retrained_v2_replace"


@st.cache_data(show_spinner=False)
def load_csv(path: Path):
    if not path.exists():
        return None
    return pd.read_csv(path)


def pct(value):
    return f"{value:.1%}"


def metric_delta(new, old):
    return f"{(new - old) * 100:+.1f} pp"


def show_missing(paths):
    missing = [str(p.relative_to(ROOT)) for p in paths if not p.exists()]
    if missing:
        st.warning("Some project outputs are not present. The related panels will be unavailable:\n\n" +
                   "\n".join(f"- `{m}`" for m in missing))


def header():
    st.title("🛡️ Health Insurance Fraud Intelligence")
    st.caption("Research dashboard · Synthetic India-informed health-insurance claims benchmark")
    st.info(
        "Research prototype only. The data are synthetic and results describe this benchmark, "
        "not real-world claim outcomes. Scores support human investigation; they must not be "
        "used as automatic claim-denial decisions.",
        icon="ℹ️",
    )


def overview_page():
    st.subheader("Research overview")
    labels = load_csv(DATA / "labels_5pct.csv")
    metrics = load_csv(DATA / "metrics_5pct.csv")
    claims = load_csv(DATA / "claims_fraud_5pct.csv")
    if labels is None or metrics is None:
        st.error("Required 5% dataset files are missing from data_v3/.")
        return

    test = labels[labels["split"].eq("test")].copy()
    fraud_test = test[test["is_fraud"].eq(1)]
    total_claims = len(labels)
    fraud_count = int(labels["is_fraud"].sum())
    fraud_rate = labels["is_fraud"].mean()

    best = metrics[(metrics["detector"] == "xgboost") &
                   (metrics["variant"] == "with_district")]
    baseline = best.iloc[0] if not best.empty else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total claims", f"{total_claims:,}")
    c2.metric("Fraudulent claims", f"{fraud_count:,}")
    c3.metric("Overall fraud rate", pct(fraud_rate))
    c4.metric("Test claims", f"{len(test):,}")

    st.markdown("### Baseline detector performance")
    if baseline is not None:
        a, b, c = st.columns(3)
        a.metric("XGBoost PR-AUC", f"{baseline['pr_auc']:.3f}")
        b.metric("Precision@K", f"{baseline['precision_at_K']:.3f}")
        c.metric("Recall@K", f"{baseline['recall_at_K']:.3f}",
                 help=f"K = {int(baseline['K']):,} claims (5% review capacity)")
    else:
        st.warning("Baseline XGBoost metrics were not found.")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Claims by split")
        split_counts = labels.groupby("split", as_index=False).agg(
            claims=("claim_id", "count"),
            fraud=("is_fraud", "sum")
        )
        fig = px.bar(split_counts, x="split", y="claims", color="split",
                     text="claims", title="Dataset split sizes")
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Claims")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("#### Fraud composition")
        composition = labels[labels["is_fraud"].eq(1)].groupby(
            "fraud_type", as_index=False
        ).size().rename(columns={"size": "claims"})
        fig = px.pie(composition, names="fraud_type", values="claims",
                     hole=0.55, title="Fraudulent claims by type")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Baseline model comparison")
    compare = metrics[metrics["variant"].eq("with_district")].copy()
    compare["detector"] = compare["detector"].str.replace("_", " ").str.title()
    fig = px.bar(compare, x="detector", y="pr_auc", color="detector",
                 text=compare["pr_auc"].round(3), title="PR-AUC at 5% fraud rate")
    fig.update_layout(showlegend=False, yaxis_title="PR-AUC", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    if claims is not None:
        st.caption("Claim-level source: claims_fraud_5pct.csv · Labels are kept separate from model features.")


def claim_explorer_page():
    st.subheader("Claim investigation")
    claims = load_csv(DATA / "claims_fraud_5pct.csv")
    features = load_csv(DATA / "features_5pct.csv")
    labels = load_csv(DATA / "labels_5pct.csv")
    scores = load_csv(DATA / "scores_5pct_xgboost_with_district.csv")
    if any(x is None for x in [claims, features, labels, scores]):
        st.error("Claim investigation requires claims_fraud_5pct.csv, features_5pct.csv, labels_5pct.csv and the XGBoost score file.")
        return

    claim_id = st.text_input("Enter a claim ID", placeholder="Example: C000123").strip().upper()
    if not claim_id:
        st.caption("Enter a claim ID to inspect its recorded details, engineered features and baseline XGBoost score.")
        st.dataframe(claims[["claim_id", "hospital_id", "procedure_package_code",
                             "claimed_amount", "is_fraud", "fraud_type", "severity_tier"]].head(15),
                     use_container_width=True, hide_index=True)
        return

    base = claims[claims["claim_id"].astype(str).str.upper().eq(claim_id)]
    if base.empty:
        st.warning("No matching claim ID was found.")
        return
    row = base.iloc[0]
    feat = features[features["claim_id"].eq(row["claim_id"])]
    lab = labels[labels["claim_id"].eq(row["claim_id"])]
    score = scores[scores["claim_id"].eq(row["claim_id"])]

    top = st.columns(4)
    top[0].metric("Claim ID", str(row["claim_id"]))
    top[1].metric("Claim amount", f"₹{row['claimed_amount']:,.0f}")
    top[2].metric("Recorded label", "Fraud" if int(row["is_fraud"]) else "Genuine")
    top[3].metric("Fraud category", str(row["fraud_type"]).title())

    left, right = st.columns(2)
    with left:
        st.markdown("#### Claim details")
        details = {
            "Patient ID": row["patient_id"], "Hospital ID": row["hospital_id"],
            "Doctor ID": row["doctor_id"], "Admission": row["admission_date"],
            "Discharge": row["discharge_date"], "Age": row["patient_age"],
            "Sex": row["patient_sex"], "District": row["patient_district"],
            "Diagnosis": row["diagnosis_code"], "Package": row["procedure_package_code"],
            "Reference package rate": f"₹{row['reference_package_rate']:,.0f}",
            "Length of stay": row["length_of_stay"],
        }
        st.dataframe(pd.DataFrame(details.items(), columns=["Field", "Value"]),
                     use_container_width=True, hide_index=True)
    with right:
        st.markdown("#### Baseline XGBoost score")
        if not score.empty:
            val = float(score.iloc[0]["score"])
            st.metric("Model score", f"{val:.4f}")
            st.progress(min(max(val, 0.0), 1.0))
            st.caption("Higher score indicates greater model suspicion in this benchmark; it is not a calibrated probability.")
        else:
            st.info("No score was found for this claim.")
        st.markdown("#### Recorded label metadata")
        if not lab.empty:
            st.json(lab.iloc[0].to_dict())

    if not feat.empty:
        st.markdown("#### Engineered features used by the detector")
        feature_cols = [c for c in feat.columns if c not in ["claim_id", "split"]]
        st.dataframe(feat[feature_cols].T.rename(columns={feat.index[0]: "Value"}),
                     use_container_width=True)


def model_benchmarks_page():
    st.subheader("Baseline model benchmarks")
    metrics = load_csv(DATA / "metrics_5pct.csv")
    if metrics is None:
        st.error("metrics_5pct.csv was not found.")
        return
    variant = st.selectbox("Feature variant", sorted(metrics["variant"].dropna().unique()),
                           index=0 if "with_district" in metrics["variant"].unique() else 0)
    view = metrics[metrics["variant"].eq(variant)].copy()
    view["detector"] = view["detector"].str.replace("_", " ").str.title()
    st.caption("All models are evaluated on the same held-out test split. K is the 5% review-capacity cutoff.")
    cols = st.columns(3)
    for col, metric, label in zip(cols, ["pr_auc", "precision_at_K", "recall_at_K"],
                                  ["PR-AUC", "Precision@K", "Recall@K"]):
        fig = px.bar(view, x="detector", y=metric, color="detector",
                     text=view[metric].round(3), title=label)
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title=label)
        col.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Detailed metrics")
    st.dataframe(view.drop(columns=["dataset"], errors="ignore").round(4),
                 use_container_width=True, hide_index=True)
    group_cols = [c for c in metrics.columns if c.startswith("recall_")]
    if group_cols:
        melted = view.melt(id_vars="detector", value_vars=group_cols,
                           var_name="fraud_group", value_name="recall")
        melted["fraud_group"] = melted["fraud_group"].str.replace("recall_", "", regex=False).str.replace("_", " · ")
        fig = px.bar(melted, x="fraud_group", y="recall", color="detector",
                     barmode="group", title="Fraud-type and severity recall")
        fig.update_layout(xaxis_title="", yaxis_title="Recall@K")
        st.plotly_chart(fig, use_container_width=True)


def adaptive_attack_page():
    st.subheader("Adaptive fraud attack — Round 1")
    metrics = load_csv(ADAPTIVE / "adaptive_evaluation_metrics.csv")
    groups = load_csv(ADAPTIVE / "adaptive_group_evaluation.csv")
    fraud_type = load_csv(ADAPTIVE / "fraud_type_analysis.csv")
    if metrics is None:
        st.error("Adaptive test evaluation outputs were not found.")
        return
    m = metrics.copy()
    if "Unnamed: 0" in m.columns:
        m = m.rename(columns={"Unnamed: 0": "scenario"})
    m["scenario"] = m["scenario"].astype(str)
    base = m[m["scenario"].str.contains("Baseline", case=False)].iloc[0]
    attacked = m[m["scenario"].str.contains("Adaptation", case=False)].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("PR-AUC", f"{attacked['PR_AUC']:.3f}", metric_delta(attacked["PR_AUC"], base["PR_AUC"]))
    c2.metric("Precision@K", f"{attacked['Precision_at_K']:.3f}", metric_delta(attacked["Precision_at_K"], base["Precision_at_K"]))
    c3.metric("Recall@K", f"{attacked['Recall_at_K']:.3f}", metric_delta(attacked["Recall_at_K"], base["Recall_at_K"]))
    st.caption(f"Fixed review capacity K = {int(base['K']):,}. The fixed-baseline-threshold recall changed from {base['Recall_at_baseline_threshold']:.3f} to {attacked['Recall_at_baseline_threshold']:.3f}.")

    plot = m.melt(id_vars="scenario", value_vars=["PR_AUC", "Precision_at_K", "Recall_at_K"],
                  var_name="metric", value_name="value")
    fig = px.bar(plot, x="metric", y="value", color="scenario", barmode="group",
                 title="Original test set vs adapted test set")
    fig.update_layout(xaxis_title="", yaxis_title="Metric value")
    st.plotly_chart(fig, use_container_width=True)

    if groups is not None:
        st.markdown("#### Recall@K by fraud type and severity")
        gm = groups.melt(id_vars=["fraud_type", "severity_tier", "total_claims"],
                         value_vars=["baseline_recall_at_K", "adapted_recall_at_K"],
                         var_name="scenario", value_name="recall")
        gm["scenario"] = gm["scenario"].map({
            "baseline_recall_at_K": "Baseline",
            "adapted_recall_at_K": "After adaptation"
        })
        gm["group"] = gm["fraud_type"].str.title() + " · " + gm["severity_tier"].str.title()
        fig = px.bar(gm, x="group", y="recall", color="scenario", barmode="group",
                     hover_data=["total_claims"], title="Which fraud groups became harder to detect?")
        fig.update_layout(xaxis_title="", yaxis_title="Recall@K")
        st.plotly_chart(fig, use_container_width=True)

    if fraud_type is not None:
        st.markdown("#### Adaptive fraudster modification summary")
        st.dataframe(fraud_type.round(4), use_container_width=True, hide_index=True)
        fig = px.bar(fraud_type, x="fraud_type", y="mean_score_reduction",
                     color="severity_tier", barmode="group",
                     title="Average score reduction by fraud type and severity")
        fig.update_layout(xaxis_title="", yaxis_title="Mean score reduction")
        st.plotly_chart(fig, use_container_width=True)

    hist = ADAPTIVE / "adj_ratio_histogram.png"
    if hist.exists():
        st.markdown("#### Adjusted-ratio distribution from the experiment")
        st.image(str(hist), use_container_width=True)


def retraining_page():
    st.subheader("Adaptive retraining experiments")
    v1 = load_csv(RETRAINED / "retraining_comparison_metrics.csv")
    v2a = load_csv(AUGMENT / "retraining_v2_metrics.csv")
    v2r = load_csv(REPLACE / "retraining_v2_metrics.csv")

    if v1 is not None:
        st.markdown("#### Initial adaptive retraining")
        st.dataframe(v1.round(4), use_container_width=True, hide_index=True)
        fig = px.bar(v1, x="model", y="pr_auc", color="model", text=v1["pr_auc"].round(3),
                     title="Initial retraining: PR-AUC")
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="PR-AUC")
        st.plotly_chart(fig, use_container_width=True)

    available = []
    for label, frame in [("Augmentation", v2a), ("Replacement", v2r)]:
        if frame is not None:
            tmp = frame.copy()
            tmp["strategy"] = label
            available.append(tmp)
    if available:
        st.markdown("#### V2 retraining strategies")
        combined = pd.concat(available, ignore_index=True)
        combined["evaluation"] = combined["test_set"].map({
            "A_clean_test": "Clean test set",
            "B_adapted_test": "Adapted test set"
        }).fillna(combined["test_set"])
        st.caption("Clean and adapted test sets are reported separately. Compare models within each test set; do not interpret the two test sets as interchangeable.")
        metric = st.selectbox("Metric to compare", ["pr_auc", "precision_at_K", "recall_at_K"],
                              format_func=lambda x: {"pr_auc": "PR-AUC",
                                                     "precision_at_K": "Precision@K",
                                                     "recall_at_K": "Recall@K"}[x])
        fig = px.bar(combined, x="evaluation", y=metric, color="model",
                     barmode="group", facet_col="strategy",
                     title=f"{metric.replace('_', ' ').upper()} by strategy and evaluation set",
                     hover_data=["K"])
        fig.update_layout(xaxis_title="", yaxis_title=metric.replace("_", " ").upper())
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(combined[["strategy", "model", "evaluation", "K",
                               "pr_auc", "precision_at_K", "recall_at_K"]].round(4),
                     use_container_width=True, hide_index=True)

        # Highlight the current experiment's central comparison without claiming general superiority.
        st.info(
            "Interpretation note: augmentation and replacement produce different trade-offs across "
            "clean and adapted test sets. The dashboard displays the recorded metrics; the preferred "
            "strategy should be selected only after reviewing both evaluation conditions and group-wise results."
        )
    else:
        st.info("V2 augmentation/replacement result files were not found.")

    groups_a = load_csv(AUGMENT / "retraining_v2_group_recall.csv")
    groups_r = load_csv(REPLACE / "retraining_v2_group_recall.csv")
    if groups_a is not None or groups_r is not None:
        st.markdown("#### Fraud-group recall for V2 strategies")
        all_groups = []
        for strategy, frame in [("Augmentation", groups_a), ("Replacement", groups_r)]:
            if frame is not None:
                t = frame.copy()
                t["strategy"] = strategy
                all_groups.append(t)
        gg = pd.concat(all_groups, ignore_index=True)
        gg["group"] = gg["fraud_type"].str.title() + " · " + gg["severity_tier"].str.title()
        gg["evaluation"] = gg["test_set"].map({"A_clean_test": "Clean test", "B_adapted_test": "Adapted test"}).fillna(gg["test_set"])
        chosen_set = st.selectbox("Fraud-group evaluation set", sorted(gg["evaluation"].unique()), key="group_eval")
        chosen_strategy = st.selectbox("Retraining strategy", sorted(gg["strategy"].unique()), key="group_strategy")
        gview = gg[(gg["evaluation"] == chosen_set) & (gg["strategy"] == chosen_strategy)]
        fig = px.bar(gview, x="group", y="recall_at_K", color="model",
                     barmode="group", title=f"{chosen_strategy}: group recall on {chosen_set}")
        fig.update_layout(xaxis_title="", yaxis_title="Recall@K")
        st.plotly_chart(fig, use_container_width=True)


def feature_shift_page():
    st.subheader("Feature-distribution and feature-shift analysis")
    shift = load_csv(ADAPTIVE / "feature_shift_summary.csv")
    if shift is None:
        st.error("feature_shift_summary.csv was not found.")
        return
    fraud_types = sorted(shift["fraud_type"].dropna().unique())
    chosen_type = st.selectbox("Fraud type", fraud_types)
    tiers = sorted(shift.loc[shift["fraud_type"].eq(chosen_type), "severity_tier"].dropna().unique())
    chosen_tier = st.selectbox("Severity", tiers)
    view = shift[(shift["fraud_type"] == chosen_type) &
                 (shift["severity_tier"] == chosen_tier)].copy()
    view = view.sort_values("share_feature_changed", ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("Claims in group", f"{int(view['claims'].max()):,}")
    c2.metric("Claims modified", f"{int(view['claims_modified'].max()):,}")
    c3.metric("Features summarized", f"{len(view):,}")

    fig = px.bar(view, x="feature", y="share_feature_changed",
                 title="Share of claims where each feature changed",
                 hover_data=["mean_before", "mean_after", "median_before", "median_after"])
    fig.update_layout(xaxis_title="", yaxis_title="Share changed", yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Before/after feature summaries")
    st.dataframe(view[["feature", "share_feature_changed", "mean_before", "mean_after",
                       "median_before", "median_after"]].round(4),
                 use_container_width=True, hide_index=True)
    st.caption(
        "These are aggregate summaries from the saved feature-shift analysis. "
        "They do not establish causality; they show which engineered features changed in the selected group."
    )


def reports_page():
    st.subheader("Research outputs and downloads")
    candidates = [
        ("Baseline metrics", DATA / "metrics_5pct.csv"),
        ("Adaptive evaluation metrics", ADAPTIVE / "adaptive_evaluation_metrics.csv"),
        ("Adaptive group evaluation", ADAPTIVE / "adaptive_group_evaluation.csv"),
        ("Feature shift summary", ADAPTIVE / "feature_shift_summary.csv"),
        ("Fraudster modification summary", ADAPTIVE / "fraud_type_analysis.csv"),
        ("Initial retraining comparison", RETRAINED / "retraining_comparison_metrics.csv"),
        ("V2 augmentation metrics", AUGMENT / "retraining_v2_metrics.csv"),
        ("V2 replacement metrics", REPLACE / "retraining_v2_metrics.csv"),
    ]
    for label, path in candidates:
        frame = load_csv(path)
        if frame is not None:
            st.download_button(
                f"Download {label} (CSV)",
                data=frame.to_csv(index=False).encode("utf-8"),
                file_name=path.name,
                mime="text/csv",
                key=f"download_{path.parent.name}_{path.name}",
            )
    st.caption("Downloads reproduce the experiment output tables; they do not contain personally identifiable real claimant data.")


def main():
    header()
    pages = {
        "Overview": overview_page,
        "Claim investigation": claim_explorer_page,
        "Model benchmarks": model_benchmarks_page,
        "Adaptive fraud attack": adaptive_attack_page,
        "Retraining experiments": retraining_page,
        "Feature shift analysis": feature_shift_page,
        "Reports & downloads": reports_page,
    }
    with st.sidebar:
        st.title("🛡️ Fraud Research")
        st.caption("Navigation")
        selected = st.radio("Go to", list(pages.keys()), label_visibility="collapsed")
        st.divider()
        st.markdown("**Dataset**")
        st.write("5% fraud benchmark")
        st.write("Patient-level train / calibration / test split")
        st.divider()
        st.caption("Synthetic research benchmark · Human-in-the-loop investigation")

    pages[selected]()


if __name__ == "__main__":
    main()
