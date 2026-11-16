
import pandas as pd

# Load the adaptation results
df = pd.read_csv(
    "data_v3/adaptive/5pct/adaptation_summary.csv"
)

# Group results by fraud type and severity
summary = df.groupby(
    ["fraud_type", "severity_tier"]
).agg(
    total_claims=("claim_id", "count"),
    modified=("changed", "sum"),
    modification_rate=("changed", "mean"),
    mean_score_before=("score_before", "mean"),
    mean_score_after=("score_after", "mean"),
    mean_score_reduction=("score_reduction", "mean")
).round(4)

# Display results
print("\nADAPTATION ANALYSIS BY FRAUD TYPE AND SEVERITY")
print("=" * 65)
print(summary.to_string())

# Save results
summary.to_csv(
    "data_v3/adaptive/5pct/fraud_type_analysis.csv"
)

print("\nAnalysis saved successfully!")
