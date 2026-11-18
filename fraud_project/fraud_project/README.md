# Measuring Detector Robustness to Adaptive Fraud in Health Insurance Claims

**A synthetic, India-informed claims benchmark with adaptive-fraud evaluation and conformal triage**

MSc Data Science (3MDS), Department of Statistics and Data Science, CHRIST (Deemed to be University), Bangalore

| | |
|---|---|
| Authors | Leon C Liju, Hans Mathews Chacko |
| Guide | Dr. Ummesalma M |
| Dataset version | **v3** (see [Dataset versions](#dataset-versions)) |
| Status | Benchmark, baseline detectors, adaptive fraudster and adversarial retraining done; conformal triage and dashboard still to do |

> **Important:** all data in this repository is **synthetic**. It is *India-informed* (official PM-JAY package prices, official incentive rules), not *India-calibrated*. Results describe the synthetic setting and do not establish performance on real Indian claims.

---

## 1. What this project is about

Fraud detectors for health insurance are normally scored on fraud that never reacts to them. Real fraudsters do react: once they learn what gets flagged, they change how they bill. A detector that scores well on static fraud can therefore look much better than it really is.

A short illustration (not real data): an insurer trains a detector that flags bills far above the package price, and it tests well. A hospital notices its large inflated bills are being flagged, so it switches to small inflations that hide inside normal price variation. The detector's test score does not change, yet the new fraud slips through.

**Research question:** how much does detector performance drop when fraudsters adapt, can retraining recover it, and can a recall guarantee (conformal triage) be kept under a limit on how many claims investigators can review?

No public labelled Indian claims data exists, so we build a benchmark ourselves.

### Objectives

1. **Static robustness:** how well do Isolation Forest, Random Forest and XGBoost catch static fraud at 2%, 5% and 8% fraud?
2. **Adaptive gap:** how much does performance drop when a simulated fraudster evades the deployed detector?
3. **Conformal triage:** can a stated recall target hold under a review-capacity limit, and does it survive adaptation?
4. **Human support:** show results on an investigator dashboard. The system ranks and explains claims; it never denies one.

---

## 2. System design (four modules)

| Module | What it does | Status |
|---|---|---|
| **M1** Synthetic benchmark and fraud injection | Generates hospitals, doctors, patients and clean claims from official HBP 2.2 prices; injects labelled fraud at 2/5/8% | Done (v3) |
| **M2** Features and detectors | Builds features with a patient-level 60/20/20 split; trains Isolation Forest, Random Forest, XGBoost | Done for 5% fraud; 2% and 8% to run |
| **M3** Adaptive evaluation and conformal triage | Adaptive fraudster, adversarial retraining, round-2 attack; conformal threshold and coverage tests | Adaptive part done; conformal triage **to do** |
| **M4** Investigator dashboard | Streamlit app with a control panel and four tabs | **To do** |

Out of scope: real claims, identity mismatch, collusion, automatic denial of claims, image-level document forensics. Document cross-verification and an LLM-guided fraudster are optional extensions.

---

## 3. Repository layout

```
config/params.json                 all generator, fraud and (v3) readmission settings
reference/                         extracted HBP 2.2 package master (optional rebuild with extract_hbp.py)
src/
  extract_hbp.py                   rebuild the HBP 2.2 table from the official PDF (needs pdfplumber)
  generate_claims.py               M1: hospitals, doctors, patients, clean claims
  inject_fraud.py                  M1: inject labelled fraud (3 types x 2 tiers)
  validate_claims.py               checks on the clean data
  validate_fraud.py                checks on each fraud file (+ overlap summary)
  make_parameter_table.py          every parameter, its source, sourced vs assumption
  build_features.py                M2: features and patient-level split
  train_detectors.py               M2: Isolation Forest, Random Forest, XGBoost
  adaptive_fraudster.py            M3: black-box adaptive fraudster (floor / sampled modes)
  adaptive_evaluation.py           M3: original model on clean vs adapted test claims
  analyze_adaptation.py            M3: adaptation summary by fraud type and tier
  analyze_feature_shift.py         M3: which features changed + amount-ratio histogram
  retrain_adaptive_detector_v2.py  M3: adversarial retraining (augment / replace), round-2 test
  archive/                         superseded scripts kept for transparency (see section 8)
run_round2_experiment.bat          Windows: runs the whole round-2 experiment
run_round2_remaining.bat           Windows: only the retraining and round-2 steps
docs/                              design document, presentation, speaking script
results/                           small summary CSVs (not the large data files)
data/                              small 5,000-patient sample (with fraud), committed
data_v3/                           full generated data (not committed, see section 4)
parameter_sources.csv              every parameter, its value, its source, sourced vs assumption
```

The full dataset is shared separately; read its `DATA_DICTIONARY.md` first.

Large generated files (data, models) are **not** committed. Everything is reproducible from `config/params.json` and the fixed seeds.

---

## 4. Reproducing the pipeline

Requirements: Python 3.10+ and `pip install -r requirements.txt`. Commands below are for Windows (`cmd`) from the repository root; they also work on Linux/macOS.

### Step 1: generate and validate the data (M1)

```
python src/generate_claims.py --config config/params.json --out data_v3
python src/inject_fraud.py --config config/params.json --data data_v3 --out data_v3 --rates 0.02 0.05 0.08
python src/validate_claims.py --config config/params.json --data data_v3
python src/validate_fraud.py --config config/params.json --data data_v3 --rate 0.02
python src/validate_fraud.py --config config/params.json --data data_v3 --rate 0.05
python src/validate_fraud.py --config config/params.json --data data_v3 --rate 0.08
```

**Expected output (use this to check your copy matches):**

| Item | Value |
|---|---|
| Patients / hospitals / doctors | 50,000 / 300 / 1,523 |
| Clean claims | 105,741 (includes 3,972 genuine readmissions) |
| Claims with 2% / 5% / 8% fraud | 107,026 / 109,012 / 111,073 |
| Validators | `ALL CHECKS PASSED` for the clean file and each fraud rate |

### Step 2: features and baseline detectors (M2)

```
python src/build_features.py --claims data_v3/claims_fraud_5pct.csv --hospitals data_v3/hospitals.csv --params config/params.json --out data_v3 --tag 5pct
python src/train_detectors.py --dir data_v3 --tag 5pct
```

Outputs `features_5pct.csv`, `labels_5pct.csv`, `feature_list_5pct.json`, saved models, and `metrics_5pct.csv`. Review capacity is 5% of test claims (K = 1,090 of 21,806 test claims). Repeat with `--tag 2pct` and `--tag 8pct` after pointing `--claims` at the matching file.

### Step 3: adaptive fraudster and evaluation (M3)

```
python src/adaptive_fraudster.py --dir data_v3 --tag 5pct --detector xgboost --variant with_district --params config/params.json --target-split test
python src/adaptive_evaluation.py
python src/analyze_adaptation.py
python src/analyze_feature_shift.py --dir data_v3 --tag 5pct --split test
```

`adaptive_evaluation.py` and `analyze_adaptation.py` have their folders set at the top of the file (default `data_v3/adaptive/5pct`).

### Step 4: adversarial retraining and the round-2 attack (M3)

Run both fraudster types end to end (this takes a while):

```
run_round2_experiment.bat > round2_log.txt 2>&1
```

Or run the pieces yourself, for example for the floor fraudster:

```
python src/adaptive_fraudster.py --dir data_v3 --tag 5pct --detector xgboost --variant with_district --params config/params.json --target-split test  --output-name 5pct_floor_test  --seed-offset 901
python src/adaptive_fraudster.py --dir data_v3 --tag 5pct --detector xgboost --variant with_district --params config/params.json --target-split train --output-name 5pct_floor_train --seed-offset 901
python src/retrain_adaptive_detector_v2.py --dir data_v3 --tag 5pct --mode augment --train-adaptive-dir data_v3/adaptive/5pct_floor_train --test-adaptive-dir data_v3/adaptive/5pct_floor_test
python src/adaptive_fraudster.py --dir data_v3 --tag 5pct --detector xgboost --variant with_district --params config/params.json --target-split test --model-path data_v3/adaptive/5pct_floor_train/retrained_v2_augment/xgboost_5pct_with_district_retrained_augment.joblib --output-name 5pct_floor_round2 --seed-offset 903
python src/retrain_adaptive_detector_v2.py --dir data_v3 --tag 5pct --mode augment --train-adaptive-dir data_v3/adaptive/5pct_floor_train --test-adaptive-dir data_v3/adaptive/5pct_floor_test --round2-adaptive-dir data_v3/adaptive/5pct_floor_round2 --reuse-retrained
```

For the sampled fraudster add `--ratio-mode sampled` to the fraudster commands (seed offsets 911, 912, 913 are used in the batch file).

---

## 5. How the benchmark works

**Synthetic world.** 50,000 patients, 300 hospitals, 16 treatment packages, claims from 2023-01-01 to 2025-12-31. Package prices come from the official PM-JAY HBP 2.2 master (medical packages are priced per bed-day). A claim's legitimate price is `reference rate x hospital incentive multiplier x small noise`, where the multiplier follows the NHA incentive rules (NABH, metro, aspirational district, PG/DNB). Dialysis and chemotherapy repeat by design (patterns cross-checked against a 1,164-patient Synthea run); one-time surgeries have maximum counts and minimum gaps. The generator is **rule-based, not learned**: with no real Indian claims to learn from, a generative model would only learn our own assumptions.

**Fraud injection.** Three types, each with an obvious and a subtle tier (50% each):

| Type | Obvious tier | Subtle tier |
|---|---|---|
| Upcoding (changed in place) | Large inflation of the legitimate price, or a swap to a costlier package | Small inflation that overlaps normal price noise |
| Duplicate (new row) | Copy at another hospital within days, same amount | Copy days later with a near-identical amount |
| Phantom (new row) | Package impossible for the patient (wrong sex or age) | Plausible package overlapping another stay at a different hospital |

Mix: 40% upcoding, 30% duplicate, 30% phantom. Fraud is clustered: about 10% of hospitals receive about 60% of it. Labels are kept in separate columns and files (`fraud_labels_*.csv`); answer-key columns are never used as model inputs (`build_features.py` asserts this).

**Features (23 + 1 optional).** Amount (raw and hospital-adjusted ratio), plausibility (sex and age rules, repeat-package flag), timing (gaps between same-patient, same-package claims), overlap (overlapping stays, including at another hospital), and hospital attributes (multiplier, volume, mean adjusted ratio from the train split, type). The optional `in_district` feature is the "district shortcut" (see limitations).

### 5.1 The 17-column claims table

`claim_id, patient_id, hospital_id, doctor_id, admission_date, discharge_date, patient_age, patient_sex, patient_district, diagnosis_code, procedure_package_code, claimed_amount, reference_package_rate, length_of_stay, is_fraud, fraud_type, severity_tier`

Columns 15 to 17 (`is_fraud`, `fraud_type`, `severity_tier`) are the **answer key** and must never be model inputs. Claim ids are renumbered in date order after injection, so ids and row order cannot reveal fraud. The same code and seeds give byte-identical files.

### 5.2 Where the realism rules come from

| Rule | Source |
|---|---|
| Package prices; medical packages priced per bed-day (2,100 / 3,300 / 8,500 / 9,000) | HBP 2.2 master |
| Hospital incentives compound on the base rate (NABH 10-15%, metro 10%, aspirational district 10%, PG/DNB 10%) | NHA HBP 2.2 user guidelines, FAQ 26 |
| Caesarean and high risk delivery mainly at public hospitals | User guidelines 3.13 and 4.1 |
| Repeat limits and minimum gaps for one-time surgeries | User guidelines FAQ 34 (exact numbers are assumptions) |
| Dialysis about every 3 days; breast-cancer chemotherapy about every 22 days | Checked against a 1,164-patient Synthea run |
| Fraud types match NHA's own concerns (upcoding across specialties, abuse-prone packages, repeat bookings) | User guidelines 2.2.4, 2.2.10, FAQ 34 |

### 5.3 Design notes

- **Rule-based, not machine-learned.** No GAN or other learned generator, because there is no real Indian claims data to learn from.
- **Synthea was evaluated and not used as the data source:** it is mostly outpatient visits, only about 4 of the 16 packages are present, it uses SNOMED codes and US costs. It was used only as a reference for dialysis and chemotherapy repeat patterns.
- **Package mapping:** PKG-08 is high risk delivery (the HBP master has no normal-delivery package); PKG-03 is heart catheterization (angiography alone is not a package). One representative procedure per package.
- **Genuine amounts exceed the reference rate legitimately.** The hospital multiplier can reach 1.53 (compounded incentives), so a plain "amount above rate" rule cannot find subtle fraud; detectors must use hospital attributes.
- **Duplicates never copy dialysis or chemotherapy**, which genuinely repeat.
- **Fraud clusters** at about 10% of hospitals (about 60% of fraud). This is realistic but makes hospital-level features easier; say so in the report.

---

## 6. Dataset versions

| Version | What it was | Problem found |
|---|---|---|
| v1 | First full dataset (placeholder package rates) | Rates not official |
| v2 | Official HBP 2.2 rates for all 16 packages, per-bed-day medical pricing, hospital incentive multipliers, public-reserved deliveries, parameter source table. 101,769 clean claims | **A pilot run showed subtle upcoding was separable by construction**: genuine noise was 0.90 to 1.10 after the hospital multiplier and subtle upcoding was 1.10 to 1.25, so the two never overlapped |
| **v3** (current) | Fixes the above (below) | |

**v3 changes** (set in `config/params.json`):

| Setting | v2 | v3 |
|---|---|---|
| Genuine price noise (sigma / clip) | 0.04 / 0.90 to 1.10 | 0.07 / 0.85 to 1.30 |
| Subtle upcoding ratio | 1.10 to 1.25 | 1.06 to 1.22 |
| Subtle duplicates | 3 to 14 days, amount within 3% | 3 to 30 days, amount within 6% |
| Genuine readmissions | none | 10% of medical admissions get a legitimate repeat 3 to 21 days later (same hospital with probability 0.6) |

In v3 about 39% of subtle upcoding falls inside the genuine 5th to 95th percentile band (v2: 0%), so subtle upcoding can no longer be found from the amount alone. **The v2 to v3 revision was made after seeing a pilot result and must be disclosed in the report.**

---

## 7. Results so far (5% fraud, v3, XGBoost, review capacity 5%)

All numbers below come from the authors' own runs. One seed and one fraud rate, so treat them as preliminary.

### 7.1 Baseline and adaptive attack

| Model | Test set | PR-AUC | Precision@K | Recall@K |
|---|---|---|---|---|
| XGBoost | Clean test claims | 0.8605 | 0.8239 | 0.7565 |
| XGBoost | Adapted by the floor fraudster | 0.5804 | 0.5587 | 0.5131 |
| XGBoost | Adapted by the sampled fraudster | 0.6417 | 0.6101 | 0.5602 |

Isolation Forest and Random Forest baselines at 2%, 5% and 8% are still to be added (see `metrics_5pct.csv`).

### 7.2 The adaptive fraudster

A black-box fraudster that may only query a detector's score. For each fraud claim it proposes small, valid modifications, keeps the one with the lowest score, and then rebuilds all features so interactions are captured. Rules: the claim must remain fraud and keep valid values.

- **Upcoding:** lowers the amount ratio, never below 1.06 (the lowest subtle ratio). *Floor* mode always tries the minimum; *sampled* mode tries 6 random ratios (a fraudster with a limited number of queries).
- **Duplicate:** shifts the resubmission date within its tier's range.
- **Phantom:** moves the claim to another hospital that offers the specialty.

What changed (floor fraudster, test fraud claims):

| Fraud type | Change | Evidence |
|---|---|---|
| Upcoding | Amount dropped to the floor | Adjusted ratio 2.07 to 1.06 (obvious), 1.14 to 1.06 (subtle); about 90% of adapted claims sit within 0.001 of 1.06, against 0.8% of genuine claims |
| Phantom | Moved to larger hospitals with cleaner history | `hosp_volume` 501 to 660; `hosp_mean_adj_ratio` 1.04 to 1.01 |
| Duplicate | Small date shifts | Overlap and gap clues mostly survive |

The spike at 1.06 means this fraudster is highly predictable; `analyze_feature_shift.py` draws the histogram.

### 7.3 Adversarial retraining

The first retraining attempt **replaced** each original fraud claim with its adapted version and scored on shifted test features, so the model forgot obvious upcoding (recall 95% to 7%). Corrected approach (`retrain_adaptive_detector_v2.py`): keep every original training claim and **add** the adapted copies (`--mode augment`), keep all features in the original feature space, and score on both clean and adapted test claims. A second round lets the fraudster attack the **retrained** model.

PR-AUC:

| Fraudster | Original, clean | Original under attack | Retrained, clean | Retrained, 1st attack | Retrained, round 2 |
|---|---|---|---|---|---|
| Floor | 0.8605 | 0.5804 | 0.8401 | 0.8823 | 0.7604 |
| Sampled | 0.8605 | 0.6417 | 0.8456 | 0.7888 | 0.7126 |

Recall@K for subtle upcoding (clean original: 0.327):

| Fraudster | Original under attack | Retrained, 1st attack | Retrained, round 2 |
|---|---|---|---|
| Floor | 0.098 | 0.710 | 0.118 |
| Sampled | 0.073 | 0.363 | 0.192 |

**Reading the results**

- Adaptive fraud cuts PR-AUC by roughly 0.2 to 0.3.
- Retraining on original plus adapted fraud recovers part of the loss at a small cost on clean data (about 0.015 to 0.02). Against an attacker who re-adapts, the gain over the unprotected model is +0.18 (floor) and +0.07 (sampled), about half of the gain seen on the first attack.
- Obvious upcoding stays well detected after retraining under the floor fraudster; duplicates and phantoms are stable. **Subtle upcoding remains vulnerable.**
- The sampled fraudster is a weaker attacker (6 queries per claim), so these numbers are not a worst case.
- Claim for the report: retraining is a **partial** defence, not a fix.

---

## 8. Experiment history and lessons (for transparency)

- `src/archive/retrain_adaptive_detector.py` (first retraining script) trained on a table where adapted fraud replaced the originals, and scored the "original" model on test features rebuilt after train adaptation. Hospital-level features (`hosp_volume`, `hosp_mean_adj_ratio`) are computed from the whole train split, so they shifted for most test rows. That explains why the original model scored 0.8605 in one table and 0.8051 in another. Do not cite its numbers.
- `adaptive_retraining.py` was only a pasted fragment of the end of `adaptive_fraudster.py` and can be deleted.
- Lesson: hospital-level aggregate features create dependence between splits; anything that edits one split must keep features in the original space.

---

## 9. Known limitations

- Synthetic data only; most parameters other than package prices and incentive rules are documented assumptions (`parameter_sources.csv` lists each one).
- Only 16 packages and 3 fraud types; identity mismatch and collusion are out of scope.
- Known generator clues (for example the `in_district` shortcut) could be exploited by a detector, and hospital-level features are computed from all claims including injected fraud, so they can carry label information; results are best reported with and without them.
- The adaptive fraudster is rule-based and its power depends on its assumptions (price floor, candidate set, query budget).
- One seed, one fraud rate (5%) and one detector family (XGBoost) in the adaptive experiments so far.
- Retraining uses the fraudster's own style of adaptation; real fraudsters may behave differently.
- State-level rate changes (up to +10%, or any reduction) are not modelled; incentives are assumed to apply at every qualifying hospital.
- Districts and hospital attributes are placeholders or random.
- A small share of genuine claims overlap another genuine stay of the same patient by chance (reported by `validate_fraud.py`).

---

## 10. What still needs to be done

**A. Finish the experiments (M2, M3)**

1. Run and record the three detectors at **2%, 5% and 8%** (`build_features.py` and `train_detectors.py` for each tag) and put `metrics_*.csv` into `results/`.
2. Repeat the adaptive and round-2 experiments at 2% and 8%, and with **several seeds** (change `--seed-offset`) to report spread, not single numbers.
3. Try **multi-round adversarial training** (retrain on round-2 claims as well) and a **stronger attacker** (more queries per claim); optional LLM-guided fraudster.
4. Run the same adaptive attack on Random Forest and Isolation Forest, not only XGBoost.
5. Add SHAP or feature-importance analysis, and the structural check against the Kaggle CMS Medicare dataset.
6. Report results with and without the district shortcut.

**B. Conformal triage (M3), not started**

7. Build the conformal threshold from the **calibration** split (`calib` rows from `build_features.py`): for target recall 1 - alpha and n calibration fraud claims, take `k = floor(alpha (n + 1))` and use the k-th lowest fraud score as the threshold.
8. Measure coverage on static test fraud, adapted test fraud, and after recalibration, by fraud type and tier, with Clopper-Pearson lower bounds, a one-sided binomial test, repeated patient-level re-splits and a cluster bootstrap.
9. Test the shift with a KS test and check whether recalibration restores coverage (recalibration needs labelled adapted fraud, so it lags the fraudster). Add the capacity-first mode.

**C. Dashboard (M4), not started**

10. Streamlit app: control panel (dataset, detector, review capacity, target error rate, severity filter) and four tabs (flagged-claims queue, detector performance, static vs adaptive, conformal triage). Include a footer notice that results are synthetic.

**D. Documentation and report**

11. Regenerate `parameter_sources.csv` for v3 (`make_parameter_table.py` needs the extracted HBP CSV in `reference/`).
12. Update the system design document from v2 to v3 (claim counts, noise range, subtle ranges, known generator clues) and fill in the phase table.
13. Write the results chapter and disclose the v2 to v3 revision and the retraining-method correction.
14. Optional: autoencoder baseline, document cross-verification.

---

## 11. Data and ethics

All records are synthetic; no real patient data was used. The system is designed to rank and explain claims for human investigators and never to deny a claim automatically.

## 12. Acknowledgements and sources

Package prices: AB PM-JAY Health Benefit Package master HBP 2.2 (National Health Authority). Incentive rules and abuse definitions: NHA, National Health Benefit Package 2.2 User Guidelines (November 2021). Repeat-treatment patterns were checked against a run of Synthea (a US-based simulator), which is consistency with a clinical simulator, not Indian ground truth.
