# Dataset v2: Synthetic India-informed health insurance claims with labelled fraud

Everything here is **synthetic**; no real patient or claim data was used. Prices come from the official
**AB PM-JAY Health Benefit Package master HBP 2.2** (National Health Authority). Everything that is not
sourced is an assumption, and `parameter_sources.csv` lists every parameter with its source and status
(27 sourced from official documents, 84 assumptions). Same code and seeds give byte-identical files.

## Files

| File | Rows |
|---|---|
| `claims_clean.csv` | 101,769 |
| `claims_fraud_2pct.csv` | 103,005 |
| `claims_fraud_5pct.csv` | 104,917 |
| `claims_fraud_8pct.csv` | 106,900 |
| `doctors.csv` | 1,523 |
| `fraud_labels_2pct.csv` | 103,005 |
| `fraud_labels_5pct.csv` | 104,917 |
| `fraud_labels_8pct.csv` | 106,900 |
| `hospitals.csv` | 300 |
| `packages.csv` | 16 |
| `parameter_sources.csv` | 113 |
| `patients.csv` | 50,000 |
| `reference/hbp_2_2_master_extracted.csv` | official master, machine-extracted (spot-check against the PDF before quoting) |

* `claims_clean.csv`: fraud-free base. `claims_fraud_2pct/5pct/8pct.csv`: the same base with fraud injected so that fraud is that share of all rows.
* `fraud_labels_*.csv`: extra detail on each row. **Do not use it as model input.**
* `hospitals.csv`, `doctors.csv`, `patients.csv`, `packages.csv`: reference tables. Hospital attributes (type, NABH level, metro, aspirational district, PG/DNB, incentive multiplier) and package attributes are **legitimate model features**; they are what a real insurer would know.

## The 17 claim columns (same order in every claims file)

| # | Column | Meaning |
|---|---|---|
| 1 | claim_id | Unique id, renumbered in date order after injection (the id does not reveal fraud) |
| 2 | patient_id | Patient |
| 3 | hospital_id | Hospital (300 in total) |
| 4 | doctor_id | Doctor at that hospital |
| 5 | admission_date | Start of stay (2023-01-01 to 2025-12-31) |
| 6 | discharge_date | Admission plus length of stay (may fall slightly after the window) |
| 7 | patient_age | Age at admission |
| 8 | patient_sex | F or M |
| 9 | patient_district | Placeholder district name |
| 10 | diagnosis_code | ICD-10 code |
| 11 | procedure_package_code | Treatment package PKG-01 to PKG-16 (internal labels; HBP code in `packages.csv`) |
| 12 | claimed_amount | Amount claimed, rupees |
| 13 | reference_package_rate | HBP 2.2 price: the package price, or bed-days x day rate for medical packages |
| 14 | length_of_stay | Days |
| 15 | is_fraud | **Answer key.** 1 = fraud |
| 16 | fraud_type | **Answer key.** none / upcoding / duplicate / phantom |
| 17 | severity_tier | **Answer key.** none / obvious / subtle |

Columns 15 to 17 must never be used as inputs to a model.

## How prices work (the part that makes it realistic)

* **Reference rate (column 13):** official HBP 2.2 price. Medical packages (diabetic ketoacidosis, pneumonia, tuberculosis, dengue) are priced per bed-day: routine ward 2,100, HDU 3,300, ICU without ventilator 8,500, ICU with ventilator 9,000, times the length of stay.
* **Genuine claimed amount (column 12):** reference rate x the hospital's **incentive multiplier** x a small noise factor (0.90 to 1.10). NHA incentives compound: entry-level NABH 10%, full NABH 15%, metro 10%, aspirational district 10%, PG/DNB course 10% (user guidelines FAQ 26). The multiplier can reach 1.53 in principle (1.33 in this dataset), so a genuine claim can legitimately be well above the reference rate. **A detector must account for the hospital's multiplier.**

## Fraud types

| Type | What it is | Obvious tier | Subtle tier |
|---|---|---|---|
| upcoding | A real claim made more expensive (changed in place) | Amount 1.6 to 2.5 times the hospital's legitimate price, or a swap to a pricier fixed-price package of the same specialty (diagnosis no longer fits the package) | 1.10 to 1.25 times the hospital's legitimate price |
| duplicate | A new claim copying a genuine one (never dialysis or chemotherapy, which genuinely repeat) | Other hospital, 0 to 2 days apart, identical amount | 3 to 14 days apart, amount within 3%, about half at the same hospital (a resubmission) |
| phantom | A new claim for care that could not have happened | Package impossible for the patient: wrong sex, or age far from the package's typical age | A plausible package that overlaps another stay of the same patient at a different hospital |

* **Clustering:** about 10% of hospitals ("fraud-prone") receive about 60% of fraud.
* **Mix:** 40% upcoding, 30% duplicate, 30% phantom; half of each type obvious, half subtle.
* **Package swaps are rare** (about 1 in 40 upcoding claims) because most packages have no pricier fixed-price package in the same specialty.
* Identity mismatch and collusion are not included.
* About 95% of subtle upcoding claims have a raw claimed/reference ratio below the highest genuine ratio, so a plain "amount above rate" rule cannot find them.

## Realism rules taken from the NHA documents

1. Medical packages priced per bed-day (HBP 2.2).
2. Hospital incentives compound on the base rate (user guidelines FAQ 26).
3. Caesarean and high risk delivery mainly at public hospitals, private only by referral (user guidelines 3.13, 4.1).
4. The NHA IT system limits repeats and sets minimum gaps for e.g. appendicectomy and cataract (FAQ 34); the exact numbers here are assumptions.
5. Dialysis (about every 3 days) and breast-cancer chemotherapy (about 22-day cycles) repeat by design; patterns were checked against a Synthea run.

## Package mapping notes

* **PKG-08 is "high risk delivery"**, because the HBP master has no normal-delivery package.
* **PKG-03 is "right/left heart catheterization"**, because coronary angiography alone is not a package (PTCA includes it).
* Where a package has several procedures, stents or regimens, one representative procedure is used (for example PTCA with one bare-metal stent at 49,300; cataract with a phaco lens at 7,500; breast cancer chemotherapy with the EC/AC regimen at 7,200).
* HBP 2.2 lists a "procedure price" and a "total package price" that differ for some packages (appendicectomy: 11,000 and 19,700). The total package price is used.

## Known limits (please state these in the report)

1. **Prices are official; almost everything else is assumed** (84 assumptions in `parameter_sources.csv`): how common each treatment is, age patterns, stay lengths, bed-category mix, hospital attributes, public/private mix, fraud rules. State this plainly.
2. **Not calibrated to real claims.** No real Indian claims data was available; results describe this synthetic setting only.
3. States may raise HBP rates up to 10% or lower them; this is not modelled. Incentives are assumed to apply at every qualifying hospital (the NHA says they are at state discretion).
4. Districts are placeholders. Hospital attributes are random, not tied to real hospitals.
5. About 0.3% of genuine claims overlap another genuine stay of the same patient by chance.
6. Dialysis is about 7.5% of claims, from fewer than 100 patients with long courses.

## Using it

* Split by **patient** (not by claim) into train, calibration and test sets, so a duplicate and its source cannot sit in different sets.
* Build features from columns 1 to 14 plus the hospital and package tables. Never use columns 15 to 17 or the labels file as inputs.
