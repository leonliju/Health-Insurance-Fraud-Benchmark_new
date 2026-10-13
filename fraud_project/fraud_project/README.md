# Health Insurance Fraud Benchmark: data generator and fraud injection (v2)

Builds a synthetic, India-informed health-insurance claims dataset with labelled fraud.
Everything is synthetic. **Prices come from the official AB PM-JAY HBP 2.2 master (National Health Authority);**
almost everything else is an assumption, listed with its source in `data/parameter_sources.csv`.
Same code and seeds give byte-identical files.

## Quick start

```bash
pip install -r requirements.txt
python src/generate_claims.py --config config/params.json --out data_full
python src/validate_claims.py --config config/params.json --data data_full          # 23 checks on the clean data
python src/inject_fraud.py --config config/params.json --data data_full --out data_full --rates 0.02 0.05 0.08
cd src && python validate_fraud.py --config ../config/params.json --data ../data_full --rate 0.05   # about 40 checks
cd .. && python src/make_parameter_table.py --config config/params.json --hbp reference/hbp_2_2_master_extracted.csv --out data_full
```

`data/` holds a small 5,000-patient sample (with fraud). `data_full/` is git-ignored. The full dataset
(`dataset_v2`) is shared separately; read its `DATA_DICTIONARY.md` first.

## Files

| File | Purpose |
|---|---|
| `config/params.json` | All settings: 16 packages with HBP 2.2 rates, bed-day rates, hospital incentives, fraud settings |
| `src/generate_claims.py` | Hospitals, doctors, patients, clean claims |
| `src/validate_claims.py` | Checks on the clean data |
| `src/inject_fraud.py` | Upcoding, duplicate and phantom fraud (obvious and subtle) at 2%, 5%, 8% |
| `src/validate_fraud.py` | Checks labels against the clean data and each fraud rule |
| `src/make_parameter_table.py` | Builds `parameter_sources.csv` and verifies every rate against the extracted HBP master |
| `src/extract_hbp.py` | Optional: rebuilds the extracted HBP table from the official PDF (needs `pdfplumber`) |
| `reference/hbp_2_2_master_extracted.csv` | The HBP 2.2 master, machine-extracted (spot-check against the PDF before quoting) |

## The 17-column claims table

`claim_id, patient_id, hospital_id, doctor_id, admission_date, discharge_date, patient_age, patient_sex,
patient_district, diagnosis_code, procedure_package_code, claimed_amount, reference_package_rate,
length_of_stay, is_fraud, fraud_type, severity_tier`

Columns 15 to 17 are the **answer key** and must never be model inputs.

## What makes it realistic (and where each rule comes from)

| Rule | Source |
|---|---|
| Package prices; medical packages priced per bed-day (2,100 / 3,300 / 8,500 / 9,000) | HBP 2.2 master |
| Hospital incentives compound on the base rate (NABH 10-15%, metro 10%, aspirational district 10%, PG/DNB 10%) | NHA HBP 2.2 user guidelines, FAQ 26 |
| Caesarean and high risk delivery mainly at public hospitals | User guidelines 3.13 and 4.1 |
| Repeat limits and minimum gaps for one-time surgeries | User guidelines FAQ 34 (exact numbers assumed) |
| Dialysis about every 3 days; breast-cancer chemotherapy about every 22 days | Checked against a 1,164-patient Synthea run |
| Fraud types match NHA's own concerns (upcoding across specialties, abuse-prone packages, repeat bookings) | User guidelines 2.2.4, 2.2.10, FAQ 34 |

## Design notes

* **Rule-based, not machine-learned.** No GAN or other learned generator, because there is no real Indian claims data to learn from.
* **Synthea was evaluated and not used as the data source:** mostly outpatient visits, only about 4 of 16 packages present, SNOMED codes, US costs.
* **Package mapping:** PKG-08 is high risk delivery (no normal-delivery package exists in the HBP master); PKG-03 is heart catheterization (angiography alone is not a package). One representative procedure per package.
* **Genuine amounts exceed the reference rate legitimately** (hospital multiplier up to 1.53), so a plain "amount above rate" rule cannot find subtle fraud. Detectors must use hospital attributes.
* **Claim ids are renumbered in date order after injection**, so ids and row order do not reveal fraud.
* **Duplicates never copy dialysis or chemotherapy**, which genuinely repeat.
* **Fraud clusters** at about 10% of hospitals (about 60% of fraud): realistic, but it makes hospital-level features easier. Say so in the report.

## Known limits

1. Prices are official; **84 other parameters are assumptions** (treatment frequencies, ages, stays, bed mix, hospital attributes, fraud rules).
2. Not calibrated to real claims; results describe this synthetic setting only.
3. State-level rate changes (up to +10%, or any reduction) are not modelled; incentives are assumed to apply everywhere.
4. Districts and hospital attributes are placeholders / random.
5. Identity-mismatch and collusion fraud are not implemented.
6. About 0.3% of genuine claims overlap another genuine stay of the same patient by chance.

## Next steps

1. Split by patient into train / calibration / test (keep a separate calibration set for conformal prediction).
2. Features (including hospital multiplier) and baseline detectors (Isolation Forest, Random Forest, XGBoost).
3. Adaptive fraudster loop and conformal triage.
4. Replace the highest-impact assumptions with published statistics (treatment frequencies first).
