@echo off
REM Second half of the round-2 experiment. Run from the fraud_project folder, AFTER the four
REM adaptive_fraudster steps have already finished (floor/sampled, test/train).
REM Needs src\retrain_adaptive_detector_v2.py (the new version with --round2-adaptive-dir).

set F=python src/adaptive_fraudster.py --dir data_v3 --tag 5pct --detector xgboost --variant with_district --params config/params.json
set R=python src/retrain_adaptive_detector_v2.py --dir data_v3 --tag 5pct --mode augment

REM ===== A. floor fraudster =====
%R% --train-adaptive-dir data_v3/adaptive/5pct_floor_train --test-adaptive-dir data_v3/adaptive/5pct_floor_test
%F% --target-split test --model-path data_v3/adaptive/5pct_floor_train/retrained_v2_augment/xgboost_5pct_with_district_retrained_augment.joblib --output-name 5pct_floor_round2 --seed-offset 903
%R% --train-adaptive-dir data_v3/adaptive/5pct_floor_train --test-adaptive-dir data_v3/adaptive/5pct_floor_test --round2-adaptive-dir data_v3/adaptive/5pct_floor_round2 --reuse-retrained

REM ===== B. sampled fraudster =====
%R% --train-adaptive-dir data_v3/adaptive/5pct_sampled_train --test-adaptive-dir data_v3/adaptive/5pct_sampled_test
%F% --ratio-mode sampled --target-split test --model-path data_v3/adaptive/5pct_sampled_train/retrained_v2_augment/xgboost_5pct_with_district_retrained_augment.joblib --output-name 5pct_sampled_round2 --seed-offset 913
%R% --train-adaptive-dir data_v3/adaptive/5pct_sampled_train --test-adaptive-dir data_v3/adaptive/5pct_sampled_test --round2-adaptive-dir data_v3/adaptive/5pct_sampled_round2 --reuse-retrained

echo.
echo Done. Results: data_v3\adaptive\5pct_floor_train\retrained_v2_augment\retraining_v2_metrics.csv
echo               data_v3\adaptive\5pct_sampled_train\retrained_v2_augment\retraining_v2_metrics.csv
