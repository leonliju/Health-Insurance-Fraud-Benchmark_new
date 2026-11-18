@echo off
REM Round-2 adaptive experiment. Run from the fraud_project folder:  run_round2_experiment.bat
REM Needs: data_v3 features/labels/models from build_features.py and train_detectors.py (5pct).
REM Replace adaptive_fraudster.py and retrain_adaptive_detector_v2.py in src with the new versions first.

set F=python src/adaptive_fraudster.py --dir data_v3 --tag 5pct --detector xgboost --variant with_district --params config/params.json
set R=python src/retrain_adaptive_detector_v2.py --dir data_v3 --tag 5pct --mode augment

REM ===== A. floor fraudster (always the minimum ratio) =====
%F% --target-split test  --output-name 5pct_floor_test  --seed-offset 901
%F% --target-split train --output-name 5pct_floor_train --seed-offset 901
%R% --train-adaptive-dir data_v3/adaptive/5pct_floor_train --test-adaptive-dir data_v3/adaptive/5pct_floor_test
%F% --target-split test --model-path data_v3/adaptive/5pct_floor_train/retrained_v2_augment/xgboost_5pct_with_district_retrained_augment.joblib --output-name 5pct_floor_round2 --seed-offset 903
%R% --train-adaptive-dir data_v3/adaptive/5pct_floor_train --test-adaptive-dir data_v3/adaptive/5pct_floor_test --round2-adaptive-dir data_v3/adaptive/5pct_floor_round2 --reuse-retrained

REM ===== B. sampled fraudster (6 random ratios per claim) =====
%F% --ratio-mode sampled --target-split test  --output-name 5pct_sampled_test  --seed-offset 911
%F% --ratio-mode sampled --target-split train --output-name 5pct_sampled_train --seed-offset 912
%R% --train-adaptive-dir data_v3/adaptive/5pct_sampled_train --test-adaptive-dir data_v3/adaptive/5pct_sampled_test
%F% --ratio-mode sampled --target-split test --model-path data_v3/adaptive/5pct_sampled_train/retrained_v2_augment/xgboost_5pct_with_district_retrained_augment.joblib --output-name 5pct_sampled_round2 --seed-offset 913
%R% --train-adaptive-dir data_v3/adaptive/5pct_sampled_train --test-adaptive-dir data_v3/adaptive/5pct_sampled_test --round2-adaptive-dir data_v3/adaptive/5pct_sampled_round2 --reuse-retrained

echo.
echo Done. Results: data_v3\adaptive\5pct_floor_train\retrained_v2_augment\retraining_v2_metrics.csv
echo               data_v3\adaptive\5pct_sampled_train\retrained_v2_augment\retraining_v2_metrics.csv
