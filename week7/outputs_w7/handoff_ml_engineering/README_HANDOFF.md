# HealthConnect admission-proxy scorer: hand-off to ML Engineering (Week 7)

**Status: EXPERIMENTAL. Release gate FAILED (mandatory criteria ['AC2', 'AC3', 'AC5', 'AC9']). The model has no predictive signal on the current data.**
What is being handed over is a *tested interface*, not a usable predictor.

## Files
| File | Purpose |
|---|---|
| model.joblib | Refined scikit-learn pipeline (D. Ridge logistic, 5 core features) |
| schema.json | Required columns, types, ranges, allowed categories, top-20% threshold, library versions |
| hc_scorer.py | AdmissionProxyScorer(folder).score(df) returns risk_score, flag_top20, status, model_status per row |
| hc_contract_tests.py | 15 interface tests. Library use: run_contract_tests(score_fn, sample_df); script use: python hc_contract_tests.py |
| sample_input.csv | 20 example rows (no identifiers) |
| requirements.txt | Versions the model was trained with (scikit-learn 1.9.0) |

## Contract
Rows with non-numeric or out-of-range values are rejected with a reason and the rest of the batch is scored. A missing required column raises SchemaError. Soft problems (missing or unseen category, missing number) are scored and visible in status. Text is matched case-insensitively. Same input gives the same output.

## Requested from ML Engineering (please complete and return)
1. Run python hc_contract_tests.py in your environment. Result: ___/15 passed. Python / scikit-learn versions: ________
2. Run run_contract_tests(your_score_fn, sample_df) with your own preprocessing in front of the model. Result: ___/15. Failing IDs: ________
3. Does your pipeline send the columns and types in schema.json? Differences: ________
4. Prediction time-point you need (booking vs after consultation; Diagnosis is only known after consultation): ________
5. Anything the Data Science side must change: ________
