# HealthConnect Rwanda: Healthcare Predictive Analytics & Population Health Intelligence
AnalystLab Africa Experience Lab, Data Science track. Author: Emmanuella Esi Arhin.

| Week | Focus | Status |
|---|---|---|
| 6 | Data preparation, EDA, statistics, forecasting, predictive models, clustering | Complete |
| 7 | Model testing, error analysis, refinement and cross-track validation | Complete (see [Week 7](#week-7-model-testing-error-analysis-and-refinement)) |

## Repository structure
```
notebooks/      HealthConnect_DataScience_Notebook.ipynb (Week 6, executed)
data/raw/       27 source CSV files (inputs, never modified)
data/processed/ cleaned and integrated datasets, data_quality_report.csv, results.json
reports/        1_Data_Quality_Report, 2_EDA_Report, 3_Statistical_Analysis_Report,
                4_Predictive_Analytics_Report, Executive_Summary (.docx)
presentation/   HealthConnect_Presentation.pptx (13 slides)
figures/        21 visualizations (title, axis labels, interpretation on each)
week7/          Week 7 work (see below)
  HealthConnect_Week7_Model_Testing_Notebook.ipynb   (executed)
  reports/      Week7_Model_Testing_and_Refinement_Report.pdf, Week7_Project_Summary.pdf
  evidence/     Week7_Testing_Validation_Record.xlsx, screenshots of cross-track messages
  handoff_ml_engineering/   tested model interface for ML Engineering (see below)
```

## Week 6 key outputs
- `data/processed/HC_Analytical_Dataset.csv`: cleaned, integrated dataset (5,000 appointments x 57 columns incl. engineered features)
- `data/processed/data_quality_report.csv`: all 42 data-quality findings

## Reproduce Week 6
```
cd notebooks
jupyter nbconvert --to notebook --execute --inplace HealthConnect_DataScience_Notebook.ipynb
```
Paths are relative to `notebooks/`. To read the CSVs elsewhere, set the `HC_DATA_DIR` environment variable.

---

## Week 7: Model testing, error analysis and refinement
Week 7 does not rebuild Week 6. It takes the Week 6 candidate (the admission-proxy classifier) through a structured test programme: **test, find issues, refine, re-test, validate, document**. Acceptance criteria (AC1-AC10) were fixed before any test was run, and a 20% patient-grouped test set was locked and used once per model version.

### Headline results
| Question | Result |
|---|---|
| Does the Week 6 result reproduce? | Yes: AUC 0.505 and the same confusion matrix |
| Does it work on unseen data? | No: locked hold-out AUC 0.479 (95% CI 0.423-0.535); permutation test p = 0.42; trained Jan-Aug and tested Sep-Dec AUC 0.467 |
| Over-fitting? | Severe: train AUC 0.75 vs validation 0.51 |
| Can the test method detect signal at all? | Yes: positive controls (Age 65+, rule-based high-risk flag) are recovered, so the null result is a property of the data |
| What did refinement change? | Removed unreliable features (FacilityID, DepartmentName, BMI, department workload index), regularised model, top-20% flag-rate budget instead of an F1 threshold that flagged 95% of appointments, executable release gate |
| After refinement | Over-fitting gap 0.24 to 0.06; hold-out AUC 0.455 (CI 0.398-0.513); permutation p = 0.73. Still no signal |
| Is the model suitable for the HealthConnect use case? | **No. Release gate FAILS** on the mandatory criteria. It is not for clinical or operational decisions |
| Model interface | 15 contract tests: 10/15 passed on the Week 6 pipeline, 15/15 on the refined scorer |

### Week 7 deliverables
| File | Purpose |
|---|---|
| `week7/HealthConnect_Week7_Model_Testing_Notebook.ipynb` | Executed notebook: tests T1-T12, error analysis, refinement, release gate, cross-track tests |
| `week7/reports/Week7_Model_Testing_and_Refinement_Report.pdf` | Main report: testing approach, results, before/after, cross-track record, issue and decision logs |
| `week7/reports/Week7_Project_Summary.pdf` | Concise 16-point project summary |
| `week7/evidence/Week7_Testing_Validation_Record.xlsx` | Testing record, acceptance criteria, gate results, contract tests, issue log, decision log, Week 8 actions |
| `week7/handoff_ml_engineering/` | Model, schema, scoring wrapper and 15 interface tests |

### Reproduce Week 7
Week 7 reads files written by the Week 6 notebook (`data/processed/HC_Analytical_Dataset.csv` and `data/processed/HC_Department_Summary.csv`) and the raw CSVs in `data/raw/`. Run Week 6 first if those files are missing.
```
cd week7
export HC_PROCESSED_DIR=../data/processed
export HC_DATA_DIR=../data/raw
jupyter nbconvert --to notebook --execute --inplace HealthConnect_Week7_Model_Testing_Notebook.ipynb
```
Takes about 7 minutes (the label-permutation tests are the slow step). All randomness is seeded (seed 42). The notebook also writes `figures_w7/` and `outputs_w7/` one level above `week7/`; these are regenerated each run and do not need to be committed.
Requires: Python 3.12, pandas, numpy, scipy, scikit-learn (1.8.0 used), statsmodels, matplotlib, seaborn.

### ML hand-off package (`week7/handoff_ml_engineering/`)
```
cd week7/handoff_ml_engineering
pip install -r requirements.txt
python hc_contract_tests.py
```
| File | Purpose |
|---|---|
| `model.joblib` | Refined scikit-learn pipeline |
| `schema.json` | Required columns, types, ranges, allowed categories, top-20% threshold, library versions |
| `hc_scorer.py` | `AdmissionProxyScorer(folder).score(df)` returns `risk_score`, `flag_top20`, `status` and `model_status` per row |
| `hc_contract_tests.py` | 15 interface tests (also usable as `run_contract_tests(score_fn, sample_df)` for any pipeline) |
| `sample_input.csv`, `requirements.txt`, `README_HANDOFF.md` | Example input, pinned versions, instructions |

**The model status is EXPERIMENTAL: it has no predictive signal on the current data.** What is handed over is a tested interface and a reusable test harness, not a usable predictor.

### Cross-track testing
- **Data Analytics (Olabisi Amusan), two-way validation:** her independent findings agree with mine on coverage, the 4,744 facility conflict, the missing attendance field, weak signal, balanced classes, rare repeat visits and unusable fields. She also ran three checks I requested against the source files: 88.0% of Follow-Up visits have no earlier visit (confirmed), DoctorYears is a clean join but its department/facility link is about random (kept as a doctor-level attribute only), and the target rate is 10.88% versus 10.71% expected if its two parts were independent. No model change followed, so before/after metrics are unchanged.
- **ML Engineering:** the interface tests were completed on the Data Science side only; no ML Engineering counterpart was available, so this is not claimed as a collaboration. The package is prepared for Week 8.

### Week 7 caveats
- No Admissions, Referrals, Laboratory or attendance/no-show table exists in the supplied data, so admission is modelled through a proxy (Emergency visit with a Severe diagnosis) and attendance questions cannot be tested on this dataset.
- Only 5,000 appointments (544 positives; 108 in the locked test set) are available, so intervals are wide.
- The Week 5 baseline model was not among the supplied files; the prevalence baseline and the Week 6 logistic regression are used as reference points.
- The data behave like simulated data, so no result should be read as a real-world clinical finding.

### Week 8 priorities
Obtain outcome tables (admissions, referrals, attendance); repair appointment keys; agree the prediction time-point (Diagnosis is only known after consultation); have ML Engineering run the contract tests when a counterpart is assigned; present the work as a validated null result plus a reusable test harness.
