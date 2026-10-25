# HealthConnect Rwanda: Healthcare Predictive Analytics & Population Health Intelligence
AnalystLab Africa Experience Lab, Week 6, Data Science track. Author: Emmanuella Esi Arhin.

## Repository structure
```
notebooks/      HealthConnect_DataScience_Notebook.ipynb (executed)
data/raw/       27 source CSV files (inputs)
data/processed/ cleaned and integrated datasets, data_quality_report.csv, results.json
reports/        1_Data_Quality_Report, 2_EDA_Report, 3_Statistical_Analysis_Report,
                4_Predictive_Analytics_Report, Executive_Summary (.docx)
presentation/   HealthConnect_Presentation.pptx (13 slides)
figures/        21 visualizations (title, axis labels, interpretation on each)
```

## Key outputs
- `data/processed/HC_Analytical_Dataset.csv`: cleaned, integrated dataset (5,000 appointments x 57 columns incl. engineered features)
- `data/processed/data_quality_report.csv`: all 42 data-quality findings

## Reproduce
```
cd notebooks
jupyter nbconvert --to notebook --execute --inplace HealthConnect_DataScience_Notebook.ipynb
```
Paths are relative to `notebooks/`. To read the CSVs elsewhere, set the `HC_DATA_DIR` environment variable.

## Key caveats
Admissions, labs, prescriptions, referrals and the facility master were not supplied; only 8.3% of billing/diagnosis/payment rows link to an appointment; 12 months of data; the data appear simulated. See the limitations section of report 4.
