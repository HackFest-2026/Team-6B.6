# ReAdmitAI: Keep patients home

A single-page demo that estimates each discharged patient's chance of 30-day hospital readmission, shown as a percentage, and helps a care team prioritise follow-up calls.

## Data
[Diabetes 130-US hospitals, 1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) (UCI).

Cohort: first encounter per patient, excluding deaths and hospice discharges and 3 unknown-gender records. 69,987 patients; 8.98% readmitted within 30 days.

## Model
Logistic regression (also random forest and gradient boosting) on 8 inputs: age, length of stay, past inpatient stays, number of diagnoses, medications, emergency admission, past ER visits, discharged home. Probabilities are Platt-calibrated.

## Verified results (held-out test patients)
| Cutoff | Flagged | Precision | Recall |
|---|---|---|---|
| Standard (50%) | 1 | 0% | 0% |
| High confidence (25%) | 138 | 28.3% | 3.1% |
| Top 20% (11.8%) | 2,791 | 14.0% | 31.5% |

Risk bands: Low below 7%, Medium 7-14%, High 14% and above. Predicted and observed rates agree within about 1-2 points (High band slightly overconfident).

## Limitations
- AUC is about 0.62: reliable on average, weak for individual patients.
- Data is from 1999-2008.
- Daily check-in adjustments and diagnosis offsets are hand-set and not validated.
- Demo only, not for clinical use.

## Repository contents
- `index.html`: the full app (open in a browser, no build step)
- `data/`: `diabetic_data.csv` and `IDS_mapping.csv` from UCI (CC BY 4.0, Strack et al., 2014)
- `scripts/verify_model.py`: rebuilds the cohort and model from the CSV and checks the app's numbers
- `docs/verification_output.txt`: output of that script
- `LICENSE`: MIT for the code (the dataset keeps its own CC BY 4.0 license)

## Verify the numbers
```
pip install -r requirements.txt
python scripts/verify_model.py
```
The script uses its own random split, so AUC and band counts differ slightly from the app (about 0.628 vs 0.621) while the cohort size (69,987) and 8.98% rate match exactly.

## Run the app
Open `index.html` in a browser.
