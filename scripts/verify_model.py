"""Rebuilds the ReAdmitAI cohort and model from data/diabetic_data.csv and checks the app's numbers."""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix

df = pd.read_csv(Path(__file__).parent.parent / "data" / "diabetic_data.csv", na_values="?", low_memory=False)
df["y"] = (df.readmitted == "<30").astype(int)
a = df[~df.discharge_disposition_id.isin([11, 13, 14, 19, 20, 21])]      # exclude deaths / hospice
a = a.sort_values("encounter_id").drop_duplicates("patient_nbr")          # first encounter per patient
a = a[a.gender != "Unknown/Invalid"]
age = {f"[{i*10}-{i*10+10})": i*10+5 for i in range(10)}
X = pd.DataFrame({
    "age": a.age.map(age), "los": a.time_in_hospital.clip(upper=30),
    "prior": np.log1p(a.number_inpatient), "diagnoses": a.number_diagnoses,
    "meds": a.num_medications, "emergency": a.admission_type_id.isin([1, 2]).astype(int),
    "er_visits": np.log1p(a.number_emergency), "home": (a.discharge_disposition_id == 1).astype(int)}).values
y = a.y.values
print(f"patients={len(a)}  readmission rate={y.mean():.4f}")

idx = np.random.RandomState(0).permutation(len(a)); n = len(a)
tr, te = idx[:int(.65*n)], idx[int(.8*n):]
mu, sd = X[tr].mean(0), X[tr].std(0)
p = LogisticRegression(max_iter=1000).fit((X[tr]-mu)/sd, y[tr]).predict_proba((X[te]-mu)/sd)[:, 1]
print(f"test AUC={roc_auc_score(y[te], p):.3f}")

for lab, lo, hi in [("Low", 0, .07), ("Medium", .07, .14), ("High", .14, 1)]:
    k = (p >= lo) & (p < hi)
    print(f"{lab:7s} n={k.sum():5d} predicted={p[k].mean():.4f} observed={y[te][k].mean():.4f}")

for name, cut in [("Standard 50%", .5), ("High confidence 25%", .25), ("Top 20%", np.quantile(p, .8))]:
    f = p >= cut
    tn, fp, fn, tp = confusion_matrix(y[te], f, labels=[0, 1]).ravel()
    print(f"{name:20s} cut={cut:.3f} flagged={tp+fp:5d} precision={tp/max(tp+fp,1):.3f} recall={tp/(tp+fn):.3f}")
