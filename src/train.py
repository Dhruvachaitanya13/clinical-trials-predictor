"""
Day 2: train a baseline (logistic regression) and main model (XGBoost),
evaluate honestly, and save the trained model for the API.
Run after data/trials_features.csv exists (from the Day 1 notebook).
"""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb

df = pd.read_csv("data/trials_features.csv")
df["label"] = (df["overall_status"].str.upper() == "COMPLETED").astype(int)

num = ["enrollment", "number_of_arms", "start_year", "duration_months",
       "n_conditions", "n_interventions"]
cat = ["phase", "sponsor_class", "allocation", "intervention_model",
       "masking", "primary_purpose"]

X = df[num + cat]
y = df["label"]
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
])

# Baseline: logistic regression
logit = Pipeline([("pre", pre),
                  ("clf", LogisticRegression(max_iter=1000))]).fit(Xtr, ytr)
print("Logit AUC:", roc_auc_score(yte, logit.predict_proba(Xte)[:, 1]))

# Main model: XGBoost
model = Pipeline([("pre", pre),
    ("clf", xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="auc"))]).fit(Xtr, ytr)

proba = model.predict_proba(Xte)[:, 1]
print("XGB AUC:", roc_auc_score(yte, proba))
print(classification_report(yte, (proba > 0.5).astype(int)))

joblib.dump(model, "src/model.joblib")
print("Saved model to src/model.joblib")
