import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data_pci_interval.csv")

X = df[["humidite", "carbone", "hydrogene", "cendres"]]
y_min = df["pci_min"]
y_max = df["pci_max"]

X_train, X_test, y_min_train, y_min_test = train_test_split(
    X, y_min, test_size=0.2, random_state=42
)
_, _, y_max_train, y_max_test = train_test_split(
    X, y_max, test_size=0.2, random_state=42
)


def make_model():
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            n_jobs=-1
        ))
    ])


model_min = make_model()
model_max = make_model()

model_min.fit(X_train, y_min_train)
model_max.fit(X_train, y_max_train)

pred_min = model_min.predict(X_test)
pred_max = model_max.predict(X_test)

# Important: garantir min <= max
pred_min2 = pred_min.copy()
pred_max2 = pred_max.copy()
swap = pred_min2 > pred_max2
pred_min2[swap], pred_max2[swap] = pred_max2[swap], pred_min2[swap]

mae_min = mean_absolute_error(y_min_test, pred_min2)
mae_max = mean_absolute_error(y_max_test, pred_max2)

print(f"MAE pci_min: {mae_min:.0f} kJ/kg")
print(f"MAE pci_max: {mae_max:.0f} kJ/kg")

joblib.dump(model_min, "pci_min_model.joblib")
joblib.dump(model_max, "pci_max_model.joblib")
print("Modèles sauvegardés: pci_min_model.joblib, pci_max_model.joblib")
