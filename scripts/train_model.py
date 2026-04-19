import json
import numpy as np
from app.feature_engineering import extract_features
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# Load data
with open("data/processed_data.json") as f:
    data = json.load(f)

X = []
y = []

# Generate labels using OLD rules (bootstrapping)
from app.rules import calculate_risk

for customer in data:
    features = extract_features(customer)
    label = calculate_risk(customer)

    X.append(features)

    # Convert labels
    if label == "LOW":
        y.append(0)
    elif label == "MEDIUM":
        y.append(1)
    else:
        y.append(2)

X = np.array(X)
y = np.array(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob, multi_class="ovr"))

# Save model
joblib.dump(model, "model/churn_model.pkl")

print("✅ Model trained and saved!")