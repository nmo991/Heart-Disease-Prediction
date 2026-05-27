import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('heart_features.csv')
print("Dataset shape:", df.shape)

# Define columns to exclude
target = 'heart_disease'
id_cols = ['Id', 'patient_id', 'PATIENT_x', 'PATIENT_y']  # Add any other ID columns

# Get feature columns (numeric only)
feature_cols = [c for c in df.columns if c not in id_cols + [target] and df[c].dtype in ['int64', 'float64']]

print(f"\nUsing {len(feature_cols)} numeric features:")
print(feature_cols)

# Prepare data
X = df[feature_cols]
y = df[target]

print(f"Features shape: {X.shape}")
print(f"Positive cases: {y.sum()}/{len(y)} ({y.mean():.2%})")
print(f"Any NaN in features: {X.isna().any().any()}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

model.fit(X_train_scaled, y_train)


# Predict
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# Evaluate
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print(f"\nAUC-ROC: {roc_auc_score(y_test, y_proba):.3f}")

# Feature importance
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== Top Features ===")
print(importance.head(10))

# Save model
import joblib
joblib.dump({
    'model': model,
    'scaler': scaler,
    'features': feature_cols
}, 'heart_model.pkl')
print("\nSaved heart_model.pkl")