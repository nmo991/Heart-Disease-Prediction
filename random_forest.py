import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load ML-ready dataset
df = pd.read_csv('heart_features.csv')
print("Dataset shape:", df.shape)

# Define target and columns to exclude
target = 'heart_disease'
id_cols = ['Id', 'patient_id', 'PATIENT_x', 'PATIENT_y']

# Select numeric features only
feature_cols = [c for c in df.columns if c not in id_cols + [target] and df[c].dtype in ['int64', 'float64']]
print(f"\nUsing {len(feature_cols)} numeric features:")
print(feature_cols)

# Features and target
X = df[feature_cols]
y = df[target]

print(f"Features shape: {X.shape}")
print(f"Positive cases: {y.sum()}/{len(y)} ({y.mean():.2%})")
print(f"Any NaN in features: {X.isna().any().any()}")

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Random Forest classifier
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    class_weight='balanced'  # handles imbalance
)

# Train
model.fit(X_train, y_train)

# Predictions
y_proba = model.predict_proba(X_test)[:, 1]  # probability of heart disease
y_pred = model.predict(X_test)               # 0 or 1 predictions

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_proba)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))
print(f"AUC-ROC: {auc_score:.3f}")
print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-Score:  {f1:.3f}")

# Save model
joblib.dump({
    'model': model,
    'features': feature_cols
}, 'heart_model_rf.pkl')

print("\nSaved heart_model_rf.pkl")