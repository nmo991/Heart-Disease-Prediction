import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import joblib

# Load model and data
print("Loading model and data...")
model_data = joblib.load('heart_model.pkl')
model = model_data['model']
scaler = model_data['scaler']
features = model_data['features']  # List of features used in training

print(f"Model expects these {len(features)} features:")
print(features)

df = pd.read_csv('heart_features.csv')
print(f"\nData has {len(df.columns)} columns")
print("First few columns:", df.columns[:10].tolist())

# IMPORTANT: Select ONLY the features the model knows
# This excludes PATIENT_x, PATIENT_y, patient_id, etc.
X = df[features]
y = df['heart_disease']

print(f"\nX shape: {X.shape}")
print(f"Any missing values: {X.isna().any().any()}")

# Scale and predict
X_scaled = scaler.transform(X)
y_pred = model.predict(X_scaled)
y_proba = model.predict_proba(X_scaled)[:, 1]

# Add predictions to dataframe
df['predicted'] = y_pred
df['probability'] = y_proba

print(f"\n=== RESULTS SUMMARY ===")
print(f"Total patients: {len(df)}")
print(f"Actual heart disease: {y.sum()} ({y.mean():.1%})")
print(f"Predicted heart disease: {y_pred.sum()} ({y_pred.mean():.1%})")

# Calculate metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
auc_score = roc_auc_score(y, y_proba)

print(f"\n=== PERFORMANCE METRICS ===")
print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-Score:  {f1:.3f}")
print(f"AUC-ROC:   {auc_score:.3f}")

# Create visualizations
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(16, 10))

# 1. Confusion Matrix
plt.subplot(2, 3, 1)
cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Healthy', 'Heart Disease'],
            yticklabels=['Healthy', 'Heart Disease'])
plt.title('Confusion Matrix', fontsize=14)
plt.ylabel('Actual')
plt.xlabel('Predicted')

# 2. ROC Curve
plt.subplot(2, 3, 2)
fpr, tpr, _ = roc_curve(y, y_proba)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

# 3. Precision-Recall Curve
plt.subplot(2, 3, 3)
precision_vals, recall_vals, _ = precision_recall_curve(y, y_proba)
plt.plot(recall_vals, precision_vals, color='green', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True, alpha=0.3)
plt.xlim([0, 1])
plt.ylim([0, 1])

# 4. Feature Importance
plt.subplot(2, 3, 4)
importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=True)

plt.barh(importance['feature'], importance['importance'], color='steelblue')
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.tight_layout()

# 5. Prediction Distribution
plt.subplot(2, 3, 5)
plt.hist([y_proba[y==0], y_proba[y==1]], 
         label=['Healthy', 'Heart Disease'],
         bins=20, alpha=0.7, color=['blue', 'red'])
plt.xlabel('Predicted Probability')
plt.ylabel('Count')
plt.title('Prediction Distribution by Actual Class')
plt.legend()
plt.grid(True, alpha=0.3)

# 6. Age Distribution by Prediction
plt.subplot(2, 3, 6)
df_correct = df[df['heart_disease'] == df['predicted']]
df_wrong = df[df['heart_disease'] != df['predicted']]

plt.hist([df_correct['age'], df_wrong['age']], 
         label=['Correct Predictions', 'Wrong Predictions'],
         bins=20, alpha=0.7, color=['green', 'red'])
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution: Correct vs Wrong')
plt.legend()
plt.grid(True, alpha=0.3)

plt.suptitle('Heart Disease Prediction Model Results', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('model_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Saved model_results.png")

# Save detailed results to CSV
results_df = df[['Id', 'age', 'is_male', 'heart_disease', 'predicted', 'probability']].copy()
results_df.to_csv('prediction_results.csv', index=False)
print("✓ Saved prediction_results.csv")

# Generate text report
with open('model_report.txt', 'w') as f:
    f.write("HEART DISEASE PREDICTION MODEL REPORT\n")
    f.write("="*50 + "\n\n")
    f.write(f"Dataset Size: {len(df)} patients\n")
    f.write(f"Heart Disease Prevalence: {y.mean():.1%}\n\n")
    f.write("MODEL PERFORMANCE:\n")
    f.write(f"- Accuracy:  {accuracy:.3f}\n")
    f.write(f"- Precision: {precision:.3f}\n")
    f.write(f"- Recall:    {recall:.3f}\n")
    f.write(f"- F1-Score:  {f1:.3f}\n")
    f.write(f"- AUC-ROC:   {auc_score:.3f}\n\n")
    f.write("TOP 5 FEATURES:\n")
    top_features = importance.sort_values('importance', ascending=False).head(5)
    for _, row in top_features.iterrows():
        f.write(f"- {row['feature']}: {row['importance']:.3f}\n")

print("✓ Saved model_report.txt")
print("\nAll done! Check the generated files:")
print("- model_results.png (visualizations)")
print("- prediction_results.csv (individual predictions)")
print("- model_report.txt (summary statistics)")