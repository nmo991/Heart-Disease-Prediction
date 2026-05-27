import pandas as pd
import numpy as np

print("Loading data...")
patients = pd.read_csv('labeled_patients.csv')
observations = pd.read_csv('synthea/output/csv/observations.csv')
conditions = pd.read_csv('synthea/output/csv/conditions.csv')
medications = pd.read_csv('synthea/output/csv/medications.csv')

# Fix observation codes
observations['DESCRIPTION'] = observations['DESCRIPTION'].fillna('').astype(str)
observations['VALUE'] = pd.to_numeric(observations['VALUE'], errors='coerce')

# Define features to extract
vital_features = {
    'blood pressure_systolic': ['systolic'],
    'blood pressure_diastolic': ['diastolic'],
    'heart rate': ['heart rate', 'pulse'],
    'bmi': ['bmi', 'body mass index'],
    'cholesterol_ldl': ['ldl'],
    'cholesterol_hdl': ['hdl'],
    'cholesterol_total': ['cholesterol total', 'total cholesterol'],
    'glucose': ['glucose', 'blood sugar'],
    'hba1c': ['hba1c', 'a1c']
}

print("\nExtracting features...")
feature_list = []

for patient_id in patients['Id']:
    patient_obs = observations[observations['PATIENT'] == patient_id]
    
    features = {'patient_id': patient_id}
    
    # Get latest value for each vital type
    for feature_name, keywords in vital_features.items():
        for keyword in keywords:
            matches = patient_obs[patient_obs['DESCRIPTION'].str.lower().str.contains(keyword, na=False)]
            if not matches.empty:
                # Take the most recent value
                latest = matches.sort_values('DATE', ascending=False).iloc[0]
                features[feature_name] = latest['VALUE']
                break
    
    feature_list.append(features)

# Create feature dataframe
feature_df = pd.DataFrame(feature_list)
print(f"Features extracted for {len(feature_df)} patients")

# Add condition count
condition_counts = conditions.groupby('PATIENT').size().reset_index(name='condition_count')
feature_df = feature_df.merge(condition_counts, left_on='patient_id', right_on='PATIENT', how='left')

# Add medication count
med_counts = medications.groupby('PATIENT').size().reset_index(name='medication_count')
feature_df = feature_df.merge(med_counts, left_on='patient_id', right_on='PATIENT', how='left')

# *** CRITICAL FIX: Convert all possible columns to numeric ***
for col in feature_df.columns:
    if col != 'patient_id':  # Skip ID column
        feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')

# Fill missing values with median of numeric columns only
numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
feature_df[numeric_cols] = feature_df[numeric_cols].fillna(feature_df[numeric_cols].median())

# Merge with labels (patients dataframe)
final_df = patients.merge(feature_df, left_on='Id', right_on='patient_id', how='inner')

# *** FINAL CHECK: Ensure all features are numeric ***
feature_cols = [c for c in final_df.columns if c not in ['Id', 'patient_id', 'heart_disease']]
for col in feature_cols:
    final_df[col] = pd.to_numeric(final_df[col], errors='coerce')

# Fill any remaining NaNs
final_df[feature_cols] = final_df[feature_cols].fillna(final_df[feature_cols].median())

# Save
final_df.to_csv('heart_features.csv', index=False)
print(f"\nFinal dataset shape: {final_df.shape}")
print(f"Final columns: {final_df.columns.tolist()}")
print(f"\nData types:\n{final_df.dtypes}")