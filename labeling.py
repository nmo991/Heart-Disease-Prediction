'''
import pandas as pd
import numpy as np

# Load
patients = pd.read_csv('synthea/output/csv/patients.csv')
conditions = pd.read_csv('synthea/output/csv/conditions.csv')

conditions['CODE'] = conditions['CODE'].astype(str)

# Create heart disease label
heart_codes = ['I25', 'I21', 'I50', 'I20']
heart_patients = conditions[conditions['CODE'].str.startswith(tuple(heart_codes))]['PATIENT'].unique()
patients['heart_disease'] = patients['Id'].isin(heart_patients).astype(int)

print(f"Heart disease prevalence: {patients['heart_disease'].mean():.2%}")

# Add age
patients['age'] = (pd.to_datetime('2024-01-01') - pd.to_datetime(patients['BIRTHDATE'])).dt.days / 365.25
patients['is_male'] = (patients['GENDER'] == 'M').astype(int)

# Save base labeled data
patients[['Id', 'age', 'is_male', 'heart_disease']].to_csv('labeled_patients.csv', index=False)
print("Saved labeled_patients.csv")
'''
import pandas as pd
import numpy as np

# Load data
patients = pd.read_csv('synthea/output/csv/patients.csv')
conditions = pd.read_csv('synthea/output/csv/conditions.csv')

# Convert CODE to string
conditions['CODE'] = conditions['CODE'].fillna('').astype(str)

# SNOMED codes for heart disease (based on what we found)
heart_codes_snomed = [
    '414545008',  # Ischemic heart disease
    '88805009',   # Chronic congestive heart failure
    '274531002',  # Abnormal findings diagnostic imaging heart
    '399261000',  # History of coronary artery bypass grafting
    '22298006',   # Myocardial infarction
    '53741008',   # Coronary heart disease
    '84114007',   # Heart failure
    '38341003',   # Hypertension
    '194828000',  # Angina pectoris
    '49436004',   # Atrial fibrillation
]

# Also search by description keywords as backup
keywords = ['heart', 'cardiac', 'coronary', 'myocardial', 'ischemic', 'failure', 'bypass']

# Method 1: Match by exact SNOMED codes
code_mask = conditions['CODE'].isin(heart_codes_snomed)

# Method 2: Match by description keywords
desc_mask = conditions['DESCRIPTION'].str.contains('|'.join(keywords), case=False, na=False)

# Combine both methods
heart_mask = code_mask | desc_mask
heart_conditions = conditions[heart_mask]

# Get unique patients
heart_patients = heart_conditions['PATIENT'].unique()

print(f"Total patients: {len(patients)}")
print(f"Heart disease condition records: {len(heart_conditions)}")
print(f"Unique patients with heart disease: {len(heart_patients)}")
print(f"Prevalence: {len(heart_patients)/len(patients):.2%}")

print("\nTop heart-related conditions:")
print(heart_conditions['DESCRIPTION'].value_counts().head(10))

# Create label
patients['heart_disease'] = patients['Id'].isin(heart_patients).astype(int)

# Calculate age
patients['age'] = (pd.to_datetime('2024-01-01') - pd.to_datetime(patients['BIRTHDATE'])).dt.days / 365.25

# Gender encoding
patients['is_male'] = (patients['GENDER'] == 'M').astype(int)

# Save base labeled data
base_df = patients[['Id', 'age', 'is_male', 'heart_disease']].copy()
base_df.to_csv('labeled_patients.csv', index=False)
print(f"\nSaved labeled_patients.csv with {base_df['heart_disease'].sum()} positive cases")