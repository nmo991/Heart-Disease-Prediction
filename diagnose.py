'''
import pandas as pd
import numpy as np

# Load your data
df = pd.read_csv('heart_features.csv')
print("Dataset shape:", df.shape)

# Identify target and drop ID columns
target = 'heart_disease'
cols_to_drop = ['Id', 'patient_id', target] + [c for c in df.columns if c.endswith('_y') or c.endswith('_x')]
cols_to_drop = [c for c in cols_to_drop if c in df.columns]

# Get feature columns
feature_cols = [c for c in df.columns if c not in cols_to_drop]
print(f"\nFeature columns ({len(feature_cols)}):")
print(feature_cols)

# Check each column's data type
print("\n=== Column Data Types ===")
for col in feature_cols:
    print(f"{col}: {df[col].dtype}")

# Find non-numeric columns
non_numeric = []
for col in feature_cols:
    try:
        pd.to_numeric(df[col], errors='raise')
    except:
        non_numeric.append(col)

print(f"\n=== Non-numeric columns ({len(non_numeric)}) ===")
for col in non_numeric:
    print(f"\n{col}:")
    print(df[col].head(10).tolist())
    print(f"Unique values: {df[col].unique()[:10]}")



import pandas as pd

# Load new conditions
conditions = pd.read_csv('synthea/output/csv/conditions.csv')
patients = pd.read_csv('synthea/output/csv/patients.csv')

# Check heart disease codes
heart_codes = ['I25', 'I21', 'I50', 'I20', 'I10']  # Added hypertension
conditions['CODE'] = conditions['CODE'].fillna('').astype(str)

matches = conditions[conditions['CODE'].str.startswith(tuple(heart_codes[:4]))]
print(f"Heart disease cases: {len(matches)}")
print(f"Unique patients: {matches['PATIENT'].nunique()}")
print(f"Total patients: {len(patients)}")
print(f"Prevalence: {matches['PATIENT'].nunique()/len(patients):.2%}")

print("\nTop heart codes:")
print(matches['CODE'].value_counts().head(10))\\


####### code checking ##############

import pandas as pd
import os

# Load conditions
csv_path = 'synthea/output/csv'
conditions = pd.read_csv(os.path.expanduser(f'{csv_path}/conditions.csv'))

print(f"Total conditions: {len(conditions)}")
print(f"Unique patients in conditions: {conditions['PATIENT'].nunique()}")

# Fix CODE column
conditions['CODE'] = conditions['CODE'].fillna('').astype(str)

# Look at ALL code patterns
print("\n=== All CODE prefixes (first 3 chars) ===")
prefixes = conditions['CODE'].str[:3].value_counts()
print(prefixes.head(20))

# Look for any cardiac-related codes
cardiac_keywords = ['I', 'heart', 'cardio', 'hyper', 'coronary']
print("\n=== Potential cardiac-related codes ===")
for keyword in cardiac_keywords:
    matches = conditions[conditions['CODE'].str.contains(keyword, case=False, na=False)]
    if len(matches) > 0:
        print(f"\n{keyword}: {len(matches)} matches")
        print(matches[['CODE', 'DESCRIPTION']].drop_duplicates().head(5))

        '''
