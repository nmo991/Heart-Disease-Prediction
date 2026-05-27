import pandas as pd
import os

# Load CSVs
csv_path = 'synthea/output/csv/'
patients = pd.read_csv(f'{csv_path}/patients.csv')
conditions = pd.read_csv(f'{csv_path}/conditions.csv')
observations = pd.read_csv(f'{csv_path}/observations.csv')
encounters = pd.read_csv(f'{csv_path}/encounters.csv')
medications = pd.read_csv(f'{csv_path}/medications.csv')

print("=== Dataset Sizes ===")
print(f"Patients: {len(patients)}")
print(f"Conditions: {len(conditions)}")
print(f"Observations: {len(observations)}")

conditions['CODE'] = conditions['CODE'].astype(str)

print("\n=== Unique CODE types (first 20) ===")

print(conditions['CODE'].unique()[:20])
print("\n=== Heart Disease Codes ===")
heart_codes = conditions[conditions['CODE'].str.startswith(('I25','I21','I50','I20'))]
print(heart_codes[['CODE','DESCRIPTION']].drop_duplicates())

print("\n=== Top Conditions ===")
print(conditions['DESCRIPTION'].value_counts().head(10))