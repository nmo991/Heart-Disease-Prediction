# Run all steps
python 1_explore.py
python 2_labeling.py
python 3_features.py
python 4_model.py
python 5_report.py

# Check outputs
ls -la *.csv *.pkl *.png *.md

# Launch results notebook (optional)
jupyter notebook exploration.ipynb

    Expected Outputs:
labeled_patients.csv – Patients with heart disease labels

heart_features.csv – Feature matrix

heart_model.pkl – Trained model + scaler

shap_summary.png – Model explainability

model_curves.png – ROC & PR curves

feature_importance.png – Top features

RESULTS.md – Summary report

