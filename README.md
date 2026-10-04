Industrial Machine Failure Prediction

Predicts whether an industrial machine will fail from its sensor readings (predictive maintenance), using a PyTorch neural network compared against classical models.

Problem

Only about 3.4% of the records in the data are failures, so accuracy is misleading. The project is judged by PR-AUC, recall and precision on the failure class.

Dataset

AI4I 2020 Predictive Maintenance Dataset (S. Matzka, UCI Machine Learning Repository): 10,000 records, 339 failures. Features: product type, air and process temperature, rotational speed, torque, tool wear. The failure-mode columns (TWF, HDF, PWF, OSF, RNF) are removed because they are derived from the target (data leakage).

Approach
Added two features: Thermal_Strain (process minus air temperature) and Power (torque x rotational speed)
Stratified 60 / 20 / 20 train, validation and test split; preprocessing fitted on the training set only
PyTorch ANN (10 -> 64 -> 32 -> 1) with class-weighted loss and early stopping
Compared with Logistic Regression and Random Forest
Decision threshold tuned on the validation set
Results (test set: 2,000 samples, 68 failures)
Model	Test PR-AUC	Test ROC-AUC
Logistic Regression	0.471	0.930
Random Forest	0.853	0.958
ANN (PyTorch)	0.701	0.970
ANN threshold	Precision	Recall	F1	Missed failures	False alarms
0.5 (default)	0.278	0.853	0.419	10	151
0.87 (tuned)	0.542	0.765	0.634	16	44



Key findings

Random Forest performed best (PR-AUC 0.853 vs 0.701 for the ANN), also in 5-fold cross-validation.
Tuning the threshold cut false alarms from 151 to 44 and raised F1 from 0.42 to 0.63, at the cost of 6 more missed failures.
The engineered features improved the ANN's mean PR-AUC from 0.713 to 0.756 (3 seeds).
How to Run
bash
pip install -r requirements.txt
jupyter notebook machine-failure-prediction.ipynb

Run all cells in order. ai4i2020.csv must be in the same folder as the notebook.

Tech Stack

Python, pandas, scikit-learn, PyTorch, Matplotlib, Seaborn

Limitations

The dataset is synthetic and has few failures, so the results are an estimate for this dataset only. Next step: tune Random Forest / XGBoost and deploy a Streamlit app.
