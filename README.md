Cattle Antibody Pairing Predictor
Description
Cattle Antibody Pairing Predictor: A Streamlit app to predict heavy/light chain pairing in cattle antibodies using Supervised Machine Learning approach. 
Upload CSV with sequences, choose combinatory analysis, select Logistic Regression or Random Forest, and view predictions with SHAP plots. 
Download results as CSV. Keep pairs <1,000 for efficiency.
Installation

Clone the repository:git clone https://github.com/Donmaston09/CAPP-App/.git


Install dependencies:pip install -r requirements.txt


Ensure model files (best_Logistic Regression_model.pkl, best_Random Forest_model.pkl) are in the project directory.

Usage

Run the app:streamlit run cattle_antibody_prediction_app.py


Upload a CSV with AA_H, CDR3_H, AA_L, CDR3_L columns.
Select combinatory analysis and model.
Click "Run" to view predictions and visualizations.
Download results.

Requirements

Python 3.8+
Libraries: streamlit, pandas, matplotlib, numpy, seaborn, scikit-learn, biopython, shap
Model files: best_Logistic Regression_model.pkl, best_Random Forest_model.pkl

Sample Data Format
AA_H,CDR3_H,AA_L,CDR3_L
QVESDREAQWKELKELSXSXSAQASEKDJKDJDLDJ,CDEFGHIKLMNPQRSTVWY,QWKEKEKDKLDKWJDGFRHSJWGS,CDEFGHIKLMNPQRSTVWY

Authors:
Anthony Onoja, Nophar Geifman, Marie Di Placido, Bharti Mittal, John Hammond, Nicos Angelopoulos
License
MIT License
