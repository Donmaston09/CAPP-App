import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.Align import PairwiseAligner
from Bio.Align.substitution_matrices import load
import shap
import pickle
import os

# Load BLOSUM62 matrix
blosum62 = load("BLOSUM62")

# Initialize the pairwise aligner
aligner = PairwiseAligner()
aligner.substitution_matrix = blosum62

# Function to calculate structural compatibility
def calculate_structural_compatibility(row):
    cdr3_heavy = row['CDR3_H']
    cdr3_light = row['CDR3_L']
    alignments = aligner.align(cdr3_heavy, cdr3_light)
    best_alignment = alignments[0]
    match_count = sum(aa1 == aa2 for aa1, aa2 in zip(cdr3_heavy, cdr3_light))
    compatibility_score = match_count / len(cdr3_heavy)
    return compatibility_score

# Define amino acid charges
amino_acid_charges = {
    "R": 1, "H": 1, "K": 1, "D": -1, "E": -1, "Y": -1, "C": 0, "P": 0, "S": 0,
    "T": 0, "Q": 0, "N": 0, "A": 0, "G": 0, "M": 0, "I": 0, "L": 0, "V": 0, "F": 0, "W": 0
}

def calculate_net_charge(sequence, amino_acid_charges):
    """Calculates the net charge of a protein sequence."""
    return sum(amino_acid_charges.get(aa, 0) for aa in sequence)

# Feature name mapping
feature_name_mapping = {
    'CDR3_H_PI': 'CDRH3 Isoelectric point', 'CDR3_H_gravy': 'GRAVY CDRH3',
    'CDR3_H_Len': 'CDRH3 Length', 'CDR3_L_gravy': 'GRAVY CDRL3',
    'CDR3_L_PI': 'CDRL3 Isoelectric point', 'CDR3_H_cys_count': 'CDRH3 Cysteine count',
    'Structural_Compatibility': 'Structural Compatibility', 'CDR3_H_instability': 'CDRH3 Instability index',
    'CDR3_L_instability': 'CDRL3 Instability index', 'CDR3_L_cys_count': 'CDRL3 Cysteine count',
    'CDR3_L_Len': 'CDRL3 Length', 'CDR3_H_Net_Charge': 'CDRH3 Net Charge',
    'CDR3_L_Net_Charge': 'CDRL3 Net Charge'
}

# Function to plot feature importance
def plot_feature_importance(model, feature_names):
    plt.figure(figsize=(10, 6))
    mapped_feature_names = [feature_name_mapping.get(f, f) for f in feature_names]

    if hasattr(model, 'coef_'):
        coefficients = model.coef_[0]
        feature_importance_df = pd.DataFrame({
            'Feature': mapped_feature_names, 'Coefficient': coefficients
        })
        feature_importance_df['Importance'] = feature_importance_df['Coefficient'].abs()
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, color='gray')
        plt.title('Feature Importance for Logistic Regression', fontsize=22)
    elif hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        sorted_idx = np.argsort(importance)[::-1]
        sns.barplot(x=importance[sorted_idx], y=np.array(mapped_feature_names)[sorted_idx], color='gray')
        plt.title('Feature Importance (Random Forest)', fontsize=22)

    plt.xlabel('Importance', fontsize=20)
    plt.ylabel('Features', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()

# Function to plot SHAP values
def plot_shap_values(shap_values, feature_names):
    mapped_feature_names = [feature_name_mapping.get(f, f) for f in feature_names]
    shap_values_rf_array = np.array(shap_values)
    shap_values_abs = np.abs(shap_values_rf_array)
    shap_values_high = shap_values_abs[:, :, 0].mean(axis=0)
    shap_values_low = shap_values_abs[:, :, 1].mean(axis=0)
    total_importance = shap_values_high + shap_values_low
    sorted_indices = np.argsort(total_importance)[::-1]
    shap_values_high = shap_values_high[sorted_indices]
    shap_values_low = shap_values_low[sorted_indices]
    mapped_feature_names_sorted = np.array(mapped_feature_names)[sorted_indices]

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(mapped_feature_names_sorted))
    ax.barh(x, shap_values_high, color="red", label="High")
    ax.barh(x, shap_values_low, left=shap_values_high, color="blue", label="Low")
    ax.set_yticks(x)
    ax.set_yticklabels(mapped_feature_names_sorted, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("SHAP value (Average impact)", fontsize=14)
    ax.set_title("SHAP Feature Importance (High vs. Low)", fontsize=16)
    ax.legend()
    st.pyplot(fig)
    plt.close()

# Function to plot SHAP dot plot
def plot_shap_dot(shap_values, X_scaled, feature_names):
    mapped_feature_names = [feature_name_mapping.get(f, f) for f in feature_names]
    shap_values = np.array(shap_values)
    shap_values_selected = shap_values[:, :, 1] if shap_values.ndim == 3 and shap_values.shape[-1] == 2 else shap_values
    if shap_values_selected.shape != X_scaled.shape:
        raise ValueError(f"Shape mismatch: shap_values_selected {shap_values_selected.shape} vs X_scaled {X_scaled.shape}")
    explanation = shap.Explanation(values=shap_values_selected, data=X_scaled, feature_names=mapped_feature_names)
    fig, ax = plt.subplots(figsize=(12, 10))
    shap.plots.beeswarm(explanation, show=False)
    st.pyplot(fig)
    plt.close()

# Function to plot SHAP waterfall plot
def plot_shap_waterfall(shap_values, X_scaled, feature_names, index=0):
    mapped_feature_names = [feature_name_mapping.get(f, f) for f in feature_names]
    shap_values = np.array(shap_values)
    shap_values_selected = shap_values[index, :, 1] if shap_values.ndim == 3 else shap_values[index]
    base_value = shap_values_selected.mean()
    explanation = shap.Explanation(values=shap_values_selected, base_values=base_value, data=X_scaled.iloc[index], feature_names=mapped_feature_names)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(explanation, show=False)
    st.pyplot(fig)
    plt.close()

# Load trained models
logreg_model = pickle.load(open('best_Logistic Regression_model.pkl', 'rb'))
rf_model = pickle.load(open('best_Random Forest_model.pkl', 'rb'))

# Streamlit app setup
st.set_page_config(page_title="Cattle Antibody Pairing Predictor", page_icon="cattle_antibody_1.png")
st.image("cattle_antibody_2.png", use_container_width=True)
st.title("Cattle Antibody Pairing Predictor")
st.subheader("Authors: Anthony Onoja, Bharti Mittal, Nophar Geifman, John Hammond, Marie Bonnet-Di Placido")

st.write("""
Welcome to the Cattle Antibody Pairing Predictor web app! This tool uses machine learning to predict the pairing status of heavy and light chains in cattle antibody sequences. It leverages physiochemical properties of CDR3 regions (length, instability, hydrophobicity, isoelectric point, cysteine count, net charge, and structural compatibility) to predict pairing preferences (High or Low) based on pLDDT scores (>0.80 for High in ultralong sequences). Logistic Regression is the best-performing model, but users can also select Random Forest.
""")

st.write("""
To use the app, upload a CSV file with 'AA_H', 'CDR3_H', 'AA_L', and 'CDR3_L' sequences. Choose whether to perform combinatory analysis (pairing all heavy and light chains) or process the data directly. Select a model and click 'Run' to view predictions and visualizations. For combinatory analysis, consider computational limits (n < 1000) and use smaller batch sizes or cluster resources for efficiency.
""")

# Display steps
steps = [
    "Prepare a CSV file with 'AA_H', 'CDR3_H', 'AA_L', and 'CDR3_L' sequences.",
    "Ensure no missing values in the CSV file.",
    "Refer to the sample dataframe format below.",
    "Upload your CSV file.",
    "Choose whether to perform combinatory analysis.",
    "Select a predictive model (Logistic Regression or Random Forest).",
    "Click 'Run' to generate predictions and visualizations."
]

st.subheader("Steps to Use the Web App:")
for i, step in enumerate(steps, start=1):
    st.write(f"{i}) {step}")

# Sample dataframe
st.subheader("Sample Dataframe Format")
sample_data = {
    'AA_H': ['QVESDREAQWKELKELSXSXSAQASEKDJKDJDLDJ', 'QWSEWAXCESDEWSDWWHEJDTHWHJHHDHDAS', 'QESASEDRDFDFDFJHGFGDGEHDGEHSHDSNSG'],
    'CDR3_H': ['CDEFGHIKLMNPQRSTVWY', 'KLMNPQRSTVWYACDEFGHI', 'QRSTVWYACDEFGHIKLMN'],
    'AA_L': ['QWKEKEKDKLDKWJDGFRHSJWGS', 'LSWEQSGHSJDJHDJSJDJDE', 'QESWEKSJEKSJDLVBSHSKDK'],
    'CDR3_L': ['CDEFGHIKLMNPQRSTVWY', 'KLMNPQRSTVWYACDEFGHI', 'QRSTVWYACDEFGHIKLMN']
}
sample_df = pd.DataFrame(sample_data)
st.write("Here's how your dataframe should look:")
st.write(sample_df)

# User input for combinatory analysis
st.header("Upload and Configure Analysis")
combinatory_choice = st.radio("Perform Combinatory Analysis?", ("No", "Yes"))
csv_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Process uploaded file
if csv_file:
    df = pd.read_csv(csv_file)

    # Combinatory analysis
    if combinatory_choice == "Yes":
        st.write("Performing combinatory analysis (pairing all heavy and light chains)...")
        first_two_columns = df.iloc[:, :2].dropna()
        last_two_columns = df.iloc[:, 2:4].dropna()
        first_two_columns['key'] = 1
        last_two_columns['key'] = 1
        df = pd.merge(first_two_columns, last_two_columns, on='key').drop('key', axis=1)
    else:
        st.write("Processing input data directly (no combinatory analysis)...")
        df = df.dropna()

    # Clean sequences
    df['CDR3_H'] = df['CDR3_H'].apply(lambda x: ''.join([aa.upper() for aa in str(x) if aa.isalpha() and aa.upper() != 'O']))
    df['CDR3_L'] = df['CDR3_L'].apply(lambda x: ''.join([aa.upper() for aa in str(x) if aa.isalpha() and aa.upper() != 'O']))

    # Calculate protein properties
    df['CDR3_H_Len'] = df['CDR3_H'].apply(len)
    df['CDR3_L_Len'] = df['CDR3_L'].apply(len)
    df['CDR3_H_instability'] = df['CDR3_H'].apply(lambda x: ProteinAnalysis(x).instability_index())
    df['CDR3_L_instability'] = df['CDR3_L'].apply(lambda x: ProteinAnalysis(x).instability_index())
    df['CDR3_H_gravy'] = df['CDR3_H'].apply(lambda x: ProteinAnalysis(x).gravy())
    df['CDR3_L_gravy'] = df['CDR3_L'].apply(lambda x: ProteinAnalysis(x).gravy())
    df['CDR3_H_cys_count'] = df['CDR3_H'].apply(lambda x: ProteinAnalysis(x).count_amino_acids()['C'])
    df['CDR3_L_cys_count'] = df['CDR3_L'].apply(lambda x: ProteinAnalysis(x).count_amino_acids()['C'])
    df['CDR3_H_PI'] = df['CDR3_H'].apply(lambda x: ProteinAnalysis(x).isoelectric_point())
    df['CDR3_L_PI'] = df['CDR3_L'].apply(lambda x: ProteinAnalysis(x).isoelectric_point())
    df['CDRH3_Net_Charge'] = df['CDR3_H'].apply(lambda x: calculate_net_charge(x, amino_acid_charges))
    df['CDRL3_Net_Charge'] = df['CDR3_L'].apply(lambda x: calculate_net_charge(x, amino_acid_charges))
    df['Structural_Compatibility'] = df.apply(calculate_structural_compatibility, axis=1)

    # Select features for prediction
    features = ['CDR3_H_Len', 'CDR3_L_Len', 'CDR3_H_instability', 'CDR3_L_instability',
                'CDR3_H_gravy', 'CDR3_L_gravy', 'Structural_Compatibility',
                'CDR3_H_PI', 'CDR3_L_PI', 'CDR3_H_cys_count', 'CDR3_L_cys_count',
                'CDRH3_Net_Charge', 'CDRL3_Net_Charge']

    # Prepare and scale data
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Model selection
    model_choice = st.selectbox("Select a Predictive Model Classifier:", ("Logistic Regression", "Random Forest"))

    if st.button("Click Run"):
        model = logreg_model if model_choice == "Logistic Regression" else rf_model
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]
        prediction_labels = ['High' if pred == 1 else 'Low' for pred in predictions]

        # SHAP analysis for Random Forest
        if model_choice == "Random Forest":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled)
            with st.expander("SHAP Feature Importance"):
                plot_shap_values(shap_values, features)
                plot_shap_dot(shap_values, X_scaled, features)
                plot_shap_waterfall(shap_values, X_scaled, features)

        # Prepare output
        features_1 = ['AA_H', 'CDR3_H', 'AA_L', 'CDR3_L'] + features
        output_df = pd.concat([df[features_1], pd.DataFrame({'Prediction': prediction_labels, 'Prediction_Probability': probabilities})], axis=1)
        output_df = output_df.loc[:, ~output_df.columns.duplicated()]

        # Display results
        st.header("Prediction Results")
        st.write("Click each expander to view results:")

        with st.expander("Preview of Prediction Results"):
            st.write(output_df.head())
            output_file_path = os.path.join(os.getcwd(), "prediction_result.csv")
            output_df.to_csv(output_file_path, index=False)
            st.download_button("Download Prediction Results", data=open(output_file_path, 'rb'), file_name="prediction_result.csv")
            st.success("Prediction Results ready for download!")

        with st.expander("Prediction Probability Distributions by Class"):
            plt.figure(figsize=(8, 4))
            sns.histplot(output_df['Prediction_Probability'][output_df['Prediction'] == "High"], bins=30, kde=True, color='darkred', label='High')
            sns.histplot(output_df['Prediction_Probability'][output_df['Prediction'] == "Low"], bins=30, kde=True, color='darkblue', label='Low')
            plt.xlabel('Prediction Probability', fontsize=20)
            plt.ylabel('Frequency', fontsize=20)
            plt.title('Prediction Probability Distribution by Class', fontsize=20)
            plt.legend(fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            st.pyplot(plt)
            plt.close()

        with st.expander("Predicted Pairing Preference Count"):
            predicted_counts = pd.Series(prediction_labels).value_counts()
            total_samples = len(output_df)
            percentages = [(count / total_samples) * 100 for count in predicted_counts]
            fig, ax = plt.subplots()
            bars = ax.bar(predicted_counts.index, predicted_counts.values, color=['darkblue', 'tomato'])
            ax.tick_params(labelsize=16)
            for bar, percentage in zip(bars, percentages):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{percentage:.1f}%', ha='center', va='bottom', fontsize=12)
            ax.set_xlabel('Predicted Pairing Preference', fontsize=18)
            ax.set_ylabel('Count', fontsize=18)
            ax.set_title('Predicted Pairing Status Distribution', fontsize=20)
            st.pyplot(fig)
            plt.close()

        if model_choice in ['Logistic Regression', 'Random Forest']:
            with st.expander('Feature Importance Analysis'):
                plot_feature_importance(model, features)

        high_predictions_df = output_df[output_df['Prediction'] == "High"]
        pair_counts = high_predictions_df.groupby(['CDR3_H', 'CDR3_L']).size().reset_index(name='count')
        top_10_pairs = pair_counts.sort_values(by='count', ascending=False).head(10)

        with st.expander("Top 10 Most Frequent Predicted 'High' CDR3_H and CDR3_L Pairs"):
            st.write(top_10_pairs)
            output_file_path = os.path.join(os.getcwd(), "top_10_high_pairs.csv")
            top_10_pairs.to_csv(output_file_path, index=False)
            st.download_button("Download Top 10 High Pairs", data=open(output_file_path, 'rb'), file_name="top_10_high_pairs.csv")
            st.success("Top 10 High Pairs ready for download!")

        prediction_grouped = output_df.groupby(['CDR3_H', 'CDR3_L']).size().reset_index(name='Frequency_count_CDR3_H')
        prediction_grouped = prediction_grouped.merge(output_df[['CDR3_H', 'CDR3_L', 'Prediction']], on=['CDR3_H', 'CDR3_L'], how='left').drop_duplicates()
        prediction_grouped['Frequency_count_CDR3_L'] = prediction_grouped.groupby('CDR3_L')['CDR3_L'].transform('count')
        final_df = prediction_grouped[['CDR3_H', 'CDR3_L', 'Frequency_count_CDR3_H', 'Frequency_count_CDR3_L', 'Prediction']]

        with st.expander("Prediction Frequency Count Results"):
            st.write(final_df.head())
            output_file_path = os.path.join(os.getcwd(), "prediction_frequency_results.csv")
            final_df.to_csv(output_file_path, index=False)
            st.download_button("Download Frequency Counts", data=open(output_file_path, 'rb'), file_name="prediction_frequency_results.csv")
            st.success("Prediction Frequency ready for download!")

        with st.expander("Prediction Count (Heavy and Light CDR3)"):
            fig, ax = plt.subplots(figsize=(14, 10))
            color_map = {'High': 'darkred', 'Low': 'darkblue'}
            for preference in color_map.keys():
                subset = final_df[final_df['Prediction'] == preference]
                ax.scatter(subset['Frequency_count_CDR3_L'], subset['Frequency_count_CDR3_H'], c=color_map[preference], label=preference, alpha=0.6)
            ax.set_xlabel('Number of CDR3_L', fontsize=20)
            ax.set_ylabel('Number of CDR3_H', fontsize=20)
            ax.tick_params(axis='both', labelsize=20)
            ax.grid(False)
            ax.legend(title='Pairing preference', fontsize=18, title_fontsize=20)
            st.pyplot(fig)
            plt.close()
