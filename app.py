import streamlit as st
import pandas as pd
import joblib
import os
import glob

#  Paths
BASE_DIR = r"C:\Users\sanja\4.Leukemia Subtype Prediction Using Gene Expression\4.Leukemia_Subtype_Prediction"
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
PLOTS_DIR = os.path.join(BASE_DIR, "results", "plots")

# Load model
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    st.error(f" Model file not found at {MODEL_PATH}")
    st.stop()

# Find latest confusion matrix image
confusion_matrix_images = glob.glob(os.path.join(PLOTS_DIR, "*_confusion_matrix.png"))
if confusion_matrix_images:
    latest_conf_matrix = max(confusion_matrix_images, key=os.path.getctime)
else:
    latest_conf_matrix = None

# Streamlit UI
st.set_page_config(page_title="Leukemia Subtype Prediction", page_icon="🧬", layout="centered")

st.title(" Leukemia Subtype Prediction (Gene Expression)")
st.write("""
Upload a CSV file containing the **10 selected gene expression features** to predict leukemia subtype.
""")

# File upload
uploaded_file = st.file_uploader(" Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Read uploaded file
    input_df = pd.read_csv(uploaded_file)
    st.subheader(" Uploaded Data Preview")
    st.dataframe(input_df.head())

    try:
        # Prediction
        preds = model.predict(input_df)
        probs = model.predict_proba(input_df)

        # Display predictions
        st.subheader(" Predictions")
        result_df = input_df.copy()
        result_df["Predicted Subtype"] = preds
        for i, cls in enumerate(model.classes_):
            result_df[f"Prob_{cls}"] = probs[:, i]
        st.dataframe(result_df)

        # Summary counts
        st.subheader(" Prediction Summary")
        st.bar_chart(result_df["Predicted Subtype"].value_counts())

        # Show confusion matrix if available
        if latest_conf_matrix and os.path.exists(latest_conf_matrix):
            st.subheader(" Confusion Matrix (Test Data)")
            st.image(latest_conf_matrix, caption="Confusion Matrix of Best Model", use_column_width=True)
        else:
            st.warning(" No confusion matrix image found in results/plots folder.")

    except Exception as e:
        st.error(f" Error during prediction: {e}")

else:
    st.info(" Please upload a CSV file to start predictions.")
