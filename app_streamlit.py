import streamlit as st
import pandas as pd
import joblib
from io import BytesIO

# Load model
@st.cache_resource
def load_model():
    return joblib.load("lead_model.pkl")

model = load_model()

def score_band(prob):
    if prob >= 0.80:
        return "HOT"
    elif prob >= 0.50:
        return "WARM"
    else:
        return "COLD"

def clean_uploaded_data(df: pd.DataFrame):
    """
    Cleans incoming CSV so the model receives compatible datatypes.
    - Converts Yes/No-like fields to 1/0
    - Replaces string 'nan', blanks, and actual NaN with proper NaN
    - Drops columns not seen during training (optional)
    """
    df = df.copy()
    
    # Normalize yes/no values
    for col in df.columns:
        unique_vals = df[col].dropna().astype(str).str.lower().unique()
        if set(unique_vals).issubset({"yes", "no", "y", "n", "true", "false", "1", "0"}):
            df[col] = df[col].astype(str).str.lower().map({
                "yes": 1, "y": 1, "true": 1, "1": 1,
                "no": 0, "n": 0, "false": 0, "0": 0
            })

    # Standardize missing values
    df = df.replace(["", " ", "nan", "NaN", "None"], pd.NA)
    
    return df


st.title("AI-Powered Lead Scoring Demo")
st.markdown("Upload a CSV with your leads. The model will classify them as HOT, WARM, or COLD.")

uploaded_file = st.file_uploader("Upload Lead CSV", type=["csv"])

if uploaded_file:
    leads = pd.read_csv(uploaded_file)
    st.write(f"✅ File uploaded successfully! {leads.shape[0]} records")

    if "Prospect ID" not in leads.columns:
        st.warning("The file must contain a 'Prospect ID' column.")
    else:
        # Clean the input data
        leads_clean = clean_uploaded_data(leads)

        try:
            probs = model.predict_proba(leads_clean)[:, 1]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        leads["score"] = probs
        leads["classification"] = leads["score"].apply(score_band)

        # Output limited columns for download
        result_df = leads[["Prospect ID", "Name", "Email", "Phone", "classification"]]

        st.subheader("Classification Summary")
        st.bar_chart(result_df["classification"].value_counts())

        st.dataframe(result_df.head(15))

        # Download button
        csv_buffer = BytesIO()
        result_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)  

        st.download_button(
            label="Download Scored CSV",
            data=csv_buffer,
            file_name="scored_leads.csv",
            mime="text/csv"
        )
