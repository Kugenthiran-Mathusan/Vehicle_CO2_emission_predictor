import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from pathlib import Path

# =========================
# Paths (relative to this file)
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "final_model.pkl"
DATA_PATH  = BASE_DIR / "cleaned_vehicle_dataset.csv"

# =========================
# Streamlit page config
# =========================
st.set_page_config(page_title="CO₂ Emissions Predictor", page_icon="🚗", layout="wide")

st.title("🚗 Vehicle CO₂ Emissions Predictor")
st.markdown("Predict vehicle **CO₂ emissions (g/km)** from specs using a Multiple Linear Regression pipeline.")

# =========================
# Cached loaders (fast & safe)
# =========================
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
        st.error("❌ final_model.pkl not found or file is empty. Make sure it exists in this folder.")
        st.stop()
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.warning("⚠️ cleaned_vehicle_dataset.csv not found. Some comparisons will be disabled.")
        return None
    return pd.read_csv(DATA_PATH)

model = load_model()
df = load_data()

# =========================
# Fuel type mapping (UI → training codes)
# Adjust ONLY if your training codes were different
# Z=Regular Petrol, X=Premium Petrol, D=Diesel, E=Ethanol, N=Natural Gas
# =========================
FUEL_UI_TO_CODE = {
    "Petrol": "Z",
    "Diesel": "D",
    "Ethanol": "E",
    "Natural Gas": "N",
    "Premium Petrol": "X",
}

# =========================
# Sidebar Inputs
# =========================
st.sidebar.header("🔧 Enter Vehicle Specifications")

engine_size = st.sidebar.number_input("Engine size (L)", min_value=0.5, max_value=10.0, step=0.1, value=2.0)
cylinders = st.sidebar.number_input("Number of Cylinders", min_value=3, max_value=12, step=1, value=4)
fuel_type_ui = st.sidebar.selectbox("Fuel Type", list(FUEL_UI_TO_CODE.keys()))
combined_l_100km = st.sidebar.number_input("Combined (L/100 km)", min_value=2.0, max_value=25.0, step=0.1, value=8.5)

# Predict button
if st.sidebar.button("Predict"):
    # Map UI label to training code
    fuel_type_code = FUEL_UI_TO_CODE[fuel_type_ui]

    # Build input row expected by the pipeline (raw columns, exact names!)
    input_data = pd.DataFrame({
        "Engine size (L)": [engine_size],
        "Cylinders": [cylinders],
        "Fuel type": [fuel_type_code],
        "Combined (L/100 km)": [combined_l_100km],
    })

    # Predict
    try:
        prediction = float(model.predict(input_data)[0])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # =========================
    # Results Section
    # =========================
    st.subheader("🔮 Predicted CO₂ Emissions")
    st.metric("CO₂ emissions (g/km)", f"{prediction:.2f}")

    # Dataset-based metrics & comparisons (if dataset available)
    if df is not None and all(col in df.columns for col in ["CO2 emissions (g/km)"]):
        avg_emission = float(df["CO2 emissions (g/km)"].mean())

        # Build X,y to evaluate model on this dataset (not a proper test set, just a display)
        X = df.drop(columns=["CO2 emissions (g/km)", "CO2 rating", "Smog rating"], errors="ignore")
        y = df["CO2 emissions (g/km)"]
        try:
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
            mae = float(mean_absolute_error(y, y_pred))
        except Exception as e:
            st.warning(f"Could not evaluate on dataset: {e}")
            y_pred, r2, rmse, mae = None, None, None, None

        st.write("---")
        st.subheader("📊 Model Performance (on provided dataset)")
        if r2 is not None:
            st.write(f"**R² Score:** {r2:.4f}")
            st.write(f"**RMSE:** {rmse:.2f} g/km")
            st.write(f"**MAE:** {mae:.2f} g/km")
        else:
            st.info("Performance not shown due to evaluation issue.")

        # =========================
        # Graphs Section
        # =========================
        col1, col2 = st.columns(2)

        # 1) Actual vs Predicted
        with col1:
            if y_pred is not None:
                st.write("### 📈 Actual vs Predicted")
                fig, ax = plt.subplots(figsize=(5,4))
                sns.scatterplot(x=y, y=y_pred, alpha=0.6, ax=ax)
                ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
                ax.set_xlabel("Actual CO₂ emissions (g/km)")
                ax.set_ylabel("Predicted CO₂ emissions (g/km)")
                ax.set_title(f"R² = {r2:.3f}")
                st.pyplot(fig)
            else:
                st.info("No predictions for plotting.")

        # 2) Residuals
        with col2:
            if y_pred is not None:
                st.write("### 📊 Residuals Distribution")
                residuals = y - y_pred
                fig, ax = plt.subplots(figsize=(5,4))
                sns.histplot(residuals, bins=30, kde=True, ax=ax)
                ax.set_xlabel("Prediction Error (g/km)")
                ax.set_title("Residuals Distribution")
                st.pyplot(fig)

        # 3) Your vs Average
        st.write("---")
        st.write("### 🚘 Your Vehicle vs Dataset Average")
        comparison_df = pd.DataFrame({
            "Category": ["Your Vehicle", "Dataset Average"],
            "CO₂ emissions (g/km)": [prediction, avg_emission]
        })
        fig, ax = plt.subplots(figsize=(6,4))
        sns.barplot(x="Category", y="CO₂ emissions (g/km)", data=comparison_df, ax=ax)
        st.pyplot(fig)
    else:
        st.info("Dataset not found — skipping performance & comparison charts.")

st.write("---")
st.markdown("Built with ❤️ using **Python, Streamlit, and Scikit-learn** ")
