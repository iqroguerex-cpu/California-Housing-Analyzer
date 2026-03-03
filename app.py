import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="California Housing Analyzer",
    page_icon="🏠",
    layout="wide"
)

# =========================
# DARK PROFESSIONAL UI
# =========================
st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
    color: white;
}
div[data-testid="stMetricValue"] {
    color: #00FFAA;
}
section[data-testid="stSidebar"] {
    background-color: #161B22;
}
</style>
""", unsafe_allow_html=True)

plt.style.use("dark_background")
sns.set_style("darkgrid")

# =========================
# LOAD DATA
# =========================
dataset = fetch_california_housing()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df["Price"] = dataset.target

# =========================
# SPLIT
# =========================
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# =========================
# SCALING
# =========================
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# =========================
# MODEL
# =========================
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# =========================
# METRICS
# =========================
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# =========================
# HEADER
# =========================
st.title("🏠 California Housing Price Analyzer")
st.caption("Multiple Linear Regression • Interactive Dashboard")

st.divider()

# =========================
# METRICS DISPLAY
# =========================
col1, col2, col3 = st.columns(3)

col1.metric("R² Score", f"{r2:.3f}")
col2.metric("MAE ($100k)", f"{mae:.3f}")
col3.metric("RMSE ($100k)", f"{rmse:.3f}")

st.divider()

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview", "📈 Predictions", "📉 Residuals", "🎛 Predict Price"]
)

# =========================
# TAB 1 — OVERVIEW
# =========================
with tab1:
    st.subheader("Feature Correlation Heatmap")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), cmap="viridis", ax=ax1)
    st.pyplot(fig1)

    st.subheader("Feature Importance")

    coef_df = pd.DataFrame({
        "Feature": dataset.feature_names,
        "Coefficient": model.coef_,
        "Impact": np.abs(model.coef_)
    }).sort_values(by="Impact", ascending=False)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=coef_df,
        x="Impact",
        y="Feature",
        palette="magma",
        ax=ax2
    )
    ax2.set_title("Feature Importance (Absolute Coefficient)")
    st.pyplot(fig2)

# =========================
# TAB 2 — PREDICTIONS
# =========================
with tab2:
    st.subheader("Actual vs Predicted Prices")

    fig3, ax3 = plt.subplots()
    ax3.scatter(y_test, y_pred, alpha=0.4, color="#00FFAA")
    ax3.set_xlabel("Actual Price")
    ax3.set_ylabel("Predicted Price")

    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    ax3.plot([min_val, max_val], [min_val, max_val], color="red")

    st.pyplot(fig3)

# =========================
# TAB 3 — RESIDUALS
# =========================
with tab3:
    residuals = y_test - y_pred

    st.subheader("Residual Plot")

    fig4, ax4 = plt.subplots()
    ax4.scatter(y_pred, residuals, alpha=0.4, color="#FF6F61")
    ax4.axhline(0, color="white")
    ax4.set_xlabel("Predicted Price")
    ax4.set_ylabel("Residual")
    st.pyplot(fig4)

# =========================
# TAB 4 — INTERACTIVE PREDICTION
# =========================
with tab4:
    st.subheader("Predict House Price")

    st.markdown("Adjust the features to estimate house price.")

    col1, col2, col3, col4 = st.columns(4)

    # Logical Ranges (based on dataset meaning)
    MedInc = col1.slider("Median Income (10k USD)",
                         0.5, 15.0, 3.5, 0.1,
                         help="Median income in block group (in $10,000 units)")

    HouseAge = col2.slider("House Age (years)",
                           1, 52, 20, 1)

    AveRooms = col3.slider("Average Rooms",
                           2.0, 15.0, 5.0, 0.1)

    AveBedrms = col4.slider("Average Bedrooms",
                            0.5, 5.0, 1.0, 0.1)

    col5, col6, col7, col8 = st.columns(4)

    Population = col5.slider("Population",
                             100, 5000, 1000, 50)

    AveOccup = col6.slider("Average Occupancy",
                           1.0, 10.0, 3.0, 0.1)

    Latitude = col7.slider("Latitude",
                           32.0, 42.0, 34.0, 0.1)

    Longitude = col8.slider("Longitude",
                            -124.0, -114.0, -118.0, 0.1)

    user_input = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                            Population, AveOccup, Latitude, Longitude]])

    user_input_scaled = sc.transform(user_input)
    prediction = model.predict(user_input_scaled)

    st.success(f"Predicted House Price: ${prediction[0]*100000:,.0f}")
