import streamlit as st
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np



# ---- App UI ----
st.title("Regression & Random Forest Platform")

st.subheader("Requirements:")
st.text("1. MSA, Zip Code, County and City!.")
st.text("2. The target variable (what you're predicting) with numerical values within the dataset.")

st.subheader("Some limitations of this app:")
st.text("1. Uploading files larger than 200MB (Streamlit’s hard cap).")
st.text("2. Non-CSV file formats (Excel, JSON, TXT, etc.).")
st.text("3. Missing a target column or trying to run with target=y that isn’t in the dataset.")
st.text("4. Selecting no features or selecting only the target as a feature.")
st.text("5. Categorical (string) columns used as features (regression/random forest expects numeric).")
st.text("6. Columns with all NaN values.")
st.text("7. Target column with non-numeric data.")
st.text("8. Mismatched row counts after dropping NaNs (X and y index misalignment).")
st.text("9. Extremely wide datasets (tens of thousands of features) → can cause memory errors with statsmodels.")
st.text("Remember, Before uploading, your file should have MSA, Zip Code, County and City!")


# ---- File uploader ----
uploaded_file = st.file_uploader("Upload a CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.error("❌ The uploaded CSV is empty.")
            st.stop()
        st.write("Preview of your data:", df.head())
    except Exception as e:
        st.error(f"❌ Error reading CSV file: {e}")
        st.stop()

    try:
        features = st.multiselect("Select Features (X):", df.columns.tolist())
        target = st.selectbox("Select Target (y):", df.columns.tolist())

        if not features:
            st.warning("⚠️ Please select at least one feature column.")
            st.stop()
        if not target:
            st.warning("⚠️ Please select a target column.")
            st.stop()
        if target in features:
            st.error("❌ Target column cannot also be a feature.")
            st.stop()

        # Drop NaNs
        X = df[features].dropna()
        y = df[target].loc[X.index]

        if not np.issubdtype(y.dtype, np.number):
            st.error("❌ Target column must be numeric.")
            st.stop()
        if not all(np.issubdtype(X[col].dtype, np.number) for col in X.columns):
            st.error("❌ All selected features must be numeric.")
            st.stop()
        if X.shape[1] > 10000:
            st.error("❌ Too many features selected (over 10,000). Try reducing the dataset.")
            st.stop()

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        except Exception as e:
            st.error(f"❌ Error during train/test split: {e}")
            st.stop()

        # ---- Choose model ----
        model_type = st.radio("Choose Model:", ("OLS Regression", "Random Forest"))

        # ---- OLS REGRESSION ----
        if model_type == "OLS Regression":
            try:
                X_train_const = sm.add_constant(X_train)
                X_test_const = sm.add_constant(X_test, has_constant="add")
                model = sm.OLS(y_train, X_train_const).fit()

                summary_df = pd.DataFrame({
                    "Feature": model.params.index,  # Keep feature names
                    "coef": model.params.values,
                    "std err": model.bse.values,
                    "t": model.tvalues.values,
                    "P>|t|": model.pvalues.values,
                    "[0.025": model.conf_int()[0].values,
                    "0.975]": model.conf_int()[1].values
                })

                st.subheader("OLS Regression Results")
                st.metric("R-squared", f"{model.rsquared:.3f}")
                st.metric("Adj. R-squared", f"{model.rsquared_adj:.3f}")
                st.dataframe(summary_df.style.format("{:.3f}"))

                # ---- Predictions + Metrics ----
                y_pred = model.predict(X_test_const)
                test_metrics = {
                    "MAE": mean_absolute_error(y_test, y_pred),
                    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                    "R²": r2_score(y_test, y_pred)
                }
                st.subheader("Test Metrics")
                st.write(test_metrics)

                # ---- MASTER CSV EXPORT ----
                coeffs_out = summary_df.copy()
                coeffs_out.insert(0, "Section", "Coefficients")

                metrics_out = pd.DataFrame([test_metrics])
                metrics_out.insert(0, "Section", "Test Metrics")

                master_df = pd.concat([coeffs_out, metrics_out], ignore_index=True)

                csv_master = master_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download OLS Results (One CSV)",
                    data=csv_master,
                    file_name="ols_results.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"❌ OLS Regression failed: {e}")

    except Exception as e:
        st.error(f"❌ Error during setup: {e}")

