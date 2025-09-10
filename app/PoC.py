import streamlit as st
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import openai
import io

openai.api_key = st.secrets["open_api_key"]

def interpret_regression(coeffs_df, metrics_df):
    coeffs_text = coeffs_df.to_string(index=False)
    metrics_text = metrics_df.to_string(index=False)

    prompt = f"""
    You are an analyst interpreting the results of an OLS regression.

    Here is the coefficients table:
    {coeffs_text}

    Here are the test metrics:
    {metrics_text}

    Please explain:
    1. Which variables are statistically significant and what their coefficient signs mean.
    2. How strong the model is (based on R², MAE, RMSE).
    3. Practical interpretation of the coefficients (direction, magnitude, relevance).
    4. Any limitations of the model.
    """

    response = openai.chat.completions.create(
        model="gpt-4o-mini",  # or gpt-4o
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content

# ---- Interpretation function --

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

        # ---- OLS ----
        if model_type == "OLS Regression":
            try:
                X_train_const = sm.add_constant(X_train)
                X_test_const = sm.add_constant(X_test, has_constant="add")
                model = sm.OLS(y_train, X_train_const).fit()

                summary_df = pd.DataFrame({
                    "coef": model.params,
                    "std err": model.bse,
                    "t": model.tvalues,
                    "P>|t|": model.pvalues,
                    "[0.025": model.conf_int()[0],
                    "0.975]": model.conf_int()[1]
                })

                st.subheader("OLS Regression Results")
                st.metric("R-squared", f"{model.rsquared:.3f}")
                st.metric("Adj. R-squared", f"{model.rsquared_adj:.3f}")
                st.dataframe(summary_df.style.format("{:.3f}"))

                y_pred = model.predict(X_test_const)
                test_metrics = {
                    "MAE": mean_absolute_error(y_test, y_pred),
                    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                    "R²": r2_score(y_test, y_pred)
                }
                st.subheader("Test Metrics")
                st.write(test_metrics)

                # ---- Interpretation ----
                if st.button("Interpret Results"):
                    interpretation = interpret_regression(summary_df, pd.DataFrame([test_metrics]))
                    st.markdown("### 📊 Interpretation")
                    st.write(interpretation)

                    # ---- Build Master CSV ----
                    coeffs_out = summary_df.copy()
                    coeffs_out.insert(0, "Section", "Coefficients")

                    metrics_out = pd.DataFrame([test_metrics])
                    metrics_out.insert(0, "Section", "Test Metrics")

                    interp_out = pd.DataFrame({"Section": ["Interpretation"], "Interpretation": [interpretation]})

                    master_df = pd.concat([coeffs_out, metrics_out, interp_out], ignore_index=True)

                    buffer = io.StringIO()
                    master_df.to_csv(buffer, index=False)
                    buffer.seek(0)

                    st.download_button(
                        label="⬇️ Download Master CSV",
                        data=buffer,
                        file_name="ols_master_results.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"❌ OLS Regression failed: {e}")

        # ---- RANDOM FOREST ----
        elif model_type == "Random Forest":
            try:
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)

                st.subheader("Random Forest Metrics")
                rf_metrics = {
                    "MAE": mean_absolute_error(y_test, y_pred),
                    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                    "R²": r2_score(y_test, y_pred)
                }
                st.write(rf_metrics)

                rf_metrics_df = pd.DataFrame([rf_metrics])
                csv_rf_metrics = rf_metrics_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Random Forest Metrics (CSV)",
                    data=csv_rf_metrics,
                    file_name="rf_test_metrics.csv",
                    mime="text/csv"
                )

                st.subheader("Feature Importances")
                importances = pd.DataFrame({
                    "Feature": features,
                    "Importance": rf.feature_importances_
                }).sort_values(by="Importance", ascending=False)
                st.dataframe(importances)

                csv_importances = importances.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Feature Importances (CSV)",
                    data=csv_importances,
                    file_name="rf_feature_importances.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"❌ Random Forest failed: {e}")

    except Exception as e:
        st.error(f"❌ Error during setup: {e}")
