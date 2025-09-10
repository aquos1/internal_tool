import streamlit as st
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # clear it
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("Password incorrect")
        return False
    else:
        return True

# Main logic
if check_password():
    st.title("Secure Demo App")
    st.write("Welcome!")

st.title("Regression & Random Forest Platform")
st.subheader("Requirements:")
st.text("1. MSA, Zip Code, County and City!.")
st.text("2. The target variable (what you're predicting) with numerical values within the dataset.")



#limitations
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

#csv
uploaded_file = st.file_uploader("Upload a CSV File", type=["csv"])

if uploaded_file is not None:
    # tries
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.error("❌ The uploaded CSV is empty.")
            st.stop()
        st.write("Preview of your data:", df.head())
    except Exception as e:
        st.error(f"❌ Error reading CSV file: {e}")
        st.stop()

    # selecting features
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

        #drop nans
        X = df[features].dropna()
        y = df[target].loc[X.index]

        #validating data types
        if not np.issubdtype(y.dtype, np.number):
            st.error("❌ Target column must be numeric.")
            st.stop()

        if not all(np.issubdtype(X[col].dtype, np.number) for col in X.columns):
            st.error("❌ All selected features must be numeric.")
            st.stop()

        if X.shape[1] > 10000:
            st.error("❌ Too many features selected (over 10,000). Try reducing the dataset.")
            st.stop()

        # splitting 80 - 20
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        except Exception as e:
            st.error(f"❌ Error during train/test split: {e}")
            st.stop()

        # selecting which model to use
        model_type = st.radio("Choose Model:", ("OLS Regression", "Random Forest"))

        # OLS REGRESSION
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

                # metrics
                y_pred = model.predict(X_test_const)
                test_metrics = {
                    "MAE": mean_absolute_error(y_test, y_pred),
                    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                    "R²": r2_score(y_test, y_pred)
                }
                st.subheader("Test Metrics")
                st.write(test_metrics)

                # downloading
                csv_summary = summary_df.to_csv().encode("utf-8")
                st.download_button(
                    label="📥 Download OLS Coefficients (CSV)",
                    data=csv_summary,
                    file_name="ols_coefficients.csv",
                    mime="text/csv"
                )

                metrics_df = pd.DataFrame([test_metrics])
                csv_metrics = metrics_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download OLS Test Metrics (CSV)",
                    data=csv_metrics,
                    file_name="ols_test_metrics.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"❌ OLS Regression failed: {e}")

        # RANDOM FOREST
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

                # ✅ Export metrics
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

                # ✅ Export feature importances
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
