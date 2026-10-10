
import json
import functools
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import torch

# ---------------- PAGE SETUP ----------------

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_PATH = BASE_DIR / "customer_churn.csv"

st.title("📊 Customer Churn Intelligence")
st.caption("AI-powered customer churn prediction using TabPFN")


# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    model_path = ARTIFACTS_DIR / "tabpfn_churn_model.pkl"
    columns_path = ARTIFACTS_DIR / "feature_columns.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not columns_path.exists():
        raise FileNotFoundError(
            f"Feature columns file not found: {columns_path}"
        )

    original_torch_load = torch.load

    try:
        torch.load = functools.partial(
            original_torch_load,
            map_location="cpu",
            weights_only=False
        )
        model = joblib.load(model_path)
    finally:
        torch.load = original_torch_load

    # Configure model for CPU inference
    model.device = "cpu"
    model.devices_ = (torch.device("cpu"),)
    model.keep_cache_on_device = False

    with open(columns_path, "r", encoding="utf-8") as file:
        feature_columns = json.load(file)

    return model, feature_columns


# ---------------- LOAD DATASET ----------------

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)

    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(
            data["TotalCharges"].astype(str).str.strip(),
            errors="coerce"
        ).fillna(0)

    if "SeniorCitizen" in data.columns:
        data["SeniorCitizen"] = pd.to_numeric(
            data["SeniorCitizen"],
            errors="coerce"
        ).fillna(0).astype(int)

    if "tenure" in data.columns:
        data["tenure"] = pd.to_numeric(
            data["tenure"],
            errors="coerce"
        ).fillna(0)

    if "MonthlyCharges" in data.columns:
        data["MonthlyCharges"] = pd.to_numeric(
            data["MonthlyCharges"],
            errors="coerce"
        ).fillna(0)

    return data


try:
    model, feature_columns = load_model()
    df = load_data()
    st.success("Model and customer dataset loaded successfully!")

except Exception as error:
    st.error("Could not load the model or dataset.")
    st.exception(error)
    st.stop()


# ---------------- CUSTOMER INPUT FORM ----------------

st.header("🔎 Customer Churn Prediction")
st.write("Enter customer details to estimate the likelihood of churn.")

input_columns = [
    col for col in feature_columns
    if col not in ["Churn", "customerID"]
]

missing_columns = [
    col for col in input_columns
    if col not in df.columns
]

if missing_columns:
    st.error(f"Dataset is missing these features: {missing_columns}")
    st.stop()

categorical_columns = [
    col for col in input_columns
    if not pd.api.types.is_numeric_dtype(df[col])
]

numeric_columns = [
    col for col in input_columns
    if col not in categorical_columns
]

with st.form("customer_form"):
    st.subheader("Customer Details")

    customer_input = {}
    col1, col2 = st.columns(2)

    for i, feature in enumerate(input_columns):
        container = col1 if i % 2 == 0 else col2

        with container:
            if feature in categorical_columns:
                choices = sorted(
                    df[feature]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                if not choices:
                    st.error(f"No available values for {feature}")
                else:
                    customer_input[feature] = st.selectbox(
                        feature,
                        choices,
                        key=f"input_{feature}"
                    )

            else:
                values = pd.to_numeric(
                    df[feature],
                    errors="coerce"
                ).dropna()

                if values.empty:
                    minimum, maximum, default = 0.0, 100.0, 0.0
                else:
                    minimum = float(values.min())
                    maximum = float(values.max())
                    default = float(values.median())

                customer_input[feature] = st.number_input(
                    feature,
                    min_value=minimum,
                    max_value=maximum,
                    value=default,
                    key=f"input_{feature}"
                )

    # IMPORTANT: The submit button must be inside the form.
    submitted = st.form_submit_button(
        "🔮 Predict Churn",
        width="stretch"
    )


# ---------------- PREDICTION ----------------

if submitted:
    try:
        with st.spinner("TabPFN is predicting..."):

            # Build customer input in the training feature order
            customer_df = pd.DataFrame(
                [{col: customer_input[col] for col in input_columns}],
                columns=input_columns
            )

            # Match the model's training data types
            for col in input_columns:
                if col in categorical_columns:
                    customer_df[col] = customer_df[col].astype(
                        df[col].dtype
                    )
                else:
                    customer_df[col] = pd.to_numeric(
                        customer_df[col],
                        errors="raise"
                    )

            if list(customer_df.columns) != input_columns:
                raise ValueError(
                    "Input feature order does not match."
                )

            st.write("Running TabPFN prediction...")

            prediction = int(model.predict(customer_df)[0])
            probabilities = model.predict_proba(customer_df)[0]

            classes = list(model.classes_)

            if 1 not in classes:
                raise ValueError(
                    f"Churn class 1 was not found. Model classes: {classes}"
                )

            churn_probability = float(
                probabilities[classes.index(1)]
            )

        # ---------------- DISPLAY PREDICTION ----------------

        st.divider()
        st.header("📊 Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            if prediction == 1:
                st.error("⚠️ Prediction: Likely to Churn")
            else:
                st.success("✅ Prediction: Likely to Stay")

        with result_col2:
            st.metric(
                label="Churn Probability",
                value=f"{churn_probability * 100:.2f}%"
            )

        st.progress(
            max(0.0, min(1.0, churn_probability))
        )

        if churn_probability >= 0.5:
            st.write(
                "The model estimates a relatively high likelihood "
                "that this customer will leave."
            )
        else:
            st.write(
                "The model estimates a lower likelihood "
                "that this customer will leave."
            )

        st.caption(
            "This is a model estimate, not a guarantee of customer behavior."
        )

        # ---------------- INDIVIDUAL CUSTOMER EXPLANATION ----------------

        st.divider()
        st.header("🔍 Why did the model make this prediction?")

        st.write(
            "The following comparisons show how the predicted churn "
            "probability changes when one customer feature is changed "
            "while all other features remain the same."
        )

        baseline_probability = churn_probability
        explanation_results = []

        important_features = [
            "Contract",
            "tenure",
            "MonthlyCharges"
        ]

        for feature in important_features:
            if feature not in input_columns:
                continue

            reference_df = customer_df.copy()

            # Categorical feature comparison
            if feature in categorical_columns:
                choices = sorted(
                    df[feature]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                if len(choices) < 2:
                    continue

                current_value = str(
                    customer_df.iloc[0][feature]
                )

                alternative = next(
                    (
                        value for value in choices
                        if value != current_value
                    ),
                    None
                )

                if alternative is None:
                    continue

                reference_df[feature] = (
                    reference_df[feature].astype(object)
                )

                reference_df.loc[
                    reference_df.index[0], feature
                ] = alternative

                reference_df[feature] = reference_df[feature].astype(
                    df[feature].dtype
                )

                comparison_value = alternative

            # Numeric feature comparison
            else:
                values = pd.to_numeric(
                    df[feature],
                    errors="coerce"
                ).dropna()

                if values.empty:
                    continue

                current_value = float(
                    customer_df.iloc[0][feature]
                )

                comparison_value = float(values.median())

                if current_value == comparison_value:
                    continue

                reference_df[feature] = comparison_value

            # Calculate the probability after changing one feature
            reference_probabilities = model.predict_proba(
                reference_df
            )[0]

            reference_probability = float(
                reference_probabilities[
                    list(model.classes_).index(1)
                ]
            )

            difference = (
                reference_probability - baseline_probability
            )

            explanation_results.append({
                "Feature": feature,
                "Customer value": str(
                    customer_df.iloc[0][feature]
                ),
                "Compared value": str(comparison_value),
                "Original churn probability":
                    baseline_probability * 100,
                "Compared churn probability":
                    reference_probability * 100,
                "Probability change (percentage points)":
                    difference * 100
            })

        # ---------------- EXPLANATION TABLE AND CHART ----------------

        if explanation_results:
            explanation_df = pd.DataFrame(
                explanation_results
            )

            explanation_df["Absolute change"] = (
                explanation_df[
                    "Probability change (percentage points)"
                ].abs()
            )

            explanation_df = explanation_df.sort_values(
                "Absolute change",
                ascending=False
            )

            st.subheader("Individual Feature Comparisons")

            st.dataframe(
                explanation_df.drop(
                    columns=["Absolute change"]
                ).round(2),
                width="stretch",
                hide_index=True
            )

            st.caption(
                "A positive change means the alternative value increased "
                "the estimated churn probability; a negative change means "
                "it decreased. These are one-feature-at-a-time comparisons, "
                "not causal effects or SHAP values."
            )

            chart_df = explanation_df.set_index("Feature")[
                "Probability change (percentage points)"
            ]

            st.subheader("Impact of Changing Individual Features")
            st.bar_chart(chart_df)

        else:
            st.info(
                "No feature comparisons could be generated."
            )

        # ---------------- CUSTOMER DETAILS ----------------

        with st.expander("View submitted customer details"):
            st.dataframe(
                customer_df,
                width="stretch"
            )

    except Exception as error:
        st.error(
            "Prediction failed. See the technical details below."
        )
        st.exception(error)

