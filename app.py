import json
import functools
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch


# ---------------- PAGE SETUP ----------------

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
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
        raise FileNotFoundError(f"Feature columns file not found: {columns_path}")

    # Load the saved model on CPU, even if it was trained on a GPU.
    original_torch_load = torch.load
    try:
        torch.load = functools.partial(
            original_torch_load,
            map_location="cpu",
            weights_only=False,
        )
        model = joblib.load(model_path)
    finally:
        torch.load = original_torch_load

    # Configure TabPFN for CPU inference.
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
            errors="coerce",
        ).fillna(0)

    if "SeniorCitizen" in data.columns:
        data["SeniorCitizen"] = pd.to_numeric(
            data["SeniorCitizen"], errors="coerce"
        ).fillna(0).astype(int)

    for column in ["tenure", "MonthlyCharges"]:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column], errors="coerce"
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
    column for column in feature_columns
    if column not in ["Churn", "customerID"]
]

missing_columns = [column for column in input_columns if column not in df.columns]
if missing_columns:
    st.error(f"Dataset is missing these features: {missing_columns}")
    st.stop()

categorical_columns = [
    column for column in input_columns
    if not pd.api.types.is_numeric_dtype(df[column])
]

with st.form("customer_form"):
    st.subheader("Customer Details")
    customer_input = {}
    col1, col2 = st.columns(2)

    for index, feature in enumerate(input_columns):
        container = col1 if index % 2 == 0 else col2

        with container:
            if feature in categorical_columns:
                choices = sorted(
                    df[feature].dropna().astype(str).unique().tolist()
                )
                if not choices:
                    st.error(f"No available values for {feature}")
                    continue

                customer_input[feature] = st.selectbox(
                    feature,
                    choices,
                    key=f"input_{feature}",
                )
            else:
                values = pd.to_numeric(df[feature], errors="coerce").dropna()
                if values.empty:
                    minimum, maximum, default = 0.0, 100.0, 0.0
                else:
                    minimum = float(values.min())
                    maximum = float(values.max())
                    default = float(values.median())

                # Streamlit number_input requires min <= value <= max.
                if minimum == maximum:
                    customer_input[feature] = st.number_input(
                        feature,
                        value=default,
                        key=f"input_{feature}",
                    )
                else:
                    customer_input[feature] = st.number_input(
                        feature,
                        min_value=minimum,
                        max_value=maximum,
                        value=min(max(default, minimum), maximum),
                        key=f"input_{feature}",
                    )

    submitted = st.form_submit_button("🔮 Predict Churn", width="stretch")


# ---------------- PREDICTION ----------------

if submitted:
    try:
        with st.spinner("TabPFN is predicting..."):
            # Build the row in the exact feature order used for training.
            customer_df = pd.DataFrame(
                [{column: customer_input[column] for column in input_columns}],
                columns=input_columns,
            )

            # Match the dataset's feature data types.
            for column in input_columns:
                if column in categorical_columns:
                    customer_df[column] = customer_df[column].astype(
                        df[column].dtype
                    )
                else:
                    customer_df[column] = pd.to_numeric(
                        customer_df[column], errors="raise"
                    )

            if list(customer_df.columns) != input_columns:
                raise ValueError("Input feature order does not match the model.")

            # One model call for the main prediction.
            probabilities = model.predict_proba(customer_df)[0]
            classes = list(model.classes_)

            if 1 not in classes:
                raise ValueError(
                    f"Churn class 1 was not found. Model classes: {classes}"
                )

            churn_probability = float(probabilities[classes.index(1)])
            # Choose the class with the highest predicted probability.
            prediction = classes[int(np.argmax(probabilities))]

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
                value=f"{churn_probability * 100:.2f}%",
            )

        st.progress(max(0.0, min(1.0, churn_probability)))

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
            "These are one-feature-at-a-time comparisons. We change one "
            "feature while keeping the customer's other details unchanged, "
            "then compare the estimated churn probabilities."
        )

        important_features = ["Contract", "tenure", "MonthlyCharges"]
        comparison_rows = []
        comparison_metadata = []

        # Prepare alternative rows first so they can be predicted in one batch.
        for feature in important_features:
            if feature not in input_columns:
                continue

            reference_row = customer_df.copy()
            current_value = customer_df.iloc[0][feature]

            if feature in categorical_columns:
                choices = sorted(
                    df[feature].dropna().astype(str).unique().tolist()
                )
                if len(choices) < 2:
                    continue

                current_text = str(current_value)
                alternative = next(
                    (value for value in choices if value != current_text),
                    None,
                )
                if alternative is None:
                    continue

                reference_row[feature] = reference_row[feature].astype(object)
                reference_row.loc[reference_row.index[0], feature] = alternative
                reference_row[feature] = reference_row[feature].astype(
                    df[feature].dtype
                )
                comparison_value = alternative

            else:
                values = pd.to_numeric(df[feature], errors="coerce").dropna()
                if values.empty:
                    continue

                comparison_value = float(values.median())
                current_numeric = float(current_value)

                if np.isclose(current_numeric, comparison_value):
                    continue

                reference_row[feature] = comparison_value

            comparison_rows.append(reference_row)
            comparison_metadata.append(
                {
                    "Feature": feature,
                    "Customer value": str(current_value),
                    "Compared value": str(comparison_value),
                }
            )

        explanation_results = []
        if comparison_rows:
            # One batch inference instead of one model call per feature.
            batch_df = pd.concat(comparison_rows, ignore_index=True)
            batch_probabilities = model.predict_proba(batch_df)
            churn_class_index = classes.index(1)

            for metadata, row_probabilities in zip(
                comparison_metadata, batch_probabilities
            ):
                compared_probability = float(
                    row_probabilities[churn_class_index]
                )
                difference = compared_probability - churn_probability

                explanation_results.append(
                    {
                        **metadata,
                        "Original churn probability (%)": churn_probability * 100,
                        "Compared churn probability (%)": compared_probability * 100,
                        "Probability change (percentage points)": difference * 100,
                        "Absolute change": abs(difference * 100),
                    }
                )

        if explanation_results:
            explanation_df = pd.DataFrame(explanation_results).sort_values(
                "Absolute change", ascending=False
            )

            st.subheader("Individual Feature Comparisons")
            st.dataframe(
                explanation_df.drop(columns=["Absolute change"]).round(2),
                width="stretch",
                hide_index=True,
            )

            st.subheader("Impact of Changing Individual Features")
            chart_df = explanation_df.set_index("Feature")[
                "Probability change (percentage points)"
            ]
            st.bar_chart(chart_df)

            st.subheader("📝 Explanation in Simple English")
            for result in explanation_df.to_dict(orient="records"):
                feature = result["Feature"]
                current_value = result["Customer value"]
                compared_value = result["Compared value"]
                change = result["Probability change (percentage points)"]

                if change > 0.05:
                    direction = "increased"
                    interpretation = (
                        "the alternative value was associated with a higher "
                        "estimated churn probability in this comparison"
                    )
                elif change < -0.05:
                    direction = "decreased"
                    interpretation = (
                        "the alternative value was associated with a lower "
                        "estimated churn probability in this comparison"
                    )
                else:
                    direction = "barely changed"
                    interpretation = (
                        "the alternative value made little difference to "
                        "the estimated churn probability in this comparison"
                    )

                st.markdown(
                    f"- **{feature}:** The customer's value is **{current_value}**. "
                    f"When it was changed to **{compared_value}**, the estimated "
                    f"churn probability {direction} by **{abs(change):.2f} "
                    f"percentage points**; {interpretation}."
                )

            st.caption(
                "These comparisons are model-based, one-feature-at-a-time "
                "what-if tests. They are not causal effects or SHAP values. "
                "The alternative is a reference value, not necessarily the "
                "best value for every customer."
            )

            # Suggestions are practical actions, not model-proven causes.
            st.subheader("💡 Possible Retention Actions")
            suggestions = {
                "Contract": (
                    "If suitable for the customer, explain the benefits of "
                    "longer-term plans and offer a transparent incentive."
                ),
                "tenure": (
                    "For a newer customer, consider onboarding support and "
                    "a follow-up to check whether their needs are being met."
                ),
                "MonthlyCharges": (
                    "Review the customer's plan and explain available options "
                    "that may better match their usage and budget."
                ),
            }
            for result in explanation_df.to_dict(orient="records"):
                feature = result["Feature"]
                if feature in suggestions:
                    st.markdown(f"- **{feature}:** {suggestions[feature]}")

        else:
            st.info("No feature comparisons could be generated for this customer.")

        # ---------------- CUSTOMER DETAILS ----------------

        with st.expander("View submitted customer details"):
            st.dataframe(customer_df, width="stretch")

    except Exception as error:
        st.error("Prediction failed. See the technical details below.")
        st.exception(error)


# ---------------- GLOBAL FEATURE IMPORTANCE ----------------

st.divider()
st.header("🌍 Global Feature Importance")
st.write(
    "The following features were ranked as important in the earlier model "
    "analysis. The chart shows an ordinal ranking, not measured importance "
    "scores. A higher rank indicates a higher position in that earlier list."
)

importance_data = pd.DataFrame(
    {
        "Feature": [
            "Contract",
            "tenure",
            "TotalCharges",
            "MonthlyCharges",
            "InternetService",
            "MultipleLines",
            "TechSupport",
            "PaymentMethod",
            "OnlineSecurity",
            "StreamingTV",
        ],
        "Importance rank": list(range(10, 0, -1)),
    }
)

st.bar_chart(
    importance_data.set_index("Feature")["Importance rank"],
    horizontal=True,
)

st.subheader("What these features mean")
feature_descriptions = {
    "Contract": "The customer's contract type, such as month-to-month or a longer-term contract.",
    "tenure": "How many months the customer has stayed with the company.",
    "TotalCharges": "The total amount charged to the customer over their recorded relationship.",
    "MonthlyCharges": "The recurring amount billed to the customer each month.",
    "InternetService": "The type of internet service used by the customer.",
    "MultipleLines": "Whether the customer has multiple telephone lines.",
    "TechSupport": "Whether the customer has a technical-support service.",
    "PaymentMethod": "The method the customer uses to pay their bill.",
    "OnlineSecurity": "Whether the customer has an online-security service.",
    "StreamingTV": "Whether the customer has a streaming-TV service.",
}

for feature, description in feature_descriptions.items():
    st.markdown(f"- **{feature}:** {description}")

st.caption(
    "This ranking is based on the earlier analysis and is not recalculated "
    "during app use. Rankings alone do not show the size or direction of a "
    "feature's effect on churn."
)
