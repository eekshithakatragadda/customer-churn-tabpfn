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

# ---------------- MODERN UI THEME ----------------
st.markdown(
    """
    <style>
    :root {
      --ink: #172033;
      --muted: #697586;
      --line: #e7ebf2;
      --surface: #ffffff;
      --canvas: #f5f7fb;
      --accent: #536dfe;
      --accent-dark: #4055db;
      --teal: #16b8a6;
    }
    .stApp { background: var(--canvas); color: var(--ink); }
    .block-container { max-width: 1420px; padding-top: 1.7rem; padding-bottom: 3rem; }
    h1, h2, h3 { letter-spacing: -0.035em; color: var(--ink); }
    h1 { font-weight: 760 !important; }
    h2 { font-weight: 720 !important; }
    p, li { color: #485467; }
    [data-testid="stSidebar"] {
      background: #111a2e;
      border-right: 1px solid #202b43;
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.4rem; }
    [data-testid="stSidebar"] * { color: #eaf0ff; }
    [data-testid="stSidebar"] hr { border-color: #2a3550; }
    .brand-lockup {
      display: flex; align-items: center; gap: 12px; padding: 7px 3px 22px;
    }
    .brand-mark {
      width: 44px; height: 44px; border-radius: 14px;
      background: linear-gradient(135deg, #6678ff, #22c9b5);
      display: flex; align-items: center; justify-content: center;
      color: white; font-size: 23px; font-weight: 800;
      box-shadow: 0 8px 22px rgba(83,109,254,.25);
    }
    .brand-title { font-size: 17px; font-weight: 750; line-height: 1.2; color: #fff; }
    .brand-subtitle { color: #a5b2cc; font-size: 11px; margin-top: 5px; letter-spacing: .08em; text-transform: uppercase; }
    .nav-label { color: #8998b6; font-size: 10px; font-weight: 800; letter-spacing: .15em; margin: 10px 0 8px; }
    [data-testid="stSidebar"] [data-testid="stButton"] button {
      width: 100%; min-height: 45px; border-radius: 12px;
      border: 1px solid transparent; background: transparent;
      color: #b9c5dd; text-align: left; justify-content: flex-start;
      padding: 0 14px; font-weight: 600; transition: all .18s ease;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button:hover {
      background: #202c47; border-color: #344363; color: #fff;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] {
      background: linear-gradient(100deg, #536dfe, #4b61e8);
      border-color: #7284ff; color: white;
      box-shadow: 0 7px 18px rgba(60,82,210,.24);
    }
    .sidebar-note {
      margin-top: 26px; padding: 13px; border: 1px solid #2b3854;
      border-radius: 14px; background: #17233b; color: #aebbd4;
      font-size: 12px; line-height: 1.55;
    }
    .eyebrow { color: var(--accent); font-size: 11px; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; margin-bottom: 7px; }
    .hero-title { font-size: clamp(28px, 3vw, 39px); font-weight: 800; letter-spacing: -.045em; line-height: 1.12; color: #172033; }
    .hero-copy { font-size: 15px; color: #68758a; margin-top: 9px; margin-bottom: 20px; max-width: 760px; }
    .hero-panel {
      background: linear-gradient(120deg, #fff 0%, #f0f3ff 100%);
      border: 1px solid #e5e9fb; border-radius: 22px;
      padding: 25px 28px; margin: 3px 0 24px;
      box-shadow: 0 8px 28px rgba(27, 42, 83, .045);
    }
    .section-kicker { color: #758198; font-size: 11px; font-weight: 800; letter-spacing: .13em; text-transform: uppercase; margin-bottom: 6px; }
    div[data-testid="stMetric"] {
      background: white; border: 1px solid var(--line); border-radius: 16px;
      padding: 17px 18px; box-shadow: 0 4px 16px rgba(20,35,70,.035);
    }
    div[data-testid="stMetricLabel"] { color: #667085; font-weight: 650; }
    div[data-testid="stMetricValue"] { color: #172033; font-weight: 780; }
    div[data-testid="stForm"] {
      background: white; border: 1px solid var(--line);
      border-radius: 18px; padding: 1.25rem 1.35rem;
      box-shadow: 0 5px 20px rgba(20,35,70,.035);
    }
    div[data-testid="stFormSubmitButton"] button, .stButton > button[kind="primary"] {
      background: linear-gradient(100deg, #536dfe, #4055db);
      border: 0; border-radius: 11px; color: #ffffff !important; font-weight: 700;
      min-height: 44px; box-shadow: 0 6px 16px rgba(83,109,254,.18);
    }
    div[data-testid="stFormSubmitButton"] button:hover, .stButton > button[kind="primary"]:hover {
      background: #4055db; color: white; border: 0;
    }
    .stSelectbox div[data-baseweb="select"] > div, .stNumberInput input {
      border-radius: 10px !important;
    }
    div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 13px; overflow: hidden; }
    div[data-testid="stAlert"] { border-radius: 13px; }
    hr { border-color: #e5e9f1; }
    .result-card {
      border-radius: 18px; padding: 20px 22px; border: 1px solid #dceee9;
      background: linear-gradient(135deg, #f0fdf8, #fff);
    }
    .result-label { color: #537568; font-size: 11px; font-weight: 800; letter-spacing: .13em; text-transform: uppercase; }
    .result-value { font-size: 25px; font-weight: 800; color: #137c67; margin-top: 7px; }
    @media (max-width: 768px) {
      .block-container { padding-top: 1rem; }
      .hero-panel { padding: 20px; }
    }
    div[data-testid="stFormSubmitButton"] button,
div[data-testid="stFormSubmitButton"] button p,
div[data-testid="stFormSubmitButton"] button span {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_PATH = BASE_DIR / "customer_churn.csv"

# ---------------- CUSTOM SIDEBAR NAVIGATION ----------------
if "active_page" not in st.session_state:
    st.session_state.active_page = "Prediction Dashboard"

with st.sidebar:
    st.markdown(
        """
        <div class="brand-lockup">
          <div class="brand-mark">↗</div>
          <div>
            <div class="brand-title">Churn Intelligence</div>
            <div class="brand-subtitle">Customer analytics</div>
          </div>
        </div>
        <div class="nav-label">WORKSPACE</div>
        """,
        unsafe_allow_html=True,
    )
    nav_items = [
        ("Prediction Dashboard", "Prediction dashboard"),
        ("Model Performance", "Model performance"),
        ("About Project", "About the project"),
    ]
    for page_name, button_label in nav_items:
        if st.button(
            button_label,
            key=f"nav_{page_name}",
            type="primary" if st.session_state.active_page == page_name else "secondary",
            use_container_width=True,
        ):
            st.session_state.active_page = page_name
            st.rerun()
    
page = st.session_state.active_page

# Page-wide header
if page == "Prediction Dashboard":
    st.markdown(
        """
        <div class="hero-panel">
          <div class="eyebrow">CUSTOMER ANALYTICS · OVERVIEW</div>
          <div class="hero-title">Predict churn before it happens.</div>
          <div class="hero-copy">Estimate customer churn risk with TabPFN, then explore individual what-if comparisons and practical retention ideas.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif page == "Model Performance":
    st.markdown(
        """
        <div class="hero-panel">
          <div class="eyebrow">MODEL EVALUATION</div>
          <div class="hero-title">Performance at a glance.</div>
          <div class="hero-copy">Compare the saved evaluation results for TabPFN, Logistic Regression and Random Forest.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="hero-panel">
          <div class="eyebrow">PROJECT OVERVIEW</div>
          <div class="hero-title">The thinking behind the model.</div>
          <div class="hero-copy">Explore the dataset, workflow, explainability approach and current limitations.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
except Exception as error:
    st.error("Could not load the model or dataset.")
    st.exception(error)
    st.stop()




# ---------------- MODEL PERFORMANCE PAGE ----------------
def show_model_performance():
    st.markdown('<div class="section-kicker">EVALUATION SUMMARY</div>', unsafe_allow_html=True)
    st.subheader("Compare the three models")
    st.write(
        "Review the saved test-set results across complementary classification metrics."
    )
    st.caption("Metrics are shown as percentages. Higher values are not equally meaningful for every metric; interpret them together.")
    results_path = BASE_DIR / "performance_results.json"
    try:
        with open(results_path, "r", encoding="utf-8") as file:
            results = json.load(file)

        metrics_df = pd.DataFrame(results["models"]).T
        metrics_df.index.name = "Algorithm"
        metric_names = {
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1_score": "F1-score",
            "roc_auc": "ROC-AUC",
        }
        display_df = metrics_df[list(metric_names)].rename(columns=metric_names)
        st.subheader("Algorithm Comparison")
        st.dataframe((display_df * 100).round(2).astype(str) + "%", use_container_width=True)

        st.subheader("Performance Comparison")
        st.bar_chart(display_df * 100)

        st.subheader("TabPFN Evaluation Metrics")
        tabpfn = results["models"]["TabPFN"]
        metric_cols = st.columns(5)
        for col, (key, label) in zip(metric_cols, metric_names.items()):
            col.metric(label, f"{tabpfn[key] * 100:.2f}%")

        st.subheader("TabPFN Confusion Matrix")
        cm = results["tabpfn_confusion_matrix"]
        confusion_df = pd.DataFrame(
            cm,
            index=["Actual Stay", "Actual Churn"],
            columns=["Predicted Stay", "Predicted Churn"],
        )
        st.dataframe(confusion_df, use_container_width=True)
        st.caption(
            f"Evaluation set: {results.get('test_samples', 'unknown')} customers. "
            "Metrics are shown as percentages. Precision, recall and F1-score "
            "refer to the churn class as recorded in the evaluation notebook."
        )
        st.info(
            "Interpretation: TabPFN has the highest recorded ROC-AUC and precision; "
            "Logistic Regression has the highest recorded accuracy, recall and F1-score. "
            "No model is best on every metric."
        )
    except FileNotFoundError:
        st.warning("performance_results.json was not found. Place it beside app.py.")
    except (ValueError, KeyError, TypeError) as error:
        st.error(f"Could not display model performance: {error}")


# ---------------- ABOUT PROJECT PAGE ----------------
def show_about_project():
    st.markdown('<div class="section-kicker">PROJECT NOTES</div>', unsafe_allow_html=True)
    st.markdown(
        """
### Customer Churn Prediction Using TabPFN with Explainability

**Problem statement**  
Customer churn occurs when customers stop using a company's services. Predicting
which customers may leave can help a business prioritize retention efforts.

**Dataset**  
This project uses the Telco Customer Churn dataset. It contains customer
characteristics, service subscriptions, contract details, tenure, billing
information and the churn target. The customer ID is not used as a predictive feature.

**Methodology**
1. Load and clean the customer data, including converting `TotalCharges` to numeric values.
2. Prepare the input features and churn target.
3. Use TabPFN as the main prediction model.
4. Compare saved test-set metrics for TabPFN, Logistic Regression and Random Forest.
5. Explain an individual prediction with one-feature-at-a-time what-if comparisons.

**Explainability**  
The dashboard changes selected values (Contract, tenure and MonthlyCharges) while
keeping the other submitted values fixed, then compares estimated churn probabilities.
These are model-based what-if comparisons, not SHAP values and not proof of causation.
The global feature chart displays an earlier ordinal ranking, not measured importance scores.

**Limitations**
- Predictions are estimates, not guarantees of customer behavior.
- Results depend on the dataset and evaluation setup.
- A reference value in a what-if comparison is not necessarily the best choice for a customer.
- Retention suggestions are general ideas and should be reviewed in the customer's context.
"""
    )


if page == "Prediction Dashboard":
    # ---------------- CUSTOMER INPUT FORM ----------------

    st.markdown('<div class="section-kicker">PREDICTION WORKSPACE</div>', unsafe_allow_html=True)
    st.subheader("Assess an individual customer")
    st.write("Enter customer details to estimate churn risk and explore selected what-if comparisons.")

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
        st.markdown("### Customer profile")
        st.caption("Choose the closest available customer details, then run the prediction.")
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

        submitted = st.form_submit_button("Predict Churn", width="stretch")


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
            st.markdown('<div class="section-kicker">MODEL OUTPUT</div>', unsafe_allow_html=True)
            st.subheader("Prediction result")
            result_col1, result_col2 = st.columns([1.15, 1])

            with result_col1:
                if prediction == 1:
                    st.error("**Likely to churn** · The model predicts the churn class.")
                else:
                    st.success("**Likely to stay** · The model predicts the stay class.")

            with result_col2:
                st.metric(
                    label="Estimated churn probability",
                    value=f"{churn_probability * 100:.2f}%",
                    help="Probability estimated by the saved TabPFN model.",
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
            st.header("Why did the model make this prediction?")
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

                st.subheader("Explanation")
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
                st.subheader("Possible Retention Actions")
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

elif page == "Model Performance":
    show_model_performance()

elif page == "About Project":
    show_about_project()
