# Credit Risk Simulator - Streamlit
# Single-file Streamlit app (app.py)
# Purpose: Interactive end-to-end credit modeling simulator for interviews/demos.
# Run: streamlit run app.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
    r2_score,
    mean_squared_error,
    accuracy_score,
)
from sklearn.tree import plot_tree, export_text
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
import io

# optional xgboost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

st.set_page_config(layout="wide", page_title="Credit Risk Simulator")
st.title("💳 Credit Risk Simulator")
st.markdown(
    """
    Interactive demo that generates synthetic borrower data, trains classical and tree-based models,
    and simulates a simplified credit underwriting & pricing workflow.
    
    Github: https://github.com/rathodhrishikesh/hr-credit-risk-simulator/
    """
)

# ------------------------ Sidebar: Data & Model Settings ------------------------
st.sidebar.header("Data generation & model settings")
N = st.sidebar.slider("Number of synthetic borrowers", 500, 20000, 3000, step=500)
SEED = st.sidebar.number_input("Random seed", value=42)
train_size = st.sidebar.slider("Train size ratio", 0.5, 0.9, 0.75, step=0.05)
use_xgb = st.sidebar.checkbox("Use XGBoost (if available)", value=HAS_XGBOOST)

st.sidebar.markdown("---")
st.sidebar.header("Underwriting parameters")
base_rate = st.sidebar.number_input("Base rate (annual %)", value=0.05, min_value=0.0, max_value=0.5, step=0.01)
risk_premium_mult = st.sidebar.number_input("Risk premium multiplier", value=1.0, min_value=0.0)
approval_threshold = st.sidebar.slider("Expected loss approval threshold (USD)", 0.0, 50000.0, 5000.0, step=500.0)

# ------------------------ Synthetic Data Generator ------------------------
@st.cache_data
def generate_synthetic_data(n=3000, seed=42):
    rng = np.random.RandomState(seed)

    # Income: lognormal-ish
    income = np.exp(rng.normal(np.log(60000), 0.6, size=n))
    income = np.clip(income, 10000, 500000)

    # Credit score approx 300-850
    credit_score = np.clip(rng.normal(680, 60, size=n).astype(int), 300, 850)

    # DTI: 0-1 fraction
    dti = np.clip(rng.beta(2.0, 6.0, size=n), 0.0, 1.0)

    # Loan amount correlated with income
    loan_amount = np.clip((income * rng.uniform(0.05, 0.5, size=n)), 1000, 200000)

    # Term months
    loan_term = rng.choice([12, 24, 36, 48, 60], size=n, p=[0.05, 0.2, 0.4, 0.2, 0.15])

    # Synthetic FICO tier
    fico_tier = pd.cut(credit_score, bins=[299, 579, 669, 739, 799, 850], labels=["Poor", "Fair", "Good", "Very Good", "Exceptional"])

    # Latent risk score and default probability
    # lower credit_score increases PD; higher DTI increases PD; low income increases PD; larger loan increases PD
    z = (
        -0.006 * credit_score
        + 3.5 * dti
        - 0.00001 * income
        + 0.000005 * loan_amount
        + rng.normal(0, 0.6, size=n)
    )
    base_pd = 1 / (1 + np.exp(-z))  # logistic

    # Create binary default label
    default_flag = rng.binomial(1, base_pd)

    # Expected loss (LGD) target continuous: assume LGD fraction 0-1 multiplied by loan_amount
    # LGD increases with defaulted flag but also varies with fico and DTI
    lgd_fraction = np.clip(0.2 + 0.5 * (1 - (credit_score - 300) / 550) + 0.6 * dti + rng.normal(0, 0.1, size=n), 0.0, 1.0)
    # for non-defaults, expected_loss = small
    expected_loss = lgd_fraction * loan_amount

    df = pd.DataFrame(
        {
            "income": income,
            "credit_score": credit_score,
            "dti": dti,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "fico_tier": fico_tier.astype(str),
            "pd_true": base_pd,
            "default_flag": default_flag,
            "lgd_fraction": lgd_fraction,
            "expected_loss": expected_loss,
        }
    )
    
    tier_colors = {
        "Poor": "🔴 Poor",
        "Fair": "🟠 Fair",
        "Good": "🟡 Good",
        "Very Good": "🔵 Very Good",
        "Exceptional": "🟢 Exceptional",
    }

    df["fico_tier"] = df["fico_tier"].map(tier_colors)
    
    return df

# Generate data
with st.spinner("Generating synthetic data..."):
    df = generate_synthetic_data(N, SEED)

st.subheader("Sample of generated borrower data")
st.dataframe(df.sample(min(50, len(df)), random_state=SEED).reset_index(drop=True))

st.markdown("""
**Variable Descriptions:**

- **income**: Annual income of the borrower (in USD).  
- **credit_score**: Borrower’s credit score (typically ranges from 300–850).  
- **dti**: Debt-to-Income ratio; higher values indicate higher credit risk.  
- **loan_amount**: Total loan amount applied for (in USD).  
- **loan_term**: Duration of the loan (in months).  
- **fico_tier**: Derived FICO credit score tier category (🔴 Poor; 🟠 Fair; 🟡 Good; 🔵 Very Good; 🟢 Exceptional)
- **pd_true**: True (simulated) probability of default for the borrower.  
- **default_flag**: Binary indicator (1 = default, 0 = repaid).  
- **lgd_fraction**: Loss Given Default, i.e., proportion of loan lost if borrower defaults.  
- **expected_loss**: Product of PD (Probability of Default) × LGD (Loss Given Default) × EAD (Exposure at Default); represents expected financial loss.

""")

# ------------------------ Train/Test Split ------------------------
features = ["income", "credit_score", "dti", "loan_amount", "loan_term"]
X = df[features]
y_class = df["default_flag"]
y_reg = df["expected_loss"]

X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
    X, y_class, y_reg, train_size=train_size, random_state=SEED, stratify=y_class if len(np.unique(y_class))>1 else None
)

# ------------------------ Modeling utilities ------------------------
@st.cache_resource
def train_logistic(X_tr, y_tr):
    # Standardize numeric features for LR
    model = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear", class_weight='balanced'))
    model.fit(X_tr, y_tr)
    return model

@st.cache_resource
def train_linear_reg(X_tr, y_tr):
    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(X_tr, y_tr)
    return model

@st.cache_resource
def train_rf_classifier(X_tr, y_tr):
    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=SEED)
    model.fit(X_tr, y_tr)
    return model

@st.cache_resource
def train_rf_regressor(X_tr, y_tr):
    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=SEED)
    model.fit(X_tr, y_tr)
    return model

@st.cache_resource
def train_xgb_classifier(X_tr, y_tr):
    if not HAS_XGBOOST:
        return None
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=SEED, n_estimators=200, max_depth=6)
    model.fit(X_tr, y_tr)
    return model

@st.cache_resource
def train_xgb_regressor(X_tr, y_tr):
    if not HAS_XGBOOST:
        return None
    model = xgb.XGBRegressor(random_state=SEED, n_estimators=200, max_depth=6)
    model.fit(X_tr, y_tr)
    return model

# Train models
with st.spinner("Training models..."):
    logreg = train_logistic(X_train, y_train_class)
    linreg = train_linear_reg(X_train, y_train_reg)
    rf_clf = train_rf_classifier(X_train, y_train_class)
    rf_reg = train_rf_regressor(X_train, y_train_reg)
    xgb_clf = train_xgb_classifier(X_train, y_train_class) if use_xgb and HAS_XGBOOST else None
    xgb_reg = train_xgb_regressor(X_train, y_train_reg) if use_xgb and HAS_XGBOOST else None

# ------------------------ Module 1-4: placed into tabs ------------------------
tabs = st.tabs(
    [
        "1️⃣  Logistic Regression",
        "2️⃣  Linear Regression",
        "3️⃣  Ensemble Models",
        "4️⃣  Underwriting & Pricing Simulator",
    ]
)

# Module 1 — Logistic Regression (Tab 0)
with tabs[0]:
    st.header("Module 1 — Logistic Regression: Default Prediction")
    st.info("""
    Predicts the probability that a borrower will default on a loan.  
      
    **Target variable:** default_flag (binary: 1 = Default, 0 = Repaid)  
    **Model type:** Classification  
    **Output:** Probability of Default (PD)
    """)
    
    col1, col2 = st.columns([1.5, 2.5])

    # Predictions & metrics
    y_pred_proba = logreg.predict_proba(X_test)[:, 1]
    y_pred = logreg.predict(X_test)

    cm = confusion_matrix(y_test_class, y_pred)
    roc_fpr, roc_tpr, _ = roc_curve(y_test_class, y_pred_proba)
    roc_auc = auc(roc_fpr, roc_tpr)

    with col1:
        st.subheader("Metrics & confusion matrix")
        fig_cm, ax = plt.subplots()
        im = ax.matshow(cm, cmap="Blues")
        for (i, j), z in np.ndenumerate(cm):
            ax.text(j, i, z, ha="center", va="center")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig_cm)
        
         # --- New Section: TP, FP, TN, FN, TPR, FPR ---
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        st.markdown(f"""
        **True Positive (TP):** {tp}  
        **False Positive (FP):** {fp}  
        **True Negative (TN):** {tn}  
        **False Negative (FN):** {fn}  
        """)

        st.markdown(f"""
        **True Positive Rate (TPR / Recall)** = TP / (TP + FN)  
        **False Positive Rate (FPR)** = FP / (FP + TN)
        """)

    with col2:
        st.subheader("ROC Curve")
        fig_roc, ax = plt.subplots()
        ax.plot(roc_fpr, roc_tpr, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig_roc)
        
        st.write("AUC: %.3f" % roc_auc)
        st.write("Accuracy: %.3f" % accuracy_score(y_test_class, y_pred))
   
    # Coefficients
    st.subheader("Logistic Regression — Coefficient importance")
    # extract coefficients from pipeline
    coef = logreg.named_steps["logisticregression"].coef_[0]
    scaler = logreg.named_steps["standardscaler"]
    # For standardized effect-size visualization, use coefficients
    coef_df = pd.DataFrame({"feature": features, "coef": coef})
    coef_df = coef_df.sort_values("coef")
    fig_coef, ax = plt.subplots(figsize=(6, 3))
    ax.barh(coef_df["feature"], coef_df["coef"])
    ax.set_title("Logistic Coefficients")
    st.pyplot(fig_coef)


# Module 2 — Linear Regression (Tab 1)
with tabs[1]:
    st.header("Module 2 — Multivariate Linear Regression: Expected Loss")
    st.info("""
    Predicts the expected financial loss for each borrower.  
      
    **Target variable:** expected_loss (continuous)  
    **Model type:** Regression  
    **Output:** Expected Loss
    """)

    col3, col4 = st.columns(2)

    # Predictions
    y_pred_reg = linreg.predict(X_test)
    r2 = r2_score(y_test_reg, y_pred_reg)
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))

    with col3:
        st.subheader("Actual vs Predicted")
        fig_scatter, ax = plt.subplots()
        ax.scatter(y_test_reg, y_pred_reg, alpha=0.4)
        ax.plot(
            [y_test_reg.min(), y_test_reg.max()],
            [y_test_reg.min(), y_test_reg.max()],
            "--",
            color="gray",
        )
        ax.set_xlabel("Actual Expected Loss")
        ax.set_ylabel("Predicted Expected Loss")
        st.pyplot(fig_scatter)
        st.write(f"R²: {r2:.3f} | RMSE: {rmse:.2f}")

    with col4:
        st.subheader("Residual distribution")
        resid = y_test_reg - y_pred_reg
        fig_resid, ax = plt.subplots()
        ax.hist(resid, bins=40)
        ax.set_xlabel("Residual (Actual - Predicted)")
        st.pyplot(fig_resid)
        st.write("Impact of credit score and DTI on pricing (partial view):")

        sample = pd.DataFrame(
            {
                "credit_score": np.linspace(350, 820, 50),
                "dti": np.repeat(X_test["dti"].median(), 50),
                "income": np.repeat(X_test["income"].median(), 50),
                "loan_amount": np.repeat(X_test["loan_amount"].median(), 50),
                "loan_term": np.repeat(X_test["loan_term"].median(), 50),
            }
        )

        # Make sure column order and names match the training data
        sample = sample[X_train.columns]
        pred_by_score = linreg.predict(sample)

        fig_effect, ax = plt.subplots()
        ax.plot(sample["credit_score"], pred_by_score)
        ax.set_xlabel("Credit Score")
        ax.set_ylabel("Predicted Expected Loss")
        st.pyplot(fig_effect)


# Module 3 — Ensemble Models (Tab 2)
with tabs[2]:
    st.header("Module 3 — Ensemble Models: Random Forest vs XGBoost")
    st.info("""
    Compares ensemble models for default prediction.  
      
    **Target variable:** default_flag (binary)  
    **Model type:** Classification  
    **Output:** Feature Importance & Model performance metrics (AUC, Accuracy)
    """)
    col5, col6 = st.columns(2)

    # RF metrics
    rf_proba = rf_clf.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test_class, rf_proba)
    rf_acc = accuracy_score(y_test_class, rf_clf.predict(X_test))

    with col5:
        st.subheader("Random Forest")
        st.write(f"AUC: {rf_auc:.3f}")
        st.write(f"Accuracy: {rf_acc:.3f}")
        # Feature importance
        fi = pd.Series(rf_clf.feature_importances_, index=features).sort_values(
            ascending=True
        )
        fig_fi, ax = plt.subplots()
        ax.barh(fi.index, fi.values)
        ax.set_title("RF Feature Importance")
        st.pyplot(fig_fi)

    with col6:
        st.subheader("XGBoost (if available)")
        if xgb_clf is not None:
            xgb_proba = xgb_clf.predict_proba(X_test)[:, 1]
            xgb_auc = roc_auc_score(y_test_class, xgb_proba)
            xgb_acc = accuracy_score(y_test_class, xgb_clf.predict(X_test))
            st.write(f"AUC: {xgb_auc:.3f}")
            st.write(f"Accuracy: {xgb_acc:.3f}")
            try:
                xgb_fi = pd.Series(xgb_clf.feature_importances_, index=features).sort_values(
                    ascending=True
                )
                fig_xgb_fi, ax = plt.subplots()
                ax.barh(xgb_fi.index, xgb_fi.values)
                ax.set_title("XGB Feature Importance")
                st.pyplot(fig_xgb_fi)
            except Exception:
                st.write("Feature importance unavailable")
        else:
            st.info("XGBoost not installed or disabled. Using Random Forest for comparisons.")

    # Side-by-side importance comparison
    st.subheader("Feature importance comparison")
    fi_rf = pd.Series(rf_clf.feature_importances_, index=features)
    fi_xgb = pd.Series(xgb_clf.feature_importances_, index=features) if xgb_clf is not None else fi_rf
    fi_compare = pd.DataFrame({"RandomForest": fi_rf, "XGBoost": fi_xgb})
    st.dataframe(fi_compare.sort_values("RandomForest", ascending=False))

    # decision tree snippet
    st.subheader("Decision tree snippet (one tree from RF)")
    try:
        estimator = rf_clf.estimators_[0]
        fig_tree, ax = plt.subplots(figsize=(12, 6))
        plot_tree(estimator, feature_names=features, max_depth=3, filled=True, fontsize=8, ax=ax)
        st.pyplot(fig_tree)
        st.markdown("**Text representation (top levels):**")
        st.code(export_text(estimator, feature_names=features, max_depth=3))
    except Exception as e:
        st.write("Could not render tree: ", e)


# Module 4 — Underwriting & Pricing Simulator (Tab 3)
with tabs[3]:
    st.header("Module 4 — Credit Underwriting & Loan Pricing Simulator")
    st.info("""
    Simulates loan approval and pricing decisions using model predictions.

    **Outputs:**  
    • Expected Loss (EL) = PD × LGD × EAD  
    • Portfolio-level KPIs segmented by FICO Tier
    """)

    # Compute PD from logistic, LGD from linear regression (as fraction of loan amount)
    pd_all = logreg.predict_proba(df[features])[:, 1]
    pred_lgd_amount = linreg.predict(df[features])
    # Convert predicted expected loss to LGD fraction by dividing by loan_amount; guard against div0
    pred_lgd_fraction = np.clip(pred_lgd_amount / (df["loan_amount"] + 1e-9), 0.0, 1.0)

    # EAD = loan_amount (simplified)
    ead = df["loan_amount"]
    expected_loss_sim = pd.Series(pd_all * pred_lgd_fraction * ead, name="expected_loss_sim")

    sim_df = df.copy()
    sim_df["pd_model"] = pd_all
    sim_df["lgd_fraction_model"] = pred_lgd_fraction
    sim_df["expected_loss_sim"] = expected_loss_sim

    # interactive threshold
    st.subheader("Portfolio-level simulation")
    thresh = st.slider(
        "Approval threshold (expected loss in USD)",
        0.0,
        float(df["loan_amount"].max()),
        float(approval_threshold),
        step=100.0,
    )

    sim_df["approved"] = sim_df["expected_loss_sim"] < thresh
    approve_rate = sim_df["approved"].mean()
    avg_loss = sim_df["expected_loss_sim"].mean()
    avg_interest = base_rate + risk_premium_mult * (sim_df["pd_model"] * 0.2)  # simple risk premium

    col7, col8, col9 = st.columns([1, 1, 1])
    with col7:
        st.metric("Approval rate", f"{approve_rate:.1%}")

    with col8:
        st.metric("Avg offered interest (annual %)", f"{(avg_interest.mean() * 100):.2f}%")
    
    with col9:
        st.metric("Average expected loss (USD)", f"{avg_loss:,.2f}")

    col10, col11 = st.columns([1, 1])    
    with col10:
        # Histogram
        st.subheader("Expected Loss distribution and approval segmentation")
        fig_hist, ax = plt.subplots()
        ax.hist(
            [
                sim_df.loc[sim_df["approved"], "expected_loss_sim"],
                sim_df.loc[~sim_df["approved"], "expected_loss_sim"],
            ],
            bins=50,
            stacked=True,
            label=["Approved", "Rejected"],
        )
        ax.set_xlabel("Expected loss (USD)")
        ax.set_ylabel("Number of loans")
        ax.legend()
        st.pyplot(fig_hist)
    
    with col11:
        # Table of portfolio KPIs by FICO tier
        st.subheader("Portfolio KPIs by FICO tier")
        kpi = sim_df.groupby("fico_tier").apply(
            lambda d: pd.Series(
                {
                    "count": len(d),
                    "approval_rate": d["approved"].mean(),
                    "avg_expected_loss": d["expected_loss_sim"].mean(),
                    "avg_pd": d["pd_model"].mean(),
                }
            )
        ).reset_index().sort_values("approval_rate", ascending=False)  # ✅ Sort by approval rate descending
        st.dataframe(kpi)

        # Allow downloading the simulated portfolio
        @st.cache_data
        def to_csv_download(df_in):
            return df_in.to_csv(index=False).encode("utf-8")

        csv = to_csv_download(sim_df.head(10000))
        st.download_button(
            "Download simulated portfolio (first 10k rows)",
            data=csv,
            file_name="simulated_portfolio.csv",
            mime="text/csv",
        )

# ------------------------ Wrap up / Notes ------------------------
st.markdown("---")
st.markdown(
    "Built by Hrishikesh Rathod\n"
    "- Email ID: hrfoster@uw.edu\n"
    "- LinkedIn: https://www.linkedin.com/in/hrishikesh-rathod/\n"
)