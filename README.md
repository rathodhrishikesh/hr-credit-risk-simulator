# 💳 Credit Risk Simulator

A **Streamlit-based interactive simulator** demonstrating how **AI and machine learning models** are applied in **financial risk analytics** — from **credit underwriting** and **loan pricing** to **portfolio risk management**.

This tool generates **synthetic borrower data**, trains multiple **ML models** (Logistic Regression, Linear Regression, Random Forest, XGBoost), and simulates real-world lending workflows such as **expected loss estimation**, **loan approval**, and **dynamic pricing**.

---

## 🚀 Features

### 🔹 Module 1 — Default Prediction
Predict whether a borrower will default or not using **Logistic Regression**.

- **Model Type:** Classification  
- **Output:** *Probability of Default (PD)*  
- **Example:** Logistic Regression predicts PD = 0.12 → 12% chance of default 

---

### 🔹 Module 2 — Loan Pricing
Estimate borrower-specific expected loss using **Multivariate Linear Regression**.

- **Model Type:** Regression  
- **Output:** *Loss Given Default (LGD)* (continuous)  
- **Example:** Predicted Expected Loss = \$900 for a \$10,000 loan → LGD = 9%

---

### 🔹 Module 3 — Ensemble Models
Benchmark and interpret advanced tree-based models (**Random Forest**, **XGBoost**).

- **Goal:** Capture non-linear borrower behavior missed by linear models  
- **Outputs:** Feature Importance, Decision Tree Visualization, AUC/Accuracy Comparison  
- **Use Case:** Identify non-linear variables (e.g., credit score × DTI interaction) driving default risk  

---

### 🔹 Module 4 — Underwriting & Pricing Simulator
Combine model outputs to simulate **loan approval** and **pricing decisions**.

**Inputs**
- PD = Probability of Default (from Logistic Regression)  
- LGD = Loss Given Default (from Linear Regression)  
- EAD = Exposure at Default (Loan Amount)

**Formula**
> Expected Loss = PD × LGD × EAD  

**Example**
> PD = 0.12, LGD = 0.09, EAD = \$10,000 →  
> Expected Loss = 0.12 × 0.09 × 10,000 = **\$108**

**Module Usage**
- Adjustable approval threshold slider  
- Portfolio-level KPIs by FICO tier  
- Approval rate, average loss, and risk-adjusted interest rate visualization  

### 🖼 Project Mind Map
![Credit Risk Simulator](public/Credit%20Risk%20Simulator.png)
---

## 🎨 Tech Stack

| Component | Technology |
|------------|-------------|
| **UI / Frontend** | Streamlit |
| **Data & ML** | Python (NumPy, pandas, scikit-learn, XGBoost) |
| **Visualization** | Matplotlib |
| **Storage / Export** | CSV export of simulated portfolios |

---

## 📊 Analytics & ML Highlights

| Technique | Purpose |
|------------|----------|
| **Logistic Regression** | Predict probability of default (PD) |
| **Linear Regression** | Predict expected loss (LGD) |
| **Random Forest / XGBoost** | Capture complex borrower behavior and benchmark models |
| **Simulation Layer** | Combine PD, LGD, and EAD to estimate portfolio risk |

---

## 🧠 Learning Objectives

This simulator is ideal for:

- Demonstrating **credit risk modeling workflows**
- Understanding **how different ML models contribute** to credit underwriting decisions  
- Exploring **AI-driven pricing and portfolio management** for lending institutions  

---

## 🌐 Try it Out

Check out the live app here: [Credit Risk Simulator](https://hr-credit-risk-simulator.streamlit.app/)
