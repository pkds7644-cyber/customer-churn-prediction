# 🛡️ Customer Churn Prediction System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4.1-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-red.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

## 📌 Project Overview
Acquiring a new customer is up to 5 times more expensive than retaining an existing one. This project is an end-to-end Machine Learning pipeline and interactive web application designed to predict customer churn. By identifying at-risk customers before they cancel their subscriptions, businesses can take proactive retention measures.

This project demonstrates a complete ML lifecycle: from Exploratory Data Analysis (EDA) and feature engineering to model hyperparameter tuning and production deployment.

## ✨ Features
* **Interactive Dashboard:** Built with Streamlit for real-time predictions and data visualization.
* **Production-Ready Pipeline:** Utilizes Scikit-Learn `Pipeline` and `ColumnTransformer` to prevent data leakage during scaling and encoding.
* **Business Insights:** Extracts Feature Importance to tell stakeholders *why* customers are leaving.
* **Real-time Inference:** Users can input customer demographics and service details to get an instant churn probability score and risk level.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Data Visualization:** Matplotlib, Seaborn
* **Web Deployment:** Streamlit, Streamlit Community Cloud

## 📊 Dataset Information
* **Source:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* **Description:** Contains 7,043 rows of telecommunications company data.
* **Features:** Includes customer demographics (age, gender, dependents), account information (tenure, contract type, payment method), and subscribed services (internet, streaming, tech support).
* **Target Variable:** `Churn` (Yes/No)

## 🧠 Machine Learning Models Used
Three models were trained and evaluated to find the best balance of interpretability and predictive power:
1. **Logistic Regression:** Used as a highly interpretable baseline.
2. **Decision Tree:** Used to capture non-linear relationships.
3. **Random Forest (Selected):** Optimized using `GridSearchCV` for hyperparameter tuning. Chosen for its robust performance on imbalanced data and its ability to accurately output Feature Importance.

## 📈 Results & Evaluation
Because the dataset is naturally imbalanced (73% of customers stay, 27% leave), accuracy alone is a misleading metric. The model was evaluated based on its ability to identify actual churners:
* **ROC-AUC Score:** 0.84 (Excellent ability to distinguish between classes)
* **Recall:** Optimized to ensure we catch as many at-risk customers as possible, minimizing false negatives.
* **Top Churn Drivers:** Month-to-month contracts, low tenure, and fiber optic internet service were mathematically identified as the highest drivers of customer churn.

## 📂 Project Structure
```text
customer-churn-prediction/
│
├── data/                   # Raw dataset (churn_data.csv)
├── models/                 # Serialized ML pipeline (churn_model.pkl)
├── notebooks/              # Jupyter notebooks for initial EDA 
├── src/                    # Modular Python scripts for training
│   ├── preprocess.py       # Cleaning and Feature Engineering logic
│   └── train.py            # Model training and tuning script
├── app/                    # Streamlit Dashboard application
│   ├── app.py              # Main landing page
│   └── pages/              # Interactive sub-pages (Prediction, Insights)
├── requirements.txt        # Production dependencies
└── README.md               # Project documentation