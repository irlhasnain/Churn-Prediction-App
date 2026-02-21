# 📉 Customer Churn Prediction App

A complete end-to-end Machine Learning project that predicts whether a customer is likely to churn, built with Python, Scikit-learn, and deployed as an interactive web application using Streamlit.

---

## 🚀 Live Demo

> Run the app locally by following the setup instructions below.

---

## 📌 Project Overview

Customer churn is one of the biggest challenges businesses face. Losing a customer is far more expensive than retaining one. This project uses historical customer data to train a Machine Learning model that identifies customers at high risk of churning — enabling businesses to take proactive retention steps.

This project covers the **full ML pipeline**:
- Data loading & exploration
- Data preprocessing & feature engineering
- Model training & evaluation
- Saving the trained model
- Deploying as an interactive web app

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| Pandas & NumPy | Data manipulation & analysis |
| Scikit-learn | ML model training & evaluation |
| Streamlit | Web app deployment |
| Matplotlib / Seaborn | Data visualization |
| Jupyter Notebook | Exploratory Data Analysis (EDA) |

---

## 📁 Project Structure

```
Churn-Prediction-App/
│
├── app.py                    # Streamlit web application
├── notebook.ipynb            # EDA, preprocessing & model training
├── model.pkl                 # Trained ML model (serialized)
├── scaler.pkl                # Fitted scaler for input normalization
├── customer_churn_data.csv   # Dataset used for training
└── README.md                 # Project documentation
```

---

## ⚙️ How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/irlhasnain/Churn-Prediction-App.git
cd Churn-Prediction-App
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

> If you don't have a requirements.txt, install manually:
```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn
```

**3. Run the Streamlit app**
```bash
streamlit run app.py
```

**4. Open your browser** at `http://localhost:8501`

---

## 🧠 Model Details

- **Algorithm:** Random Forest Classifier (/ Logistic Regression — update as applicable)
- **Target Variable:** `Churn` (1 = Churned, 0 = Retained)
- **Key Features Used:** Contract type, tenure, monthly charges, payment method, internet service, etc.
- **Preprocessing:** Standard scaling applied via `scaler.pkl`

---

## 📊 Model Performance

> **Best Model:** SVM (C=0.01, Linear Kernel) — selected via GridSearchCV (5-fold cross-validation)

| Metric | Score |
|--------|-------|
| Accuracy | 84% |
| Precision (Churn Class) | 84% |
| Recall (Churn Class) | 100% |
| F1 Score (Churn Class) | 91% |
| Weighted Avg F1 | 77% |

> ⚠️ **Note:** The model predicts churn (class 1) with high recall (100%) — meaning it catches almost every customer who will churn. This is intentional, as missing a churner is more costly for a business than a false alarm.

---

## 💡 Key Learnings

- Built a real-world ML pipeline from raw data to deployed application
- Handled class imbalance and feature encoding for categorical variables
- Serialized trained model using `pickle` for deployment
- Created an intuitive UI with Streamlit for non-technical users

---

## 🙋‍♂️ Author

**Hasnain Khan**  
Data Science Aspirant | Python • ML • SQL • Power BI  
📍 Bhopal, Madhya Pradesh, India

[![LinkedIn](https://www.linkedin.com/posts/hasnain-khan-9a3004326_datascience-machinelearning-python-activity-7427190073437196289-sD8M?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFI6AIsB3s1K7WDP-6V5_ahEkBOPYKI6Jt4)
[![GitHub](https://github.com/irlhasnain)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
