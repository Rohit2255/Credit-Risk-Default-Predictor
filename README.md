# 🧠 Credit Risk Default Predictor

This project is part of my **#100DaysOfDataScience** challenge.

It predicts the likelihood of a customer defaulting on their credit card payment based on historical financial behavior using classical ML techniques. A strong emphasis is placed on solving **class imbalance** using **SMOTE** and evaluating the model with metrics beyond accuracy.

---

## 📁 Dataset

- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Observations:** 284,807 transactions
- **Fraud Cases:** 492  
- **Features:**  
  - 28 anonymized PCA features (V1 to V28)  
  - `Time`, `Amount`, `Class` (target)

---

## ✅ Tech Stack

- Python 🐍
- Pandas & NumPy
- Scikit-learn
- Matplotlib & Seaborn
- Imbalanced-learn (SMOTE)
- Joblib (model serialization)

---

## 🚀 Workflow

1. **Exploratory Data Analysis**
   - Distribution of fraud vs normal
   - Correlation analysis
2. **Preprocessing**
   - Feature scaling (StandardScaler)
   - Train-test split (stratified)
3. **SMOTE**  
   - Synthetic Minority Oversampling for balancing fraud class
4. **Modeling**
   - Random Forest Classifier
   - Evaluation using classification report and ROC AUC
5. **Model Export**
   - Saved model, scaler, and feature list using `joblib`

---

## 📊 Performance

| Metric               | Value     |
|----------------------|-----------|
| **ROC AUC Score**    | 0.913     |
| **F1 Score (Fraud)** | 0.83      |
| **Precision (Fraud)**| 0.84      |
| **Accuracy**         | ~99.8%    |

---

## 📂 Output Files

- `credit_default_model.pkl` — Trained Random Forest model  
- `scaler.pkl` — StandardScaler used for feature scaling  
- `feature_columns.pkl` — List of features used during training  

---

## 📌 Key Takeaways

- Addressing class imbalance is **crucial** in fraud and credit risk modeling.
- Accuracy alone is misleading — metrics like **ROC-AUC, Precision, Recall, F1-score** matter more.
- With proper preprocessing and model tuning, traditional ML can be very effective!

---

## ✍️ Author

**Rohit Kumar Yadav**  
🔗 [LinkedIn](https://www.linkedin.com/in/rohit-kumar-yadav-b97360194/)  
🌱 #100DaysOfDataScience

---

## ⭐ What's Next?

- Deep learning approach with Autoencoders and Anomaly Detection  
- Streamlit web app for user interaction  
- Deployment via Flask/Streamlit + Docker

---

## 📌 Note

This is NOT the same as my earlier **Credit Card Fraud Detection** project — that was focused on binary fraud classification. This one is oriented toward **credit risk default prediction**, model serialization, and real-world deployment planning.

---

