# asteroid-hazardous-prediction-
# 🚀 Asteroid Risk Assessment using Machine Learning

## 📌 Overview

This project focuses on predicting whether an asteroid is **hazardous or non-hazardous** using various Machine Learning models. The goal is to assist in early risk detection by analyzing asteroid-related features such as size, velocity, and distance from Earth.

---

## 🎯 Objectives

* Build a classification model to predict asteroid hazard status
* Compare multiple ML algorithms
* Handle class imbalance effectively
* Evaluate models using appropriate metrics like **F1 Score and AUC**

---

## 🧠 Models Used

The following models were implemented and compared:

* Logistic Regression (Baseline model)
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)
* Decision Tree
* XGBoost (Best performing model)

---

## ⚙️ Tech Stack

* **Programming Language:** Python
* **Libraries:**

  * NumPy
  * Pandas
  * Scikit-learn
  * Matplotlib
  * Seaborn
  * XGBoost

---

## 📊 Dataset

* The dataset contains asteroid-related parameters such as:

  * Diameter
  * Relative velocity
  * Miss distance
  * Orbit-related features
* Target variable:

  * **Hazardous (Yes/No)**

---

## 🧹 Data Preprocessing

* Handled missing values
* Feature selection and scaling
* Addressed class imbalance
* Train-test split applied

---

## 🔍 Model Evaluation Metrics

Due to class imbalance, accuracy alone is not reliable.

We used:

* **Precision** → Correct hazardous predictions
* **Recall** → Ability to detect hazardous asteroids
* **F1 Score** → Balance of precision & recall
* **AUC-ROC Curve** → Overall model performance

---

## 📈 Results

| Model               | Accuracy | Precision | Recall | F1 Score | AUC      |
| ------------------- | -------- | --------- | ------ | -------- | -------- |
| Logistic Regression | ~74%     | 0.25      | 0.65   | 0.36     | 0.80     |
| SVM                 | ~80%     | 0.35      | 0.86   | 0.49     | 0.89     |
| KNN (k=5)           | ~89%     | 0.55      | 0.13   | 0.21     | 0.82     |
| Decision Tree       | ~80%     | 0.34      | 0.81   | 0.48     | 0.87     |
| XGBoost             | ~81%     | 0.37      | 0.89   | **0.52** | **0.91** |

---

## 🏆 Best Model

**XGBoost** performed the best because:

* Highest **F1 Score (~52%)**
* Highest **AUC (~0.91)**
* Better handling of class imbalance
* Captures complex patterns effectively

---

## 📉 Why Not Accuracy?

The dataset is imbalanced (more non-hazardous asteroids).

👉 A model predicting all asteroids as non-hazardous can still get high accuracy.
👉 Hence, **F1 Score** is a better metric.

---

## 🔢 KNN – Choosing K Value

* Used **k = 5**
* Selected using **Cross Validation**
* Balanced overfitting (k small) and underfitting (k large)

---

## 📊 Visualizations

* ROC Curve Comparison
* Model Performance Line Graph
* Confusion Matrices for each model

---

## 📌 Conclusion

* Multiple models were evaluated for asteroid hazard prediction
* Accuracy alone was misleading due to class imbalance
* F1 Score and AUC provided better evaluation
* **XGBoost emerged as the most reliable model**

---

## 🔮 Future Improvements

* Use larger and more diverse datasets
* Apply deep learning models
* Perform hyperparameter tuning
* Deploy as a real-time prediction system

---

## 👨‍💻 Author

**Nehal Stay**

---

## 📎 How to Run

1. Clone the repository
2. Install required libraries

   ```bash
   pip install -r requirements.txt
   ```
3. Run the main script

   ```bash
   python main.py
   ```

---

## ⭐ Acknowledgment

This project was developed as part of a Machine Learning academic study to understand classification techniques and model evaluation.

---
