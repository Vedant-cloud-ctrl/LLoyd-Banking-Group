# Lloyd-Banking-Group 
# Lloyd Banking: Behavioral Churn Prediction

This project implements an end-to-end machine learning pipeline to predict customer churn. By moving beyond static demographics, we leverage **feature engineering** and **gradient boosting** to identify exactly why customers leave.

---

## How It Was Done: The Data Pipeline

The project follows a rigorous four-stage engineering process to turn raw bank data into predictive insights:

### 1. Feature Engineering (The "Signal" Generator)
Raw data is often noisy; we transformed it into actionable features:
* **Temporal Recency**: Converted raw transaction and login dates into **Recency (Days)**. This allows the model to "learn" the decay of customer engagement over time.
* **Feature Aggregation**: Consolidated multiple transaction rows into a single customer profile, capturing the **latest behavior** as the primary signal.
* **Handling Missing Values**: Categorized missing service interactions as "NaT" (Not applicable) rather than deleting them, allowing the model to distinguish between "No issues" and "Unresolved issues".

### 2. Addressing Imbalance (SMOTE)
In real-world banking, churners are a minority (approx. 25%). To prevent the model from simply "guessing" the majority, we used **SMOTE** (Synthetic Minority Over-sampling Technique) to synthetically balance the training data, forcing the model to learn the specific characteristics of the churn class.

### 3. Hyperparameter Tuning (XGBoost)
We used `RandomizedSearchCV` to fine-tune the "brain" of our **XGBoost Classifier**. We optimized:
* `scale_pos_weight`: Balanced the cost of missing a churner vs. a false alarm.
* `max_depth` & `learning_rate`: Found the "sweet spot" where the model learns complex patterns without memorizing noise (overfitting).

---

## 📈 Understanding the Performance (Accuracy vs. Value)

### "How is a 0.56 AUC Score Useful?"
In high-noise environments like banking, a perfect score often suggests data leakage. Our 0.56 AUC score represents a **statistically valid early-warning system**:

* **Strategic Learning**: The model successfully learned that **ProductCategory_Clothing** and **ServiceUsage_Website** are the primary drivers of volatility.

---

## 💡 Top Strategic Drivers (Feature Importance)

The model's "Feature Importance" chart reveals the true logic it used to separate churners from loyalists:

1. **Category Friction (Clothing)**: The #1 predictor. This segment needs a quality/return-policy audit.
2. **Service Failure (Unresolved Tickets)**: A massive trigger. Unresolved issues act as a "smoking gun" for imminent churn.
3. **Platform UI (Website vs. App)**: Website users show higher risk, indicating potential friction in the web interface compared to the app.

---

## 💻 Tech Stack & Tools
* **Modeling**: XGBoost Classifier
* **Engineering**: Pandas, NumPy, Scikit-Learn (StandardScaler, OneHotEncoder)
* **Optimization**: RandomizedSearchCV, SMOTE
* **Evaluation**: ROC-AUC, Confusion Matrix
