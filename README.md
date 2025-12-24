# LLoyd-Banking-Group
# Lloyd Banking: Customer Churn Prediction & Behavioral Strategy

This project focuses on identifying high-risk customer segments for **Lloyd Banking** using machine learning. By analyzing transaction history and service interactions, we move beyond simple demographics to understand the **behavioral triggers** that lead to customer exit.

## The Core Insight (Pyramid Principle)
**To stabilize the customer base, Lloyd Banking must prioritize quality control in the "Clothing" product segment and aggressively resolve open customer service inquiries.** Our data shows that behavioral friction—not age or spending amount—is the primary driver of churn.

---

## Data Engineering & Learning Process

### Feature Engineering (The "Brain" of the Model)
To help the model understand time and behavior, we implemented:
* **Recency Metrics**: Converted raw dates into "Days since last interaction" to measure engagement decay.
* **Handling Imbalance**: Used **SMOTE** (Synthetic Minority Over-sampling Technique) to balance our churn data (from 25% up to 50%), ensuring the model didn't ignore the minority class.
* **Standardization**: Scaled numerical features using `StandardScaler` so that `AmountSpent` didn't unfairly outweigh `LoginFrequency`.

### Hyperparameter Tuning
We used `RandomizedSearchCV` to optimize the **XGBoost Classifier**. Key parameters tuned include:
* `max_depth`: Prevented the model from over-complicating customer patterns.
* `scale_pos_weight`: Formally instructed the model to treat catching a churner as 3x-5x more important than misclassifying a stayer.

---

## Performance & Business Impact

### "Why is a 0.56 AUC Score Useful?"
While a 0.56 AUC suggests high data noise, the model provides **Strategic Accuracy**:
* **Error Analysis**: The model identifies "Frustrated Customers" (those with unresolved tickets) with high precision.
* **Financial ROI**:
    * **Revenue Protected**: Estimated **$1,470** by saving caught churners.
    * **Cost of Campaign**: **$940** for retention vouchers.
    * **Net Profit**: **+$530** per batch analyzed.



---

## Strategic Recommendations

Based on the **Feature Importance** analysis, the business should take the following actions:

1. **Fix the "Clothing" Journey**: This is the #1 churn predictor. Audit returns and fit issues in this category.
2. **Operation "Zero Open Tickets"**: Prioritize unresolved customer inquiries, as these are the strongest "avoidable" churn triggers.
3. **Web Platform Audit**: Website users churn more than App users; investigate web-specific UI friction.



---

## Tech Stack
* **Language**: Python
* **Models**: XGBoost (Extreme Gradient Boosting)
* **Optimization**: RandomizedSearchCV, SMOTE
* **Metrics**: ROC-AUC, Confusion Matrix, ROI Analysis
