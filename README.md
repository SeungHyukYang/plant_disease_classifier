# Plant Disease Classifier

A binary classification project for predicting the presence of plant disease based on environmental and visual features. This project explores multiple machine learning models and techniques to handle class imbalance and improve recall for disease prediction.

---

## Project Structure

```
plant_disease_dataset/
├── data/
│   └── plant_disease_dataset.csv
├── models/
│   ├── logistic_model.py         # Standard logistic regression
│   ├── logistic_failed1.py       # Logistic regression without stratify
│   ├── logistic_balanced.py      # Logistic regression with class_weight='balanced'
│   ├── rf_model.py               # RandomForest baseline
│   ├── rf_with_smote.py          # RandomForest with SMOTE
│   ├── smote_experiments.py      # SMOTE sampling strategy tuning
│   └── xgb_model.py              # XGBoost with scale_pos_weight
├── utils.py                      # Data loading, splitting, scoring
├── main.py                       # Main execution script
```


---

## Models & Experiments

### 1. Logistic Regression
- Baseline model
- Low recall on minority class (disease present)
- Attempted `class_weight='balanced'` to improve recall
- Tried with/without `stratify` in train/test split

### 2. Random Forest
- Improved precision and recall over Logistic
- Further improved with **SMOTE** oversampling

### 3. SMOTE Experiments
- Tried `sampling_strategy=0.5`, `0.8`, and `1.0`
- Trade-off observed between overfitting and recall
- Chose optimal strategy by maximizing **recall + F1-score**

### 4. XGBoost
- Removed SMOTE, used `scale_pos_weight` to handle imbalance
- Tuned parameters like `learning_rate`, `max_depth`, `n_estimators`
- Tracked `logloss`, `aucpr` for model evaluation

---

## Key Techniques

- **Class Imbalance Handling**
  - `class_weight='balanced'` in Logistic Regression
  - `SMOTE` oversampling
  - `scale_pos_weight` in XGBoost

- **Model Evaluation**
  - Precision, Recall, F1-score
  - Confusion Matrix
  - Overfitting check (train vs test accuracy)

- **Feature Engineering (Next Step)**
  - Interaction terms (e.g., `humidity * temperature`)
  - More domain-specific features for non-linearity

---

## Key Insights

- In real-world disease detection, **high recall** is critical to avoid false negatives (i.e., missing actual disease cases)
- Random Forest with SMOTE and XGBoost with class weighting both improved recall
- Model overfitting was carefully monitored and avoided

---

## Run

1. Install required packages:
   ```bash
   pip install pandas scikit-learn imbalanced-learn xgboost matplotlib
2. Run main script:
      python main.py

3. Modify main.py to switch between models:
    from models.xgb_model import train_xgboost  # Example

## Visualization
Confusion matrix and metric plots included in SMOTE experiments

Future plan: integrate SHAP or feature importance plots

## Author
Seung Hyuk (Sam) Yang

   
