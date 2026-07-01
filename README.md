# Teen Mental Health Prediction Model

A machine learning classification model that predicts depression risk in teenagers based on social media usage patterns, sleep habits, stress levels, and other behavioural features.

---

## Problem Statement

Depression among teenagers is a growing concern and is often underdiagnosed. With only 2.6% of cases being positive (depressed) in the dataset, this is a highly imbalanced classification problem. The goal is to build a model that can flag at-risk teens early, prioritising recall over precision — because missing a depressed teen is far more costly than a false alarm.

---

## Dataset

- **Source:** Kaggle — Teen Mental Health Dataset
- **Size:** 1,200 rows, 13 features
- **Target variable:** `depression_label` (0 = not depressed, 1 = depressed)
- **Class distribution:** 97.4% negative (1169) vs 2.6% positive (31) — severely imbalanced

**Features used:**

| Feature | Description |
|---|---|
| age | Teen's age (13–19) |
| daily_social_media_hours | Hours spent on social media per day |
| sleep_hours | Average hours of sleep per night |
| screen_time_before_sleep | Hours of screen use before sleeping |
| academic_performance | GPA score |
| physical_activity | Hours of physical activity per day |
| stress_level | Self-reported stress score (1–10) |
| anxiety_level | Self-reported anxiety score (1–10) |
| addiction_level | Social media addiction score (1–10) |
| gender_male_flag | Binary encoded gender |
| Instagram_user_flag | Whether teen uses Instagram |
| Tiktok_user_flag | Whether teen uses TikTok |
| social_interaction_encoded | Ordinal encoded social interaction level |

---

## Approach

### 1. Exploratory Data Analysis (EDA)
- Checked class distribution, missing values, duplicates
- Analysed feature correlations with target
- Found top predictors: `daily_social_media_hours` (0.138), `anxiety_level` (0.098), `stress_level` (0.072), `sleep_hours` (−0.089)

### 2. Feature Engineering
- Label encoded `social_interaction_level` (low=0, medium=1, high=2) — ordinal column
- One hot encoded `gender` and `platform_usage` — nominal columns
- Dropped `User_id`, raw text columns, and `gender_female_flag` (dummy variable trap)

### 3. Handling Class Imbalance
Given the extreme 37:1 negative-to-positive ratio, three strategies were applied:
- `scale_pos_weight = 38` inside XGBoost — penalises missing positive class 38x more
- **BalancedBaggingClassifier (BBC)** — trains 5 separate models each on a balanced bucket (all 25 positives + random 100 negatives at 1:5 ratio), then averages probabilities
- **StratifiedKFold** — ensures every cross validation fold preserves the 2.6% positive rate

### 4. Model
- **Algorithm:** XGBoost (`XGBClassifier`) wrapped inside `BalancedBaggingClassifier`
- **BBC buckets:** 5 estimators, sampling strategy 1:5, with replacement (bagging)
- **XGBoost parameters:** n_estimators=100, learning_rate=0.1, max_depth=4, subsample=0.8, scale_pos_weight=38

### 5. Evaluation Strategy
- Threshold analysis across 0.1 to 0.9 instead of default 0.5
- Primary metric: **Recall** — catching depressed teens matters more than avoiding false alarms
- Secondary metrics: Precision, F1, AUC-ROC, AUC-PR, Log Loss

---

## Results

### Final Test Set Performance (240 rows, 6 positives)

| Metric | Class 0 (Healthy) | Class 1 (Depressed) |
|---|---|---|
| Precision | 99% | 19% |
| Recall | 93% | **67%** |
| F1 Score | 96% | 30% |
| Accuracy | 92% | — |

### Confusion Matrix

```
                   Predicted Healthy   Predicted Depressed
Actual Healthy          217                  17
Actual Depressed          2                   4
```

- **True Positives (TP):** 4 depressed teens correctly caught
- **False Negatives (FN):** 2 depressed teens missed
- **False Positives (FP):** 17 healthy teens flagged (acceptable false alarms)
- **True Negatives (TN):** 217 healthy teens correctly cleared

### Cross Validation (StratifiedKFold, 5 folds)

| Metric | Mean | Std |
|---|---|---|
| Recall | 0.48 | 0.16 |
| Precision | 0.163 | 0.069 |
| F1 | 0.241 | 0.093 |

### Log Loss
- Train: 0.1116 (excellent)
- Test: 0.1429 (excellent)
- Gap: 0.03 (no significant overfitting)

---

## Key Learnings

- **Accuracy is a misleading metric for imbalanced data** — a model predicting everyone as healthy gets 97.4% accuracy but catches zero depressed teens
- **BalancedBaggingClassifier outperformed plain undersampling** — Random Forest with RandomUnderSampler gave 0% recall across all CV folds; BBC + XGBoost gave 48–67% recall
- **Threshold tuning matters** — default 0.5 missed most positives; tuning the threshold is essential for rare event detection
- **With only 31 positives**, model stability is inherently limited — high standard deviation (0.16) in CV recall is expected and not a model failure

---

## Tech Stack

```
Python          pandas          numpy
scikit-learn    XGBoost         imbalanced-learn
SHAP            matplotlib      SQLite (pandasql)
```

---

## Project Structure

```
Teen-mental-health-prediction-model/
├── README.md
└── teen-mental-health-prediction-model.ipynb
```

---

## How to Run

1. Open the notebook on [Kaggle](https://www.kaggle.com) or download and run locally
2. Install dependencies:
```bash
pip install xgboost imbalanced-learn shap scikit-learn pandas numpy matplotlib
```
3. Run all cells sequentially

---

## Author

**Atharv** — [GitHub](https://github.com/cuuri0usatharv)
