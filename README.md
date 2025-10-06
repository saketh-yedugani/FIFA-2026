# FIFA 2026 World Cup Winner Prediction

This repository contains a data-driven project where we predict the **Winner of the 2026 FIFA World Cup** using historical team performance data from **2006 to 2022**. The data was scraped from [Fbref](https://fbref.com/) and processed for modeling.

## Project Overview

- **Objective:** Predict which team is most likely to win the 2026 World Cup based on past tournament data.
- **Data Source:** Fbref (2006, 2010, 2014, 2018, 2022 World Cups)
- **Model Used:** XGBoost Classifier
- **Methodology:** 
  - Combined data from multiple years into a single dataset.
  - Labeled past winners (`Winner = 1`) and other teams (`Winner = 0`).
  - Performed feature engineering to create meaningful metrics.
  - Trained and validated the XGBoost model to estimate win probabilities for each team.



## Feature Engineering

From the raw tournament stats, we created the following features to feed into the model:

| Feature | Description |
|---------|-------------|
| `Win_Rate` | Ratio of Wins (`W`) to Matches Played (`MP`) |
| `Avg_GF` | Average Goals Scored per Match (`GF / MP`) |
| `Avg_GA` | Average Goals Conceded per Match (`GA / MP`) |
| `GD_per_MP` | Average Goal Difference per Match (`GD / MP`) |

These features help quantify team performance beyond raw counts of wins or goals, making the model more predictive.

---

## Why XGBoost?

We chose **XGBoost** for the following reasons:

1. **High accuracy:** It often outperforms traditional algorithms on structured/tabular data.  
2. **Handles imbalanced data well:** Only one winner per tournament, so XGBoost’s gradient boosting can manage class imbalance effectively.  
3. **Feature importance:** XGBoost allows us to understand which team metrics influence the predicted win probability the most.  
4. **Robustness and speed:** Efficient implementation with built-in regularization reduces overfitting.

---

## 📈 Modeling Process

1. Split the dataset into **train** and **validation** sets.
2. Trained an XGBoost classifier with the engineered features.
3. Predicted win probabilities for all teams in the validation set.
4. Aggregated results to identify **top contenders** for 2026.

```python
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Example of model training
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# Predict probabilities for validation set
y_pred_proba = model.predict_proba(X_val)[:, 1]
