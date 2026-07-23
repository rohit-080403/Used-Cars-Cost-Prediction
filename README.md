# Audi Car Price Prediction

Predicting the resale price of used Audi cars from their specifications, using a full supervised machine learning pipeline — EDA, preprocessing, dimensionality reduction, multiple regression algorithms, boosting, cross-validation, and model persistence.

## Problem Statement

Given specifications of a used Audi car (model, year, transmission, mileage, fuel type, tax, mpg, engine size), predict its resale `price`.

**Type:** Supervised Learning — Regression

## Dataset

- **File:** `audi.csv`
- **Records:** 10,668
- **Target variable:** `price`
- **Features:** `model`, `year`, `transmission`, `mileage`, `fuelType`, `tax`, `mpg`, `engineSize`
- **Data quality:** No missing values, no major cleaning required

## Project Structure

```
├── audi.csv                  # dataset
├── Car_Price_Prediction.ipynb  # main notebook (all phases)
├── model.pkl                 # final trained model
├── preprocessing.pkl         # saved encoders + scaler for new predictions
└── README.md
```

## Pipeline / Phases

| Phase | Description |
|---|---|
| 1 | Setup & data load — imports, shape/dtype/null/duplicate checks |
| 2 | EDA — distribution plots, price vs. year/mileage, boxplots by category, correlation heatmap |
| 3 | Preprocessing — label encoding (`model`, `fuelType`), one-hot encoding (`transmission`), train/test split, `StandardScaler` |
| 4 | Dimensionality reduction — PCA (unsupervised) and LDA (supervised, using binned price tiers) — demonstrated for concept coverage |
| 5 | Baseline models — Linear Regression, Decision Tree, Random Forest, KNN (multiple parameter variants each) |
| 6 | Boosting — AdaBoost and XGBoost (multiple parameter variants), predicted-vs-actual analysis |
| 7 | Cross-validation — 5-fold CV (R²) across all models to confirm stability, not just a single lucky split |
| 8 | Final comparison, best model refit on full data, saved with `pickle` |

## Preprocessing Choices

- **`model`, `fuelType` → Label Encoding** — kept compact given `model` has 26 unique values; tree-based models tolerate the resulting false ordinality well.
- **`transmission` → One-Hot Encoding** — only 3 categories with no natural order, so one-hot avoids implying a false rank.
- **`StandardScaler`** — required for Linear Regression, PCA, and LDA (distance/variance-based); applied uniformly across all models for a fair comparison, though tree-based models don't strictly need it.

## Models Compared

Linear Regression · Decision Tree · Random Forest · KNN · AdaBoost · XGBoost — 11 model/parameter combinations in total.

## Evaluation Metrics

Regression metrics (not accuracy/F1, which are for classification):

- **R² Score** — proportion of price variance explained by the model
- **MAE** — average absolute error, in the same currency units as price
- **RMSE** — penalizes large errors more heavily than MAE
- **5-fold Cross-Validation (mean ± std of R²)** — confirms the result generalizes rather than being a lucky split

## Result

- **Best model:** selected by highest test-set R², confirmed with cross-validation
- **R² ≈ 0.94**, **CV std ≈ 0.0043** — high explanatory power with very stable performance across folds
- Errors are lowest on mainstream models and highest on rare, high-end trims (RS/S-series), due to fewer training examples in those categories

## How to Reuse the Saved Model

```python
import pickle

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('preprocessing.pkl', 'rb') as f:
    prep = pickle.load(f)

# New raw input must go through the same encoders + scaler before predicting
```

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
```

Install with:
```
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```