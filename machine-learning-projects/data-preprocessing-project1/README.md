# Bank Customer Churn Prediction: Data Quality, Modern ML, and AutoML

This project predicts customer attrition in banking, with a strong focus on **Data Quality (DQ)** and model reliability.
It combines rigorous data profiling with modern machine learning, hyperparameter tuning, explainability, and AutoML benchmarking.

## Project Scope
- Theoretical part: seminar papers (`Data_Quality_Seminarski.*`, `Data_Quality_Academic_Paper.*`)
- Practical part: Jupyter notebooks, primarily:
  - `practical_part_eng.ipynb` (English practical notebook)

## End-to-End Workflow (Enhanced Notebook)
The current pipeline in `practical_part_eng.ipynb` includes:

1. Data loading and initial quality checks
2. Attribute typing and profiling
3. Statistical characterization (mean/median/std/skewness/kurtosis)
4. Distribution and normality analysis
5. Correlation and redundancy analysis (Pearson, Spearman, VIF, Cramer's V)
6. Anomaly detection (Z-score, IQR, Mahalanobis distance)
7. Baseline modeling (pre-preprocessing)
8. Feature engineering (interaction, ratio, domain-driven features)
9. Data drift analysis (PSI + KS test)
10. Preprocessing and transformations (winsorization, Box-Cox, standardization)
11. Post-preprocessing modeling and baseline comparison
12. Modern ML stack:
   - LightGBM
   - CatBoost
   - SHAP interpretability
   - Optuna hyperparameter search (XGBoost)
13. H2O AutoML benchmark
14. Final model comparison

## Tech Stack
- **Language:** Python (notebook-based workflow)
- **Core libs:** `pandas`, `numpy`, `scipy`, `scikit-learn`
- **Modeling:** `xgboost`, `lightgbm`, `catboost`
- **Tuning:** `optuna`
- **Explainability:** `shap`
- **AutoML:** `h2o` (`H2OAutoML`)
- **Visualization:** `matplotlib`, `seaborn`

## Model Strategy
The project compares multiple levels of modeling maturity:
- **Baseline XGBoost** (raw/preprocessed comparison)
- **Enhanced XGBoost** after feature engineering and preprocessing
- **LightGBM** and **CatBoost** as modern gradient boosting alternatives
- **Optuna-tuned XGBoost** for automated hyperparameter optimization
- **H2O AutoML** as external benchmark and validation layer

Final comparison (Section 14) is built dynamically from available model metrics, including Optuna results when that cell is executed.

## Explainability
Model interpretation is performed with **SHAP**:
- feature importance summary
- feature impact distribution
- fallback to XGBoost built-in feature importance if SHAP fails in runtime

## Dataset
- Local file used in notebooks: `BankChurners.csv`
- Source dataset (Kaggle):
  - https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers

## Notes
- Some components are optional at runtime (LightGBM, CatBoost, SHAP, Optuna, H2O).
- The notebook includes safety checks and graceful fallbacks when a package is unavailable.
- Exact metric values depend on your environment and executed notebook path/cells.
