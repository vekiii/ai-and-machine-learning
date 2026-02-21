# Bank Customer Churn Prediction: A Data Quality & AutoML Approach

This project explores customer attrition prediction in the banking sector with a primary focus on **Data Quality (DQ)**. The work demonstrates how rigorous statistical processing, cleaning, and data transformation directly impact the stability and precision of Machine Learning models.

## 🚀 Project Overview
The main goal of this project is to identify customers planning to close their accounts (Attrition). Rather than treating algorithms as "black boxes," this project emphasizes **Data Engineering** and distribution analysis, proving the principle that "smart data" is more important than the complexity of the algorithm itself.
The project consists of **Theoretical Part in the form of a Seminar Paper** and **Practical Part in the form of a Jupyter Notebook**.

### Key Objectives:
* **Quality Analysis**: Implementation of data quality dimensions (accuracy, completeness, consistency).
* **Statistical Profiling**: Detailed analysis of central tendency, variance, and distribution moments (Skewness and Kurtosis).
* **Hybrid Approach**: Benchmarking a manually optimized XGBoost model against **H2O AutoML** validation.

---

## 🛠️ Technologies Used
* **Language:** Python 3.12.7
* **Libraries:** `Pandas`, `NumPy`, `Scikit-Learn`, `XGBoost`, `Matplotlib`, `Seaborn`
* **AutoML Platform:** `H2O.ai` (H2OAutoML)
* **Environment:** Jupyter Notebook

---

## 📊 Data Quality & Preprocessing Workflow
The core of this project is a data preparation pipeline that includes:

1. **Cleaning & Imputation**: Identification and treatment of missing values and "Unknown" categories.
2. **Outlier Detection**: Implementation of **Mahalanobis Distance** to identify multivariate extreme values that could compromise model stability.
3. **Normalization**: Use of **Box-Cox transformations** to correct skewness in financial attributes and achieve a normal distribution.
4. **Dimensionality Reduction**: 
    * Multicollinearity analysis via correlation matrices.
    * Variance Inflation Factor (VIF) checks.
    * Feature reduction by ~50% while maintaining predictive power.
---

## 🤖 Modeling & AutoML Validation
The project compares two approaches to test the hypothesis regarding the importance of data preparation:

### 1. Manual XGBoost
The model was trained on a highly purified dataset (`data_normalized`). The focus was on achieving maximum precision with a minimal number of relevant attributes.

### 2. H2O AutoML (Benchmark)
AutoML was used as an objective verifier on the cleaned data.
* **Winning Model**: `StackedEnsemble_AllModels` achieved an **AUC of 0.991**.
* **Conclusion**: Validation confirmed that the manual dimensionality reduction did not cause a loss of critical information, as the AutoML achieved top-tier results on a significantly smaller number of features.

---

## 📈 Results
| Model | Metric (AUC) | Status |
| :--- | :--- | :--- |
| **XGBoost (Manual)** | ~0.97 | High Interpretability |
| **H2O Stacked Ensemble** | **0.991** | Maximum Precision |

**Key Takeaway:** High-quality data preparation allows for the development of more stable and simpler models that are easier to implement in real-world banking systems.

---

## Dataset link:
https://www.kaggle.com/datasets/jacopoferretti/superstore-dataset
