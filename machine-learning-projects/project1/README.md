# 📊 Analysis and Classification of Morphological Characteristics of Possums

This project explores a biological dataset on possums through three key machine learning problems: **geographical region classification**, **sex classification**, and **age regression**. The main focus of the study is **model interpretability (XAI)** and understanding why models fail in specific biological scenarios.

## 🗂️ About the dataset
The dataset contains morphological measurements (body, head, and tail lengths, skull width, etc.) of possum individuals collected from different locations across Australia. The data were scaled using `StandardScaler` prior to the modeling process.  
Dataset link: https://www.kaggle.com/datasets/abrambeyer/openintro-possum

---

## 🔍 Key findings and interpretability

### 1. Region classification (Successful)
* **Model:** Logistic Regression / Random Forest  
* **Result:** High accuracy and stability  
* **XAI insight (SHAP):** The analysis revealed that **ear conch length (`earconch`)** is the most important predictor. Possums from Victoria exhibit significantly different ear morphology compared to other populations, allowing the model to easily distinguish them.

### 2. Sex classification (Biological overlap problem)
* **Problem:** The model achieves an accuracy of only **~65%**  
* **Lasso (L1) reduction:** The use of L1 regularization resulted in most variable coefficients being shrunk to zero, suggesting that there is no clear morphological “signature” for sex  
* **LIME analysis:** A detailed inspection of individual errors (e.g., a male classified as female) showed that individual size variations (chest, abdomen) often override sex-based differences  
* **Conclusion:** By comparing correctly and incorrectly classified males, it was determined that the model makes decisions based on noise in the data, as the morphological measurements of the two sexes are too similar

### 3. Age prediction (Low signal)
* **Model:** Support Vector Regression (SVR)  
* **Result:** $R^2 \approx 0.15$  
* **Interpretability:** SHAP values are highly concentrated around zero, mathematically confirming that physical body measurements are not a reliable indicator of an individual’s age in this sample

---

## 🛠️ Technologies and methods
* **Programming language:** Python  
* **Libraries:** `pandas`, `numpy`, `scikit-learn`, seaborn...
* **Interpretability (XAI):** `SHAP` (global explanations), `LIME` (local explanations)
* **Dimensionality reduction:** PCA (Principal Component Analysis), L1 (Lasso) regularization

---

## 📈 Final conclusion
This project demonstrates the importance of **explainable AI**. Rather than focusing solely on low performance metrics for sex and age prediction, the use of SHAP and LIME allowed us to scientifically demonstrate that the issue lies in **biological overlap of features**, not in the model architecture itself.

---

### How to run the project?
1. Clone the repository: `git clone [REPOSITORY_URL_HERE]`
2. Install dependencies: `pip install pandas sklearn shap lime matplotlib seaborn`
3. Launch Jupyter Notebook: `jupyter notebook`

