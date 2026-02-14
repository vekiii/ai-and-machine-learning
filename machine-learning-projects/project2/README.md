# E-Commerce Transaction Analysis & Clustering

This project focuses on the segmentation and analysis of transaction data to optimize business performance. Using Unsupervised Learning techniques (Clustering) and Dimensionality Reduction (PCA), the study identifies key patterns in customer behavior, regional issues, and product category performance.

---

## 📊 Key Findings

### 1. Optimal Number of Clusters
The analysis utilized multiple algorithms to determine the most effective number of clusters. 
* **Result:** All metrics initially suggested **K=2** as the mathematical optimum.
* **Decision:** Due to the nature of the task and the need to distinguish between regular and extreme transactions, **K=3** was chosen.
* **Observation:** The proximity of the two clusters on the right side (regular vs. extreme) is so high that they overlap at the boundaries, a fact also confirmed by the **MeanShift** algorithm, which grouped them together.

### 2. PCA Visualization (Variance: 58.2%)
The Principal Component Analysis (PCA) reveals the underlying structure of the data:
* **Cluster 1 (Left):** Distinctly separated, representing **deficit transactions** (losses).
* **Clusters 0 & 2 (Right):** Represent **regular and extreme transactions**, with some overlap at the borders.
* **PC1 (42.8% Variance):** Primarily driven by the correlation between **Profit and Discount**.
* **PC2 (Delivery Speed):** Shows distinct lines representing delivery times (0-7 days). Interestingly, all transaction types are equally distributed across these lines, suggesting delivery speed does not significantly impact business outcomes.
* **PC3 (Segments):** Clearly separates **Consumer, Corporate, and Home Office** segments.

---

## 👥 Segment Insights
| Segment | Behavior |
| :--- | :--- |
| **Consumer** | Least likely to make extreme purchases; contribute most to losses due to heavy discount usage. |
| **Home Office** | Most balanced distribution; identified as the most reliable customer group. |
| **Corporate** | Focused on standard purchases, utilizing discounts when available, but capable of extreme transactions. |

---

## 📉 Regional & Category Focus: Central Region & Furniture
The analysis pinpointed specific areas requiring operational improvements:

* **Region:** The **Central region** shows the highest need for intervention.
* **Category:** **Furniture** is the most problematic category.
* **Key Issue:** Discounts in the Central region for Furniture are excessively high, averaging **over 30%**, which directly correlates with the identified deficit transactions.
* **Action Items:** Targeted improvements are needed for the **Top 15 cities** in the Central region and specific **Furniture sub-categories**.

---

## 🛠 Tech Stack
* **Python** (Pandas, NumPy)
* **Scikit-Learn** (K-Means, PCA, MeanShift)
* **Visualization:** Matplotlib, Seaborn
* **Clustering Metrics:** Dendrograms, Silhouette Scores

---

## 🚀 How to Use
1. Clone the repository.
2. Run the Jupyter Notebook to see the step-by-step PCA loadings and clustering process.
3. Refer to the `Results` section for specific city-level data in the Central region.

## Dataset link:
https://www.kaggle.com/datasets/jacopoferretti/superstore-dataset
