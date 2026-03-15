# 🧠 Sentiment Analysis on Bitcoin Tweets
### Deep Learning with LSTM, GRU, and CNN | PyTorch

---

## 📋 Project Overview

This project implements a sentiment analysis pipeline on a dataset of Bitcoin-related tweets using three neural network architectures: **LSTM**, **GRU**, and **CNN**. The goal is to classify tweets into three sentiment categories (Negative, Neutral, Positive) and to compare the performance of the three architectures.

The project covers the full ML pipeline — from exploratory data analysis and text preprocessing, through tokenization and model training, to final evaluation and prediction on new data.

---

## 📁 Project Structure

```
├── sentiment_analysis_bitcoin.ipynb   # Main notebook (all steps)
├── bitcoin_tweets.csv                 # Raw dataset
├── best_lstm_cls.pt                   # Best LSTM classification weights
├── best_gru_cls.pt                    # Best GRU classification weights
├── best_cnn_cls.pt                    # Best CNN classification weights
└── README.md                          # This file
```

---

## 📊 Dataset

- **Source:** [Bitcoin Tweets — Kaggle](https://www.kaggle.com/datasets/sujaykapadnis/bitcoin-tweets/data)
- **Size:** 75,955 tweets (after cleaning)
- **Period:** January 2022 — June 2023
- **Columns used:** `text`, `sentiment_label`, `sentiment_score`

### Label Distribution
| Label    | Count  | Percentage |
|----------|--------|------------|
| Neutral  | 41,584 | 54.7%      |
| Positive | 22,515 | 29.6%      |
| Negative | 11,856 | 15.6%      |

---

## 🔧 Pipeline

### Step 1 — Exploratory Data Analysis (EDA)
- Dataset structure, missing values, duplicate detection
- Manual resolution of 14 conflicting duplicate labels
- Classification label and regression score distribution analysis
- Tweet length analysis → determined `MAX_SEQ_LEN = 29`
- Most frequent words per sentiment class, word clouds
- Time distribution of tweets (Jan 2022 — Jun 2023)
- Engagement analysis (likes, retweets, quotes)

### Step 2 — Text Preprocessing
Full cleaning pipeline applied to each tweet:
1. Unicode normalization (NFKC)
2. Lowercasing
3. URL removal
4. @mention removal
5. `#` symbol stripped, word kept
6. `$` symbol stripped, word kept
7. Emoji removal
8. Newline/tab removal
9. Punctuation & special character removal
10. NLTK stopword removal (negations preserved)
11. Extra whitespace removal

### Step 3 — Tokenization & Vocabulary Building
- Train/Val/Test split: **70% / 15% / 15%** (stratified on label)
- Vocabulary built on training data only (`MIN_FREQ=2`)
- Vocabulary size: **20,739 tokens** (including `<PAD>` and `<UNK>`)
- `<UNK>` rate: Val=4.52%, Test=4.31%
- Pre-padding to `MAX_SEQ_LEN=29`
- PyTorch `Dataset` and `DataLoader` with `BATCH_SIZE=64`

### Step 4 — Model Architecture

All three models share the same structure:
```
Embedding → [LSTM / GRU / CNN] → Dropout → Fully Connected → Output
```

#### Hyperparameters (Final Configuration)
| Parameter     | Value        |
|---------------|--------------|
| EMBED_DIM     | 128          |
| HIDDEN_DIM    | 128          |
| NUM_LAYERS    | 2            |
| DROPOUT       | 0.5          |
| NUM_FILTERS   | 128 (CNN)    |
| KERNEL_SIZES  | [2, 3, 4] (CNN) |
| VOCAB_SIZE    | 20,739       |
| MAX_SEQ_LEN   | 29           |

#### Trainable Parameters
| Model | Classification | Regression |
|-------|---------------|------------|
| LSTM  | 2,919,171     | 2,918,913  |
| GRU   | 2,853,123     | 2,852,865  |
| CNN   | 2,803,587     | 2,802,817  |

### Step 5 — Training
- **Optimizer:** AdamW (`lr=0.001`)
- **Loss:** CrossEntropyLoss (classification), MSELoss (regression)
- **Early stopping:** patience=3 epochs
- **Gradient clipping:** `max_norm=1.0`
- **Device:** NVIDIA T4 GPU (Google Colab)

#### Hyperparameter Tuning Experiments
| Experiment | DROPOUT | NUM_LAYERS | HIDDEN_DIM | Best GRU Val Acc |
|------------|---------|------------|------------|-----------------|
| Baseline   | 0.0     | 2          | 128        | 0.709           |
| + Dropout  | 0.5     | 2          | 128        | 0.702 ✅ chosen  |
| Reduced layers | 0.5 | 1         | 128        | 0.697           |
| Reduced dim | 0.5   | 1          | 64         | 0.704           |

### Step 6 — Final Test Evaluation

#### Classification Results (Test Set)
| Model | Accuracy | F1 Weighted | F1 Macro |
|-------|----------|-------------|----------|
| LSTM  | 0.6896   | 0.6799      | 0.6307   |
| GRU   | **0.6995** | **0.6950** | **0.6568** |
| CNN   | 0.7036   | 0.6912      | 0.6387   |

#### Per-class F1 (Test Set)
| Model | Negative | Neutral | Positive |
|-------|----------|---------|----------|
| LSTM  | 0.4965   | 0.7522  | 0.6434   |
| GRU   | **0.5509** | 0.7505 | **0.6689** |
| CNN   | 0.4927   | **0.7670** | 0.6564 |

### Step 7 — Prediction on New Tweets
- 15 manually crafted tweets (5 per class)
- Preprocessed and tokenized using the same pipeline
- Predictions and confidence scores from all 3 models

---

## 📈 Key Findings

1. **GRU outperforms LSTM and CNN** on F1 Macro and per-class F1 for both Negative and Positive classes, making it the best overall architecture for this task. Its simpler gating mechanism (2 gates vs LSTM's 4) proves sufficient for short tweet sequences.

2. **CNN achieves the highest accuracy** (0.7036) and best Neutral F1 (0.7670), but is the most biased toward the majority class. It has the worst Negative F1 (0.4927), indicating it sacrifices minority class performance for overall accuracy.

3. **LSTM and CNN show similar performance profiles** — the extra complexity of LSTM's cell state provides no meaningful advantage over GRU on short sequences.

4. **The dominant misclassification pattern** across all models is Negative tweets being predicted as Neutral (52.67% LSTM, 46.71% GRU, 55.99% CNN). Notably, Negative→Positive and Positive→Negative confusions are minimal (~1-5%), indicating the models correctly distinguish polar opposites but struggle with the Neutral boundary.

5. **Overfitting was present** in all experiments, manifesting as a large gap between train and val loss. Early stopping successfully captured the best generalization weights (typically at epoch 3-5). Dropout (0.5) reduced overfitting and particularly helped CNN.

6. **The sentiment score** (`sentiment_score`) appears to be a confidence score from an automated labeler rather than a signed sentiment value. All scores fall in [0.35, 1.0] with significant overlap between classes (mean scores: Negative=0.677, Neutral=0.699, Positive=0.759), making regression a less meaningful task on this dataset.

7. **Tweet volume increased significantly** from August 2022 onwards, coinciding with the crypto bear market acceleration — a real-world event reflected in the dataset.

---

## ⚙️ Requirements

```
torch
numpy
pandas
matplotlib
seaborn
scikit-learn
nltk
wordcloud
```

Install with:
```bash
pip install torch numpy pandas matplotlib seaborn scikit-learn nltk wordcloud
```

NLTK downloads required:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
```

---

## 🚀 How to Run

1. Clone the repository
2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/sujaykapadnis/bitcoin-tweets) and place `bitcoin_tweets.csv` in the project root
3. Install requirements
4. Open `sentiment_analysis_bitcoin.ipynb`
5. Update `DATA_PATH` in the notebook to point to your CSV file
6. Run all cells sequentially

> **Note:** Training 6 models requires a GPU for reasonable runtime. Use Google Colab with T4 GPU (`Runtime → Change runtime type → T4 GPU`) for best performance. On CPU, training takes approximately 1.5 hours for classification models only.

---

## 📝 Notes

- The notebook is structured as a clean notebook — training outputs are not included inline due to computational constraints. All results are documented in this README and in the saved `.pt` weight files.
- Pre-trained weights are provided in the repository for direct evaluation without retraining.
- The regression task code is included but commented out, as the `sentiment_score` column was determined to be an unreliable regression target.

---
