# Data Quality Engineering for Training Generative Decoder-Only Transformers

This repository contains the practical implementation and theoretical research for a seminar project at the Faculty of Electronic Engineering, University of Niš. The project explores how data quality signals and preprocessing techniques impact the performance of Large Language Models (LLMs).

## 📂 Project Structure

The repository is divided into two main language versions, each following an identical structure:

* **/eng**: English version of the project.
* **/srb**: Serbian version of the project.

Within each language folder, you will find:
* **Academic Paper (PDF/DOCX)**: The theoretical foundation of the project titled *"Data Quality Engineering for Training Generative Decoder-Only Transformers"*.
* **/good-gpt**: Implementation using high-quality data.
* **/trash-gpt**: Implementation using low-quality data to demonstrate model degradation.

Both the `good-gpt` and `trash-gpt` folders contain:
1.  `steps1-6.ipynb`: Data loading, analysis of quality signals, and filtering.
2.  `steps7-8.ipynb`: Fine-tuning GPT-2 and evaluating the resulting model.

---

## 🛠 Project Workflow

### Phase 1: Data Quality Analysis (Steps 1–6)
Using the **RedPajama-Data-V2** dataset as a source, we analyze and filter documents based on several quality signals:
* **Language Identification**: Ensuring text matches the target language with high confidence.
* **Heuristic Filtering**: Applying Gopher-inspired rules (Word count, mean word length, symbol-to-word ratio, and stop-word fractions).
* **Statistical Analysis**: Visualizing the distribution of quality scores to distinguish between "clean" and "noisy" data.

### Phase 2: GPT-2 Fine-Tuning & Testing (Steps 7–8)
We fine-tune two versions of a GPT-2 model to compare results:
* **High-Quality Dataset**: 3,500 documents passed through rigorous quality filters.
* **Low-Quality Dataset**: 3,500 documents containing noise (HTML tags, boilerplate text, low-information content).

---

## 📉 Key Findings: High-Quality vs. Low-Quality
The project demonstrates that models trained on low-quality data ("Trash-GPT") exhibit:
* **Model Collapse**: Frequent repetition of sentences or phrases.
* **Semantic Loss**: A breakdown in the logical context of the generated text.
* **Random Citations**: Hallucinating references or metadata typical of uncleaned web-scraped data.

In contrast, the **High-Quality** model shows better coherence, adherence to natural language structures, and higher informational density.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* Jupyter Notebook / Google Colab
* Hugging Face Transformers & Datasets libraries
* PyTorch

### Installation
```bash
pip install transformers datasets torch pandas matplotlib

### Link to the dataset: https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2
