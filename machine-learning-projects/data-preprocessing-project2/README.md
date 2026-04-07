# Data Quality Engineering for Training Generative Decoder-Only Transformers

This repository contains the practical implementation and theoretical research for a seminar project at the Faculty of Electronic Engineering, University of Niš. The project explores how data quality signals and preprocessing techniques impact the performance of Large Language Models (LLMs).

## 📂 Project Structure

The repository is divided into two main language versions, each following an identical structure:

* **/eng**: English version of the project.
* **/srb**: Serbian version of the project.

Within each language folder, you will find:
* **Academic Paper (PDF/DOCX)**: The theoretical foundation of the project titled *"Data Quality Engineering for Training Generative Decoder-Only Transformers"*.
* **Jupyter notebooks**: The practical implementation.

The notebooks are organized as follows:
1.  `steps1-8_modified.ipynb`: Data loading, quality-signal analysis, and preprocessing.
2.  `steps9-10.ipynb`: GPT-2 fine-tuning on the original low-quality dataset.
3.  `steps9-11_modified.ipynb`: GPT-2 fine-tuning on the preprocessed version of the original low-quality dataset.

---

## 🛠 Project Workflow

### Phase 1: Data Quality Analysis (Steps 1–6)
Using the **RedPajama-Data-V2** dataset as a source, we analyze and filter documents based on several quality signals:
* **Language Identification**: Ensuring text matches the target language with high confidence.
* **Heuristic Filtering**: Applying Gopher-inspired rules (Word count, mean word length, symbol-to-word ratio, and stop-word fractions).
* **Statistical Analysis**: Visualizing the distribution of quality scores to distinguish between "clean" and "noisy" data.

### Phase 2: GPT-2 Fine-Tuning & Testing
We fine-tune GPT-2 in two low-quality-data scenarios:
* **Original Low-Quality Data** (`steps9-10.ipynb`): baseline fine-tuning on noisy documents.
* **Preprocessed Low-Quality Data** (`steps9-11_modified.ipynb`): fine-tuning after cleaning/filtering the same source.

---

## 📉 Key Findings: Original vs. Preprocessed Low-Quality Data
The project demonstrates that models trained directly on low-quality data ("Trash-GPT") exhibit:
* **Model Collapse**: Frequent repetition of sentences or phrases.
* **Semantic Loss**: A breakdown in the logical context of the generated text.
* **Random Citations**: Hallucinating references or metadata typical of uncleaned web-scraped data.

In contrast, the model fine-tuned on the preprocessed low-quality dataset shows better coherence, stronger context retention, and higher informational density.

---

## ✅ Current preprocessing status

After lowering the preprocessing thresholds, the cleaned dataset contains **459 documents**.

An earlier stricter version left **274 documents**, so the relaxed thresholds are currently preferred because they provide more training data for GPT-2.

---

## 🌍 Optional augmentation by back-translation

If the cleaned dataset is still too small, you can augment it by back-translating each document:

1. Translate English text to **Spanish**.
2. Translate the Spanish text back to **English**.
3. Translate English text to **German**.
4. Translate the German text back to **English**.

This creates paraphrased English variants that can be appended to `cleaned_dataset` to increase the number of training examples.

This augmentation is optional and is intended to multiply the training set before GPT-2 fine-tuning.

---

## ☁️ Google Colab workflow

Because translation and augmentation can take a long time on CPU, the cleaned dataset can be exported locally and then loaded in Google Colab.

### Export `cleaned_dataset`

Save the processed dataset as JSONL:

```python
cleaned_dataset = ...  # your preprocessed Hugging Face Dataset
cleaned_dataset.to_json("cleaned_dataset.jsonl", orient="records", lines=True, force_ascii=False)
```

### Import the dataset in Colab

After uploading the file to Colab, restore it with Hugging Face Datasets:

```python
from datasets import load_dataset

dataset = load_dataset("json", data_files="cleaned_dataset.jsonl", split="train")
```

If you prefer, you can also store the file in Google Drive and mount the drive in Colab.

### Batch size note

When you run the augmentation pipeline in Colab, you can usually increase the batch size compared with a local CPU run. The exact value depends on the available RAM/VRAM and the translation model you choose.

---

## ⚠️ Dataset-loading note

When loading `togethercomputer/RedPajama-Data-V2`, do **not** pass `trust_remote_code=True` to `load_dataset(...)` in this project. That argument is not a valid builder configuration field for this dataset and causes this error:

```text
ValueError: BuilderConfig ... doesn't have a 'trust_remote_code' key.
```

Use the loader without that argument:

```python
from datasets import load_dataset

ds = load_dataset(
	"togethercomputer/RedPajama-Data-V2",
	name="default",
	partition="head_middle",
	snapshots=["2023-06"],
	languages=["en"],
	streaming=True,
)
```

---

## 📝 Current workflow summary

1. Load the RedPajama-Data-V2 dataset.
2. Apply language and quality filtering.
3. Keep the resulting `cleaned_dataset`.
4. Optionally export it to JSONL.
5. Move it to Google Colab if you want faster augmentation.
6. Optionally perform back-translation augmentation.
7. Fine-tune GPT-2 on the expanded dataset.

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
```
### Link to the dataset:
https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2
