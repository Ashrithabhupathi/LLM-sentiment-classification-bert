# LLM-Based Sentiment Classification (Fine-Tuned BERT)

## 📌 Project Overview

This project builds an LLM-powered Natural Language Processing pipeline to classify movie reviews as Positive or Negative using a fine-tuned BERT transformer model.

The model was trained on 50,000 IMDb movie reviews and achieved:

- ✅ 89.1% Accuracy  
- ✅ 0.891 F1-Score  
- 🚀 Outperformed traditional TF-IDF + Logistic Regression baseline  

This project demonstrates real-world NLP workflow including preprocessing, transformer fine-tuning, model evaluation, and performance comparison.

---

## 🎯 Business Problem

Companies receive massive volumes of customer feedback daily. Manually analysing sentiment is inefficient and slow.

This solution automates sentiment detection using a transformer-based LLM, enabling:

- Faster decision-making
- Customer experience monitoring
- Product feedback analysis
- Market sentiment tracking

---

## 🛠 Tech Stack

- Python
- HuggingFace Transformers
- PyTorch
- Scikit-learn
- Pandas
- NumPy
- Matplotlib / Seaborn

---

## 📂 Dataset

- IMDb Movie Reviews Dataset (50,000 labelled reviews)
- Balanced dataset (Positive / Negative)

---

## ⚙️ Project Workflow

1. Data Loading & Cleaning
2. Text Tokenization using BERT tokenizer
3. Train-Test Split
4. Fine-tuning Pre-trained BERT Model
5. Model Evaluation (Accuracy, Precision, Recall, F1-score)
6. Baseline Model Comparison (TF-IDF + Logistic Regression)
7. Performance Analysis

---

## 📊 Model Performance

| Model                          | Accuracy | F1-Score |
|--------------------------------|----------|----------|
| TF-IDF + Logistic Regression   | ~84%     | ~0.84    |
| Fine-Tuned BERT (LLM)          | **89.1%**| **0.891**|

The fine-tuned transformer model significantly outperformed the traditional ML baseline.

---

## 📈 Key Insights

- Transformer models capture contextual meaning better than bag-of-words approaches.
- Fine-tuning pre-trained LLMs provides strong performance even with limited training epochs.
- LLM-based NLP solutions are scalable for enterprise-level sentiment monitoring.

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
