# 📌 TANLP Project: Sentiment Analysis of Majhitar Restaurants

## 📖 Overview

This project implements and compares two sentiment analysis approaches on real-world restaurant reviews:

* VADER (lexicon-based baseline model)
* RoBERTa (transformer-based SOTA model)

The goal is to evaluate performance in terms of **accuracy, speed, and practical usability**.

---

## 🎯 Objectives

* Perform sentiment classification on real-world data
* Compare baseline vs advanced NLP models
* Generate actionable insights from reviews
* Visualize sentiment trends and rankings

---

## 📂 Project Structure

```text
TANLP-Sentiment-Analysis/
│
├── data/
│   └── majhitar_active_places_only.csv
│
├── models/
│   ├── vader_model.py
│   ├── roberta_model.py
│
├── analysis/
│   └── comparison_dashboard.py
│
├── results/
│   ├── model1_vader_results.csv
│   ├── model2_roberta_results.csv
│   ├── model1_vader_plots.png
│   ├── model2_roberta_plots.png
│   └── Final_Comparison_Dashboard.png
│
├── requirements.txt
├── run_all.py
└── README.md
```

---

## ⚙️ Models Used

### 🔹 VADER

* Rule-based sentiment analysis
* Fast and lightweight
* No training required

### 🔹 RoBERTa

* Transformer-based deep learning model
* Context-aware predictions
* Higher accuracy

---

## 📊 Features

* Sentiment classification (Positive / Neutral / Negative)
* Score distribution visualization
* Restaurant ranking
* Model comparison dashboard
* N-gram analysis for negative reviews

---

## 🚀 How to Run

### 1. Clone Repository

```bash
git clone <your-repo-link>
cd TANLP-Sentiment-Analysis
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Project

```bash
python run_all.py
```

---

## 📁 Outputs

```text
results/
├── model1_vader_results.csv
├── model2_roberta_results.csv
├── model1_vader_plots.png
├── model2_roberta_plots.png
└── Final_Comparison_Dashboard.png
```

---

## 📈 Key Insights

* VADER is extremely fast and efficient
* RoBERTa provides better contextual understanding
* Trade-off exists between speed and accuracy
* N-gram analysis highlights common customer complaints

---

## ⚖️ Model Comparison

```text
Feature          | VADER        | RoBERTa
-----------------|--------------|---------
Type             | Rule-based   | Transformer
Speed            | Very Fast    | Slower
Accuracy         | Moderate     | High
Training Needed  | No           | Pretrained
Interpretability | High         | Moderate
```

---

## 🧠 Learning Outcomes

* Understanding NLP model differences
* Hands-on sentiment analysis pipeline
* Data visualization and interpretation
* Performance benchmarking

---

## 🔍 Future Work

```text
- Fine-tuned BERT model
- Multilingual sentiment analysis
- Deployment using Flask/FastAPI
```

---

## 📌 Conclusion

This project demonstrates practical sentiment analysis using both traditional and modern NLP approaches, highlighting the importance of selecting models based on real-world constraints like speed and accuracy.

---

## 👩‍💻 Author

```text
Name   : Your Name
Course : Text Analytics & NLP (TANLP)
Branch : B.Tech CSE (Data Science)
```
