# 📂 Dataset Description

## 📌 Overview

This dataset contains customer reviews of restaurants located in Majhitar. It is used for performing sentiment analysis using both lexicon-based (VADER) and transformer-based (RoBERTa) NLP models.

---

## 📊 File Included

```text id="x7g2q1"
majhitar_active_places_only.csv
```

---

## 🔍 Sample Data Preview

```text id="w3p8zd"
Restaurant        | Review                              | 
------------------|-------------------------------------|
Cafe Delight      | Food was amazing and service fast   | 
Spice Hub         | Average taste, nothing special      | 
Food Corner       | Very slow service and rude staff    | 
```

---

## 🧹 Data Preprocessing Notes

* Empty and null reviews are removed
* Text is preserved in raw form for realistic NLP analysis
* No aggressive cleaning to retain sentiment cues

---

---

## ⚠️ Important Notes

```text id="b6r9qh"
- File path used in code:
  data/majhitar_active_places_only.csv

- Do NOT change column names

- Ensure CSV format is maintained
```
