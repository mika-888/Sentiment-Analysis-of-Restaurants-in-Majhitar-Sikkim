import pandas as pd
import torch
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import matplotlib.patches as mpatches

# --- Ensure results folder ---
os.makedirs("results", exist_ok=True)

# --- Configuration ---
INPUT_FILE = 'data/majhitar_active_places_only.csv'
OUTPUT_CSV = 'results/model2_bert_results.csv'
OUTPUT_GRAPH = 'results/model2_bert_6plots.png'

# --- Initialize BERT ---
print("Initializing BERT model...")
device = 0 if torch.cuda.is_available() else -1

bert_pipe = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    device=device
)

COLORS = {'Positive': '#5a9e6f', 'Neutral': '#7eaec4', 'Negative': '#c4735a'}

# --- Helper: Convert star rating → sentiment ---
def convert_label(label):
    stars = int(label[0])  # e.g. "5 stars" → 5
    if stars >= 4:
        return "Positive"
    elif stars == 3:
        return "Neutral"
    else:
        return "Negative"

# --- Load Data ---
df = pd.read_csv(INPUT_FILE)

# --- Prediction ---
def analyze_bert(text):
    res = bert_pipe(str(text)[:512])[0]
    sentiment = convert_label(res['label'])
    confidence = res['score']

    # Convert to scale (-1 to +1)
    score = confidence if sentiment == "Positive" else (-confidence if sentiment == "Negative" else 0)

    return pd.Series([sentiment, confidence, score])

print(f"Analyzing {len(df)} reviews with BERT...")

start = time.perf_counter()

df[['BERT_Pred', 'BERT_Conf', 'BERT_Scale']] = df['Review'].apply(analyze_bert)

elapsed = time.perf_counter() - start
print(f"Done in {elapsed:.2f}s\n")

# --- Leaderboard ---
leaderboard = (
    df.groupby('Restaurant')
      .agg(
          Avg_Conf=('BERT_Conf', 'mean'),
          Pos_Pct=('BERT_Pred', lambda x: (x == 'Positive').mean() * 100),
          Review_Count=('Review', 'count')
      )
      .sort_values('Pos_Pct', ascending=False)
      .reset_index()
)

# --- VISUALIZATION (6 PLOTS) ---
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(18, 10))
fig.suptitle('Model 2 — BERT Sentiment Analysis', fontsize=16)

# 1. Pie Chart
ax1 = fig.add_subplot(2, 3, 1)
counts = df['BERT_Pred'].value_counts()
ax1.pie(counts, labels=counts.index, autopct='%1.1f%%',
        colors=[COLORS[s] for s in counts.index])
ax1.set_title('Sentiment Distribution')

# 2. Confidence Histogram
ax2 = fig.add_subplot(2, 3, 2)
sns.histplot(df['BERT_Conf'], bins=15, kde=True, color='purple', ax=ax2)
ax2.set_title('Confidence Distribution')

# 3. Top Restaurants
ax3 = fig.add_subplot(2, 3, 3)
top10 = leaderboard.head(10)
ax3.barh(top10['Restaurant'], top10['Pos_Pct'], color=COLORS['Positive'])
ax3.set_title('Top Restaurants (% Positive)')
ax3.invert_yaxis()

# 4. Confidence Boxplot
ax4 = fig.add_subplot(2, 3, 4)
sns.boxplot(x='BERT_Pred', y='BERT_Conf', data=df, palette=COLORS, ax=ax4)
ax4.set_title('Confidence Spread')

# 5. Countplot
ax5 = fig.add_subplot(2, 3, 5)
sns.countplot(x='BERT_Pred', data=df, palette=COLORS, ax=ax5)
ax5.set_title('Sentiment Count')

# 6. Scale Distribution
ax6 = fig.add_subplot(2, 3, 6)
sns.histplot(df['BERT_Scale'], bins=15, color='gray', ax=ax6)
ax6.set_title('Scaled Sentiment Score')

plt.tight_layout()
plt.savefig(OUTPUT_GRAPH, dpi=150)
plt.show()

# --- Save Results ---
df.to_csv(OUTPUT_CSV, index=False)

print(f"\nResults saved to {OUTPUT_CSV}")
print("BERT model complete.")
