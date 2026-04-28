import os
import pandas as pd
import time
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import numpy as np

# --- Ensure results folder ---
os.makedirs("results", exist_ok=True)

# --- Initialization ---
nltk.download('vader_lexicon', quiet=True)
vader = SentimentIntensityAnalyzer()

device = 0 if torch.cuda.is_available() else -1
roberta_pipe = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=device
)

label_map = {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"}

# --- Load Dataset ---
INPUT_FILE = 'data/majhitar_active_places_only.csv'
df = pd.read_csv(INPUT_FILE)

# --- 1. VADER ---
print("Benchmarking VADER...")
v_start = time.perf_counter()

df['Vader_Score'] = df['Review'].apply(
    lambda x: vader.polarity_scores(str(x))['compound']
)

df['Vader_Pred'] = df['Vader_Score'].apply(
    lambda s: 'Positive' if s >= 0.05 else ('Negative' if s <= -0.05 else 'Neutral')
)

v_time = time.perf_counter() - v_start

# --- 2. RoBERTa ---
print("Benchmarking RoBERTa...")
r_start = time.perf_counter()

results = roberta_pipe(list(df['Review']), batch_size=16, truncation=True)

df['Roberta_Label'] = [label_map[r['label']] for r in results]
df['Roberta_Score'] = [r['score'] for r in results]

r_time = time.perf_counter() - r_start

# --- 3. Metrics ---
agreement = (df['Vader_Pred'] == df['Roberta_Label']).mean() * 100

print("\n" + "="*50)
print("NUMERICAL PERFORMANCE DATA")
print("="*50)
print(f"Total Reviews       : {len(df)}")
print(f"Model Agreement     : {agreement:.2f}%")
print(f"VADER Throughput    : {len(df)/v_time:.0f} reviews/sec")
print(f"RoBERTa Throughput  : {len(df)/r_time:.0f} reviews/sec")
print(f"Speed Difference    : {r_time/v_time:.1f}x slower")
print("="*50)

# --- Plotting ---
plt.style.use('ggplot')
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Comparative Analysis: VADER vs RoBERTa', fontsize=16)

# Plot 1: Sentiment Comparison
comp_df = pd.DataFrame({
    'Sentiment': ['Positive', 'Neutral', 'Negative'],
    'VADER': [
        sum(df['Vader_Pred'] == 'Positive'),
        sum(df['Vader_Pred'] == 'Neutral'),
        sum(df['Vader_Pred'] == 'Negative')
    ],
    'RoBERTa': [
        sum(df['Roberta_Label'] == 'Positive'),
        sum(df['Roberta_Label'] == 'Neutral'),
        sum(df['Roberta_Label'] == 'Negative')
    ]
}).melt(id_vars='Sentiment', var_name='Model', value_name='Count')

sns.barplot(x='Sentiment', y='Count', hue='Model', data=comp_df, ax=axes[0, 0])
axes[0, 0].set_title('Sentiment Distribution Comparison')

# Plot 2: Latency
latency = pd.DataFrame({
    'Model': ['VADER', 'RoBERTa'],
    'Time (ms)': [v_time/len(df)*1000, r_time/len(df)*1000]
})

sns.barplot(x='Model', y='Time (ms)', data=latency, ax=axes[0, 1])
axes[0, 1].set_title('Inference Latency per Review')

# Plot 3: Heatmap
ct = pd.crosstab(df['Vader_Pred'], df['Roberta_Label'])
sns.heatmap(ct, annot=True, fmt='d', cmap='YlGnBu', ax=axes[1, 0])
axes[1, 0].set_title('Model Agreement Heatmap')

# Plot 4: Scatter Comparison
label_map_num = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
df['Roberta_Num'] = df['Roberta_Label'].map(label_map_num)

df['Roberta_Jitter'] = df['Roberta_Num'] + np.random.uniform(-0.2, 0.2, len(df))

sns.scatterplot(
    x='Roberta_Jitter',
    y='Vader_Score',
    hue='Roberta_Label',
    data=df,
    ax=axes[1, 1],
    s=50
)

axes[1, 1].axhline(0.05, linestyle='--')
axes[1, 1].axhline(-0.05, linestyle='--')
axes[1, 1].set_xticks([-1, 0, 1])
axes[1, 1].set_xticklabels(['Negative', 'Neutral', 'Positive'])
axes[1, 1].set_title('VADER vs RoBERTa Scores')

# --- Save Outputs ---
plt.tight_layout()
plt.savefig("results/Final_Comparison_Dashboard.png", dpi=300)
plt.show()

df.to_csv("results/final_comparison_data.csv", index=False)

print("\nAnalysis complete. Results saved in 'results/' folder.")
