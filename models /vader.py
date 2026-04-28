import time
import warnings
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")

# --- Ensure folders exist ---
os.makedirs("results", exist_ok=True)

# --- NLTK setup ---
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# --- Configuration ---
INPUT_FILE = 'data/majhitar_active_places_only.csv'
OUTPUT_CSV = 'results/model1_vader_results.csv'
OUTPUT_GRAPH = 'results/model1_vader_plots.png'
NEUTRAL_THRESH = 0.05

# --- Load data ---
print("Loading dataset ...")
df = pd.read_csv(INPUT_FILE)
print(f"{len(df)} rows loaded.\n")

# --- Validation ---
for col in ['Review', 'Restaurant']:
    if col not in df.columns:
        raise SystemExit(f"Missing column: {col}")

has_label = 'Sentiment' in df.columns
has_rating = 'Rating' in df.columns

# --- Clean data ---
df['Review'] = df['Review'].astype(str).str.strip()
df['Restaurant'] = df['Restaurant'].astype(str).str.strip()
df = df[df['Review'].str.lower() != 'nan']
df = df[df['Review'] != ''].reset_index(drop=True)

# --- Initialize VADER ---
vader = SentimentIntensityAnalyzer()

def vader_label(score):
    if score >= NEUTRAL_THRESH:
        return 'Positive'
    elif score <= -NEUTRAL_THRESH:
        return 'Negative'
    return 'Neutral'

# --- Scoring ---
print("Scoring reviews with VADER ...")
t0 = time.perf_counter()

scores = df['Review'].apply(lambda x: vader.polarity_scores(str(x)))

df['Vader_Score'] = scores.apply(lambda x: x['compound'])
df['Vader_Pos']   = scores.apply(lambda x: x['pos'])
df['Vader_Neg']   = scores.apply(lambda x: x['neg'])
df['Vader_Neu']   = scores.apply(lambda x: x['neu'])
df['Vader_Pred']  = df['Vader_Score'].apply(vader_label)

elapsed = time.perf_counter() - t0
throughput = len(df) / elapsed

print(f"Done in {elapsed:.3f}s ({throughput:.0f} reviews/sec)\n")

# --- Evaluation ---
if has_label:
    df['Ground_Truth'] = df['Sentiment'].str.strip().str.capitalize()
    print("Classification Report:\n")
    print(classification_report(df['Ground_Truth'], df['Vader_Pred']))

# --- Leaderboard ---
leaderboard = (
    df.groupby('Restaurant')
    .agg(
        Avg_Score=('Vader_Score', 'mean'),
        Review_Count=('Vader_Score', 'count'),
        Positive_Pct=('Vader_Pred', lambda x: (x == 'Positive').mean() * 100),
        Negative_Pct=('Vader_Pred', lambda x: (x == 'Negative').mean() * 100)
    )
    .sort_values('Avg_Score', ascending=False)
    .reset_index()
)

print("\nTop Restaurants:")
print(leaderboard.head(10))

# --- Save CSV ---
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nResults saved → {OUTPUT_CSV}")

# --- FULL VISUALIZATION (ALL PLOTS RESTORED) ---
COLORS = {'Positive': '#5a9e6f', 'Neutral': '#7eaec4', 'Negative': '#c4735a'}
plt.style.use('seaborn-v0_8-whitegrid')

fig = plt.figure(figsize=(18, 10))
fig.suptitle('Model 1 — VADER Sentiment Analysis', fontsize=14)

# Plot 1: Pie Chart
ax1 = fig.add_subplot(2, 3, 1)
counts = df['Vader_Pred'].value_counts()
ax1.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
        colors=[COLORS[s] for s in counts.index], startangle=140)
ax1.set_title('Sentiment Distribution')

# Plot 2: Histogram
ax2 = fig.add_subplot(2, 3, 2)
for label, grp in df.groupby('Vader_Pred'):
    ax2.hist(grp['Vader_Score'], bins=15, alpha=0.65,
             color=COLORS[label], label=label)
ax2.axvline(NEUTRAL_THRESH, linestyle='--')
ax2.axvline(-NEUTRAL_THRESH, linestyle='--')
ax2.set_title('Score Distribution')
ax2.legend()

# Plot 3: Top Restaurants
ax3 = fig.add_subplot(2, 3, 3)
top10 = leaderboard.head(10)
ax3.barh(top10['Restaurant'], top10['Avg_Score'])
ax3.set_title('Top Restaurants')
ax3.invert_yaxis()

# Plot 4: Sub-score Distribution
ax4 = fig.add_subplot(2, 3, 4)
sns.boxplot(data=df[['Vader_Pos','Vader_Neg','Vader_Neu']], ax=ax4)
ax4.set_title('VADER Sub-scores')

# Plot 5: Countplot
ax5 = fig.add_subplot(2, 3, 5)
sns.countplot(x='Vader_Pred', data=df, palette=COLORS, ax=ax5)
ax5.set_title('Sentiment Count')

# Plot 6: Rating vs Score OR fallback
ax6 = fig.add_subplot(2, 3, 6)
if has_rating:
    ax6.scatter(df['Rating'], df['Vader_Score'], alpha=0.5)
    ax6.set_title('Rating vs Score')
else:
    ax6.barh(leaderboard.head(10)['Restaurant'],
             leaderboard.head(10)['Positive_Pct'])
    ax6.set_title('Top Positive %')

plt.tight_layout()
plt.savefig(OUTPUT_GRAPH, dpi=150)
plt.show()

print("\nVADER model complete.")
