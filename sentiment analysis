"""
Task 4: Sentiment Analysis
==============================
Dataset: product_reviews.csv (Amazon-style product reviews, sample data)

Workflow:
  1. Classify each review as positive / negative / neutral
  2. Detect specific emotions using a lexicon
  3. Analyze patterns across products and time
  4. Validate against star ratings (a natural ground-truth proxy)
  5. Turn findings into decision-support recommendations

Run with: python sentiment_analysis.py
Outputs: printed report + PNG charts in ./charts/
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from lexicon_sentiment import score_text, detect_emotions, tokenize

CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)
sns.set_style("whitegrid")


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ----------------------------------------------------------------------
# 1. Load + classify
# ----------------------------------------------------------------------
df = pd.read_csv("product_reviews.csv")

section("STEP 1: Classifying reviews")
scores = df["review_text"].apply(score_text)
df["sentiment_score"] = scores.apply(lambda s: s["compound"])
df["sentiment_label"] = scores.apply(lambda s: s["label"])
df["emotions"] = df["review_text"].apply(detect_emotions)

print(df[["review_text", "sentiment_label", "sentiment_score"]].head(5).to_string(index=False))


# ----------------------------------------------------------------------
# 2. Overall sentiment distribution
# ----------------------------------------------------------------------
section("STEP 2: Overall sentiment distribution")
dist = df["sentiment_label"].value_counts()
print(dist)
print(f"\n{(dist.get('positive', 0) / len(df) * 100):.1f}% positive, "
      f"{(dist.get('negative', 0) / len(df) * 100):.1f}% negative, "
      f"{(dist.get('neutral', 0) / len(df) * 100):.1f}% neutral")

plt.figure(figsize=(6, 4))
colors = {"positive": "#00875A", "neutral": "#6B778C", "negative": "#DE350B"}
order = ["positive", "neutral", "negative"]
sns.barplot(x=dist.reindex(order).index, y=dist.reindex(order).values,
            hue=dist.reindex(order).index,
            palette=[colors[o] for o in order], legend=False)
plt.title("Overall Sentiment Distribution")
plt.ylabel("Number of reviews")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/01_sentiment_distribution.png", dpi=150)
plt.close()


# ----------------------------------------------------------------------
# 3. Validation: does sentiment agree with star rating?
# ----------------------------------------------------------------------
section("STEP 3: Validating against star ratings")
agreement = (
    ((df["star_rating"] >= 4) & (df["sentiment_label"] == "positive")) |
    ((df["star_rating"] == 3) & (df["sentiment_label"] == "neutral")) |
    ((df["star_rating"] <= 2) & (df["sentiment_label"] == "negative"))
).mean()
print(f"Sentiment label agrees with expected star-rating bucket "
      f"{agreement * 100:.1f}% of the time.")
print("(Disagreements are expected -- text sentiment and star rating "
      "capture related but different signals, e.g. a 3-star review can "
      "still read as mostly positive text.)")

plt.figure(figsize=(7, 4.5))
sns.boxplot(data=df, x="star_rating", y="sentiment_score",
            hue="star_rating", palette="RdYlGn", legend=False)
plt.title("Sentiment Score vs. Star Rating")
plt.xlabel("Star rating")
plt.ylabel("Sentiment compound score")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/02_sentiment_vs_rating.png", dpi=150)
plt.close()


# ----------------------------------------------------------------------
# 4. Emotion detection
# ----------------------------------------------------------------------
section("STEP 4: Emotion patterns")
all_emotions = [e for emos in df["emotions"] for e in emos if e != "none detected"]
emotion_counts = Counter(all_emotions)
print("Emotion frequency across all reviews:")
for emo, count in emotion_counts.most_common():
    print(f"  {emo}: {count}")

if emotion_counts:
    plt.figure(figsize=(7, 4.5))
    emo_series = pd.Series(emotion_counts).sort_values(ascending=True)
    plt.barh(emo_series.index, emo_series.values, color="#6554C0")
    plt.title("Detected Emotions Across Reviews")
    plt.xlabel("Mentions")
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/03_emotion_frequency.png", dpi=150)
    plt.close()


# ----------------------------------------------------------------------
# 5. Sentiment by product (business-relevant breakdown)
# ----------------------------------------------------------------------
section("STEP 5: Sentiment by product")
by_product = df.groupby("product")["sentiment_score"].mean().sort_values()
print(by_product.round(3))

plt.figure(figsize=(8, 5))
colors_p = ["#DE350B" if v < 0 else "#00875A" for v in by_product.values]
plt.barh(by_product.index, by_product.values, color=colors_p)
plt.axvline(0, color="#42526E", linewidth=0.8)
plt.title("Average Sentiment Score by Product")
plt.xlabel("Avg. sentiment (compound score)")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/04_sentiment_by_product.png", dpi=150)
plt.close()


# ----------------------------------------------------------------------
# 6. Sentiment trend over time
# ----------------------------------------------------------------------
section("STEP 6: Sentiment trend over time")
df["review_date"] = pd.to_datetime(df["review_date"])
df["month"] = df["review_date"].dt.to_period("M").astype(str)
monthly = df.groupby("month")["sentiment_score"].mean()
print(monthly.round(3))

plt.figure(figsize=(8, 4.5))
plt.plot(monthly.index, monthly.values, marker="o", color="#0052CC")
plt.axhline(0, color="#DFE1E6", linewidth=1)
plt.title("Average Sentiment Trend Over Time")
plt.xlabel("Month")
plt.ylabel("Avg. sentiment score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/05_sentiment_trend.png", dpi=150)
plt.close()

print(f"\nAll charts saved to ./{CHART_DIR}/")


# ----------------------------------------------------------------------
# 7. Business takeaways
# ----------------------------------------------------------------------
worst_product = by_product.index[0]
best_product = by_product.index[-1]

section("SUMMARY: What this tells the business")
print(f"""
- {(dist.get('negative', 0) / len(df) * 100):.0f}% of reviews are net-negative
  in tone -- worth a closer read even where star ratings look acceptable.
- '{worst_product}' has the lowest average sentiment ({by_product.iloc[0]:.3f})
  -- flag for product/quality review.
- '{best_product}' has the strongest sentiment ({by_product.iloc[-1]:.3f})
  -- a candidate for marketing case studies or testimonials.
- Most common detected emotion: {emotion_counts.most_common(1)[0][0] if emotion_counts else 'n/a'}
  -- {'trust/satisfaction language dominates positive reviews' if emotion_counts and emotion_counts.most_common(1)[0][0] in ('trust','joy') else 'frustration/disappointment language is common in negative reviews'}.
- Sentiment vs. rating agreement ({agreement*100:.0f}%) confirms text sentiment
  is a reasonably reliable signal, but not a full substitute for star ratings
  -- best used alongside them, not instead of them.
""")

