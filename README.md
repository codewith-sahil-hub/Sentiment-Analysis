# CodeAlpha_SentimentAnalysis

## Task 4: Sentiment Analysis — Data Analytics Internship (CodeAlpha)

Classifies product reviews as positive/negative/neutral and detects
specific emotions, using a **custom lexicon-based NLP engine** built
from scratch (no external sentiment libraries required).

## Files

```
CodeAlpha_SentimentAnalysis/
├── lexicon_sentiment.py         # The sentiment/emotion engine (core NLP logic)
├── sentiment_analysis.py        # Main analysis script (run this)
├── generate_sample_reviews.py   # Generates sample review data
├── product_reviews.csv          # Sample dataset (250 Amazon-style reviews)
├── charts/                      # Output visualizations
└── requirements.txt
```

> **Note:** `product_reviews.csv` is generated sample data, used because
> live review scraping wasn't available in this environment. To use real
> data, point `web_scraper_template.py` from Task 1 at an actual review
> source (checking robots.txt/ToS first), then feed the resulting CSV into
> `sentiment_analysis.py` — no changes needed as long as it has a
> `review_text` column.

## How the sentiment engine works

`lexicon_sentiment.py` implements the same core approach as VADER
(Valence Aware Dictionary and sEntiment Reasoner), built from scratch:

1. **Polarity lexicon** — ~50 hand-scored positive/negative words
   (e.g. "excellent" = +3.0, "terrible" = -3.0)
2. **Negation handling** — "not good" flips the polarity of "good"
3. **Intensifiers/dampeners** — "very good" scores higher than "good";
   "somewhat good" scores lower
4. **Compound score** — word scores are summed and normalized to a
   -1 to +1 scale, then thresholded into positive/neutral/negative
5. **Emotion lexicon** — a separate word-to-emotion mapping (joy, anger,
   sadness, trust, surprise, fear) tags specific emotions per review

This is a rule-based, fully transparent implementation — good for
understanding *how* lexicon-based sentiment scoring works. A production
system would typically use a trained model or the full VADER/NLTK lexicon
for broader vocabulary coverage.

## Analysis workflow

1. Classify every review (positive/negative/neutral)
2. Validate sentiment against star ratings (a natural ground-truth check)
3. Detect emotions per review
4. Break down sentiment by product (business-actionable)
5. Track sentiment trend over time
6. Translate findings into recommendations

## How to run

```bash
pip install -r requirements.txt
python sentiment_analysis.py
```

## Key findings (on the sample dataset)

- 64% positive / 22% negative / 14% neutral overall
- Sentiment label agreed with the expected star-rating bucket 84% of
  the time — a reasonable but not perfect proxy, confirming text and
  star ratings capture related but distinct signals
- Joy and trust were the most common detected emotions in positive
  reviews; sadness and anger concentrated in negative ones
- Sentiment varied meaningfully by product — a clear signal for
  prioritizing product/quality review vs. marketing case studies

## Notes

Completed as part of the CodeAlpha Data Analytics Internship —
Task 4: Sentiment Analysis.
