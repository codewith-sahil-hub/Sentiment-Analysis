"""
Generates a SAMPLE review dataset (Amazon-style product reviews) for
demonstrating the sentiment pipeline offline. Star ratings loosely drive
which sentence templates get used, with noise, so sentiment doesn't
perfectly agree with rating -- like real data.

Replace this with real scraped reviews (see web_scraper_template.py
from Task 1, pointed at a real review source) before final submission.
"""
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

random.seed(7)
np.random.seed(7)

PRODUCTS = ["Wireless Earbuds", "Yoga Mat", "Coffee Maker", "Desk Lamp",
            "Backpack", "Bluetooth Speaker", "Running Shoes", "Water Bottle"]

POSITIVE_SENTENCES = [
    "Absolutely love this product, exceeded my expectations!",
    "Great quality and fast shipping, highly recommend.",
    "This is the best purchase I've made this year.",
    "Works perfectly and looks amazing too.",
    "Really impressed with the build quality, feels sturdy and reliable.",
    "Comfortable, easy to use, and worth every penny.",
    "Fantastic value, I'm very happy with this.",
    "Excellent product, would buy again without hesitation.",
]
NEGATIVE_SENTENCES = [
    "Terrible quality, broke after two days of use.",
    "Really disappointed, this was a waste of money.",
    "The product arrived defective and customer service was unhelpful.",
    "Cheap materials, feels flimsy and uncomfortable.",
    "Not worth the price, I regret this purchase.",
    "Awful experience, would not recommend to anyone.",
    "Stopped working after a week, very frustrating.",
    "Poor design, difficult to use and overpriced.",
]
NEUTRAL_SENTENCES = [
    "It's okay, does what it says but nothing special.",
    "Average product, some pros and some cons.",
    "Not bad, not great either. It works.",
    "Decent for the price, though I expected slightly more.",
    "It arrived on time and matches the description.",
    "Fairly standard product, met basic expectations.",
]

def make_review(star: int) -> str:
    if star >= 4:
        pool, extra = POSITIVE_SENTENCES, NEUTRAL_SENTENCES
        weight = 0.85
    elif star == 3:
        pool, extra = NEUTRAL_SENTENCES, POSITIVE_SENTENCES + NEGATIVE_SENTENCES
        weight = 0.7
    else:
        pool, extra = NEGATIVE_SENTENCES, NEUTRAL_SENTENCES
        weight = 0.85

    n_sentences = random.choice([1, 1, 2])
    sentences = []
    for _ in range(n_sentences):
        source = pool if random.random() < weight else extra
        sentences.append(random.choice(source))
    return " ".join(sentences)


N = 250
rows = []
start_date = datetime(2026, 1, 1)

for i in range(N):
    star = np.random.choice([1, 2, 3, 4, 5], p=[0.08, 0.09, 0.15, 0.30, 0.38])
    review_date = start_date + timedelta(days=int(np.random.uniform(0, 240)))
    rows.append({
        "review_id": i + 1,
        "product": random.choice(PRODUCTS),
        "star_rating": int(star),
        "review_date": review_date.strftime("%Y-%m-%d"),
        "review_text": make_review(star),
    })

df = pd.DataFrame(rows).sort_values("review_date").reset_index(drop=True)
df.to_csv("/home/claude/sentiment_project/product_reviews.csv", index=False)
print(f"Generated {len(df)} sample reviews across {df['product'].nunique()} products")

