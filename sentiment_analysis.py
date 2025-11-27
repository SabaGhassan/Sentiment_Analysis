import pandas as pd
from textblob import TextBlob

# 1. Load dataset
print("Loading dataset...")
df = pd.read_csv("reviews.csv")

# 2. Define Sentiment Function
def analyze_sentiment(text):
    try:
        polarity = TextBlob(text).sentiment.polarity
        return polarity
    except:
        return 0

# 3. Apply Sentiment Analysis
print("Applying sentiment analysis...")
df["sentiment_score"] = df["review"].apply(analyze_sentiment)

# 4. Add Sentiment Label
def get_label(score):
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"

df["sentiment_label"] = df["sentiment_score"].apply(get_label)

# 5. Save Output File
output_file = "sentiment_output.csv"
df.to_csv(output_file, index=False)
print(f"Sentiment analysis complete! Output saved as {output_file}")

# 6. Display a summary
print("\n=== SENTIMENT SUMMARY ===")
print(df["sentiment_label"].value_counts())

print("\n=== SAMPLE OUTPUT ===")
print(df.head())