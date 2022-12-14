import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter


sns.set_theme(style="white", palette="pastel")
plt.rc('font', size=18)
text_df = pd.read_pickle("/home/ubuntu/consumer-feedback-miner/src/pre_process/cleaned_review_text.pkl")


"""
Figure 3 - Top 25 Most Frequent Words in the Yelp Review Text
"""
fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))
fig.tight_layout(pad=5)

word_count = Counter(" ".join(text_df["norm_text"]).split())
word_count_df = pd.DataFrame(word_count.items(),columns=["word", "count"])
word_count_df = word_count_df.sort_values("count", ascending=False).reset_index(drop=True)
sns.barplot(word_count_df.head(25)["count"], word_count_df.head(25)["word"], ax=ax)
ax.set_title("25 Most Frequent Words", fontsize=16)
ax.set_xlabel("Frequency", fontsize=16)
ax.set_ylabel("Word", fontsize=16)
ax.tick_params(axis="y", labelsize="16")


"""
Figure 4 - Top 25 Most Frequent Bigrams in the Yelp Review Text
"""
fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))
fig.tight_layout(pad=5)

word_count = Counter([bigram for bigram in text_df["bigram"].tolist() for bigram in bigram])

word_count_df = pd.DataFrame(word_count.items(),columns=["word", "count"])
word_count_df = word_count_df.sort_values("count", ascending=False).reset_index(drop=True)
sns.barplot(word_count_df.head(25)["count"], word_count_df.head(25)["word"], ax=ax)
ax.set_title("25 Most Frequent Bigrams", fontsize=16)
ax.set_xlabel("Frequency", fontsize=16)
ax.set_ylabel("Bigram", fontsize=16)
ax.tick_params(axis="y", labelsize="12")


"""
Figure 5 - Review Volume Over Time
"""
fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(15,10))
fig.tight_layout(pad=5)

fig_1_data = text_df.groupby("review_month_year")["review_id"].count().reset_index().rename(columns={"review_id": "review_count"})

sns.lineplot(x="review_month_year", y="review_count", data=fig_1_data, ax=ax1, c="purple")
ax1.set_title("Review Count by Month", fontsize=16)
ax1.set_xlabel("Month of Review", fontsize=16)
ax1.set_ylabel("Review Count", fontsize=16)
ax1.tick_params(axis="y", labelsize="16")

fig_2_data = text_df.groupby(["review_month_year", "rating_category"])["review_id"].count().reset_index().rename(columns={"review_id": "review_count"})

sns.lineplot(x="review_month_year", y="review_count", hue="rating_category", data=fig_2_data, style="rating_category", ax=ax2)
ax2.set_title("Review Count by Month and Rating Category", fontsize=16)
ax2.set_xlabel("Month of Review", fontsize=16)
ax2.set_ylabel("Review Count", fontsize=16)
ax2.legend(title="Rating Category")
ax2.tick_params(axis="y", labelsize="16")

"""
Figure 6 - Count of Rating Category
"""
plt.figure(figsize=(9, 6))
rating_category_df = pd.DataFrame(Counter(text_df["rating_category"]).items(), columns=["rating_category", "count"])
total = rating_category_df["count"].sum()
ax = sns.barplot(x="rating_category", y="count", data=rating_category_df)

for p in ax.patches:
    percentage = "{:.1f}%".format(100 * p.get_height()/total)
    x = p.get_x() + (p.get_width() / 2)
    y = (p.get_height() / 2)
    ax.annotate(percentage + "\n of total", (x, y), ha="center")

ax.set_title("Review Count by Rating Category", fontsize=16)
ax.set_xlabel("Rating Category", fontsize=16)
ax.set_ylabel("Review Count", fontsize=16)
ax.tick_params(axis="y", labelsize="16")


"""
Figure 7 - Review Word and Sentence Volume
"""
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15, 8))
fig.tight_layout(pad=5)

sns.histplot(x="word_count", data=text_df, ax=ax1)
ax1.set_title("Number of Words per Review", fontsize=16)
ax1.set_xlabel("Word Count", fontsize=16)
ax1.set_ylabel("Review Count", fontsize=16)
ax1.tick_params(axis="y", labelsize="16")

sns.histplot(x="sentence_count", data=text_df, ax=ax2)
ax2.set_title("Number of Sentences per Review", fontsize=16)
ax2.set_xlabel("Sentence Count", fontsize=16)
ax2.set_ylabel("Review Count", fontsize=16)
ax2.tick_params(axis="y", labelsize="16")


"""
Figure 8 - Word and Sentence Volume by Rating Category
"""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 8))

sns.histplot(x="word_count", data=text_df, hue="rating_category", kde=True, ax=ax1)
ax1.set_title("Word Count by Rating Category", fontsize=16)
ax1.set_xlabel("Word Count", fontsize=16)
ax1.set_ylabel("Review Count", fontsize=16)
ax1.get_legend().set_title("Rating Category")
ax1.tick_params(axis="y", labelsize="16")
ax1.legend(fontsize=16)

sns.histplot(x="sentence_count", data=text_df, hue="rating_category", kde=True, ax=ax2)
ax2.set_title("Sentence Count by Rating Category", fontsize=16)
ax2.set_xlabel("Sentence Count", fontsize=16)
ax2.set_ylabel("Review Count", fontsize=16)
ax2.get_legend().set_title("Rating Category")
ax2.tick_params(axis="y", labelsize="16")


"""
Figure 9 - Top 10 Most Frequent Bigrams and Trigrams Split by Rating Category
"""
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 10))
fig.tight_layout(w_pad=12, h_pad=4)

def get_gram_count_by_rating_category(text_df: pd.DataFrame, ngram: str, rating_category: str) -> Counter:
    return Counter(
        [gram for gram in text_df[text_df["rating_category"] == rating_category][ngram].tolist() for gram in gram]
    )

bigram_count_negative = get_gram_count_by_rating_category(text_df, "bigram", "negative")
bigram_count_positive = get_gram_count_by_rating_category(text_df, "bigram", "positive")
bigram_count_neutral = get_gram_count_by_rating_category(text_df, "bigram", "neutral")

def plot_ngram_frequency_barplot(text_df: pd.DataFrame, ax, ngram: str, rating_category: str):
    count = get_gram_count_by_rating_category(text_df, ngram, rating_category)
    count_df = pd.DataFrame(count.items(), columns=[ngram, "count"])
    count_df = count_df.sort_values("count", ascending=False).reset_index(drop=True)
    sns.barplot(count_df.head(10)["count"], count_df.head(10)[ngram], ax=ax)
    ax.set_title(f"10 Most Frequent {ngram.title()}s in {rating_category.title()} Reviews", fontsize=16)
    ax.set_xlabel("Frequency", fontsize=16)
    ax.set_ylabel("", fontsize=16)
    ax.tick_params(axis="y", labelsize="12")

    return ax

for ax, gram in zip([ax1, ax2], ["bigram", "trigram"]):
    plot_ngram_frequency_barplot(text_df, ax, gram, rating_category="negative")

for ax, gram in zip([ax3, ax4], ["bigram", "trigram"]):
    plot_ngram_frequency_barplot(text_df, ax, gram, rating_category="positive")

for ax, gram in zip([ax5, ax6], ["bigram", "trigram"]):
    plot_ngram_frequency_barplot(text_df, ax, gram, rating_category="neutral")
