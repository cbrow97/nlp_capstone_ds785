# %%
from sentiment.sentiment_models import RobertaBaseSentiment
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(df, labels):
    cm = confusion_matrix(s_df["rating_category"], s_df["predicted_sentiment"], labels=labels)
    ax = plt.subplot()

    accuracy = np.trace(cm) / float(np.sum(cm))
    stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

    ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    ax.set_xlabel(stats_text, fontsize=16)



# %%
"""
Make predictions using the three originally planned labels of negative, positive, and neutral.
"""
s_df = pd.read_csv("/home/ubuntu/consumer-feedback-analyzer/src/eda/clean_labeled_review_sentences.csv")

neg_pos_neu_mapper = {
    "negative": "negative",
    "somewhat negative": "negative",
    "neutral": "neutral",
    "somewhat positive": "positive",
    "positive": "positive",
}

s_df["rating_category"] = s_df["label"].str.lower().map(neg_pos_neu_mapper)

model = RobertaBaseSentiment()
s_df[f"predicted_sentiment"] = s_df["review_sentences"].apply(lambda x: model.predict(x))

labels = ['negative', 'positive', 'neutral']
plot_confusion_matrix(s_df, labels)

# %%

"""
Make predictions when neutral labels are mapped to negative.
"""
neg_pos_mapper = {
    "neutral": "negative",
    "negative": "negative",
    "positive": "positive",
}

s_df["predicted_sentiment"] = s_df["predicted_sentiment"].str.lower().map(neg_pos_mapper)

labels = ['negative', 'positive']
index_to_drop = s_df[s_df["rating_category"] == "negative"].sample(n=717-267).index
s_df = s_df.drop(index=index_to_drop)
plot_confusion_matrix(s_df, labels)

# %%
"""
Apply predictions using negative and positive labels only.
"""
neg_pos_mapper = {
    "neutral": "negative",
    "negative": "negative",
    "positive": "positive",
}
s_df = pd.read_csv("/home/ubuntu/consumer-feedback-analyzer/src/model_orchestration/review_sentences.csv")

model = RobertaBaseSentiment()
s_df["predicted_sentiment"] = s_df["review_sentences"].apply(lambda x: model.predict(x))
s_df["predicted_sentiment"] = s_df["predicted_sentiment"].str.lower().map(neg_pos_mapper)

s_df.to_csv("review_sentences.csv")

