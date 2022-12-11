from sentiment.sentiment_baseline_models import VaderSentiment, TextBlobSentiment, RobertaBaseSentiment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from make_confusion_matrix import make_confusion_matrix


def plot_roc_auc(true_labels, prediction_probs):
    plt.figure(figsize=(15, 8))
    for i, label in enumerate(["negative", "neutral", "positive"]):
        fpr, tpr, _ = roc_curve(
            true_labels[:,i].astype(int), 
            prediction_probs[:, i])
        auc = roc_auc_score(
            true_labels[:,i].astype(int), prediction_probs[:, i])
        plt.plot(fpr, tpr, label='%s %g' % (label, auc))
    plt.xlabel('False Positive Rate', fontdict={"size": 18})
    plt.ylabel('True Positive Rate', fontdict={"size": 18})
    plt.legend(loc='lower right', prop={'size': 18})
    plt.title('RoBERTa Baseline Sentiment Model - AUC ROC', fontdict={"size": 18})
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

def plot_baseline_auc_roc(model):
    true_labels_ = [[0 for _ in range(0, 3)] for label in range(0, len(sentence_df))]

    label_mapper = {
        "negative": 0,
        "neutral": 1,
        "positive": 2,
    }

    for i, row in sentence_df.reset_index(drop=True).iterrows():
        true_labels_[i][label_mapper[row["rating_category"]]] = 1

    true_labels_ = np.array(true_labels_)

    preds = np.array([list(model.predict(row["review_sentences"]).logits[0].detach().numpy()) for _, row in sentence_df.iterrows()])
    plot_roc_auc(true_labels_, preds)

"""
Prepare full review text data
"""
text_df = pd.read_pickle("/home/ubuntu/consumer-feedback-analyzer/src/pre_process/cleaned_review_text.pkl")
text_df = text_df[["review_text", "review_sentences", "rating_category"]]
index_to_drop = text_df[text_df["rating_category"] == "negative"].sample(n=492-232).index
text_df = text_df.drop(index=index_to_drop)

"""
Prepare manually labeled sentences data
"""
sentence_df = pd.read_csv(f"clean_labeled_review_sentences.csv")

sentence_df["rating_category"] = sentence_df["label"].str.lower().map({
    "negative": "negative",
    "somewhat negative": "negative",
    "neutral": "neutral",
    "somewhat positive": "positive", 
    "positive": "positive",
    })
sentence_df = sentence_df[["review_sentences", "rating_category"]]
index_to_drop = sentence_df[sentence_df["rating_category"] == "negative"].sample(n=719-265).index
sentence_df = sentence_df.drop(index=index_to_drop)


"""
Predict sentiment of review text and review sentences using the three baseline models 
"""
models = {
    "vader": VaderSentiment(),
    "textblob": TextBlobSentiment(),
    "RoBERTa": RobertaBaseSentiment(),
}
for model_type, model in models.items():
    text_df[f"{model_type}_sentiment"] = text_df["review_text"].apply(lambda x: model.predict(x))
    sentence_df[f"{model_type}_sentence_sentiment"] = sentence_df["review_sentences"].apply(lambda x: model.predict(x))


"""
Plot the confusion matrices for the predictions from all three models for both review text and review sentences
"""
sns.set(font_scale=1.2)
labels = ['negative', 'positive', 'neutral']

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 6))
fig.tight_layout(pad=5)

for model_type, ax in zip(models.keys(), (ax1, ax2, ax3)):
    cf = confusion_matrix(text_df["rating_category"], text_df[f"{model_type}_sentiment"], labels=labels)
    make_confusion_matrix(cf, title=f"{model_type.title()} Model Performance", cbar=False, categories=labels, ax=ax)


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 6))
fig.tight_layout(pad=5)

for model_type, ax in zip(models.keys(), (ax1, ax2, ax3)):
    cf = confusion_matrix(sentence_df["rating_category"], sentence_df[f"{model_type}_sentence_sentiment"], labels=labels)
    make_confusion_matrix(cf, title=f"{model_type.title()} Model Performance", cbar=False, categories=labels, ax=ax)

