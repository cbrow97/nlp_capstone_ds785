import pandas as pd
from datasets import load_dataset, load_dataset
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModelForSequenceClassification, Trainer
from google.colab import drive
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import torch as pt

def generate_train_test_valid_files(fp):
    data = pd.read_csv(f"{fp}/clean_labeled_review_sentences.csv")

    data["label"] = data["label"].str.lower().map({
          "negative": 0,
          "somewhat negative": 0,
          "neutral": 1,
          "somewhat positive": 2, 
          "positive": 2,
        })

    data = data[["review_sentences", "label"]]
    train_df = data.sample(frac=0.7)
    test_df = data[~data.index.isin(train_df.index)].sample(frac=0.5)
    validation_df = data[~data.index.isin(test_df.index) & ~data.index.isin(train_df.index)]

    train_df.to_csv(f"{fp}/review_sentences_train.csv", index=False)
    test_df.to_csv(f"{fp}/review_sentences_test.csv", index=False)
    validation_df.to_csv(f"{fp}/review_sentences_validation.csv", index=False)

def load_data(fp):
    data_files = {
        "train": "review_sentences_train.csv",
        "test": "review_sentences_test.csv",
        "validation": "review_sentences_validation.csv"
      }
    dataset = load_dataset("/content/drive/MyDrive/roberta_sentiment/", data_files=data_files)
    return dataset

def tokenize_function(example):
    return tokenizer(example["review_sentences"], truncation=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)
  
def plot_auc_roc(true_labels, prediction_probs):
    plt.figure(figsize=(15, 8))
    for i, label in enumerate(["negative", "neutral", "positive"]):
        fpr, tpr, _ = metrics.roc_curve(
            true_labels[:,i].astype(int), prediction_probs[:, i])
        auc = metrics.roc_auc_score(
            true_labels[:,i].astype(int), prediction_probs[:, i])
        plt.plot(fpr, tpr, label='%s %g' % (label, auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.title('RoBERTa Tuned Sentiment Model - AUC ROC')


def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=3, return_dict=True)

def predict(text):
    label_mapper = {
        0: "negative",
        1: "neutral",
        2: "positive"
    }
    tokenized_text = tokenizer.encode(
                              text,
                              truncation=True,
                              padding=True,
                              return_tensors="pt"
                          )

    prediction = model(tokenized_text)
    prediction_logits = prediction[0]
    m = pt.nn.Softmax(dim=1)
    prediction_probs = m(prediction_logits)[0].detach().numpy()
    predicted_label = prediction_probs.argmax()

    return label_mapper[predicted_label]


"""
 - Tokenize data
 - Train model using hyperparameter_search
 - Save best model
"""
fp = "/content/drive/MyDrive/roberta_sentiment"

dataset = load_data(fp)

checkpoint = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    
training_args = TrainingArguments("roberta-sentence-trainer-default", evaluation_strategy="steps")

trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

best_trial = trainer.hyperparameter_search(
    direction="maximize", 
    backend="ray", 
    n_trials=10
)

best_trial.save_model(f"{fp}/twitter-roberta-base-sentiment-latest_3_label")


"""
Generate predictions using test set
"""
predictions = trainer.predict(tokenized_datasets["test"])
preds = np.argmax(predictions.predictions, axis=-1)
true_labels = np.array(tokenized_datasets["test"].data["label"])


"""
Print performance metrics and plot AUC ROC
"""
print(metrics.confusion_matrix(true_labels, preds))
print(metrics.classification_report(true_labels, preds))

true_labels_ = [[0 for _ in range(0, 3)] for label in true_labels]

for i, (_, true) in enumerate(zip(true_labels_, true_labels)):
  true_labels_[i][true] = 1

true_labels_ = np.array(true_labels_)

plot_auc_roc(true_labels_, predictions.predictions)


"""
Load the best model. Load the review sentences. Predict the sentiment of each sentence.
"""
tokenizer = AutoTokenizer.from_pretrained(f"{fp}/twitter-roberta-base-sentiment-latest_3_label_hp_2")
model = AutoModelForSequenceClassification.from_pretrained(f"{fp}/twitter-roberta-base-sentiment-latest_3_label_hp_2")

review_sentences_df = pd.read_csv("review_sentences.csv")
review_sentences_df["predicted_sentiment"] = review_sentences_df["review_sentences"].apply(lambda x: predict(x))
review_sentences_df.to_csv("predicted_review_sentences.csv")
