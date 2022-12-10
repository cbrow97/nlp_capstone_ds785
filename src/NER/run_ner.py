import pandas as pd
import spacy
from ner_SERVICE import add_service_ent


nlp = spacy.load("en_core_web_trf")
food_nlp = spacy.load("./output/model-best")

food_nlp.replace_listeners("tok2vec", "ner", ["model.tok2vec"])

nlp.add_pipe('ner', source=food_nlp, name="food_nlp", after="ner")

add_service_ent(nlp)

s_df = pd.read_csv("/home/ubuntu/consumer-feedback-analyzer/src/NER/predicted_review_sentences.csv")

tagged_entites = []
for _, row in s_df.iterrows():
    doc = nlp(row["review_sentences"])
    for word in doc.ents:
        tagged_entites.append(
            (row["review_id"], row["review_sentences"], word.text, word.label_, row["predicted_sentiment"])
        )

result_df = pd.DataFrame(tagged_entites, columns=["review_id", "review_sentences", "entity", "entity_type", "predicted_sentiment"])
result_df["entity"] = result_df["entity"].str.lower()

result_df.to_csv("ner_and_sentiment_output.csv", index=False)
