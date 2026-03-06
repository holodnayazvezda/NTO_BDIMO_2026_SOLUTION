import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


def conver_to_text(row):
    return " ".join(map(str, row))


train = pd.read_csv("train.tsv", sep="\t")

train_texts = train.drop(columns=["category_id", "department_id"]).apply(
    conver_to_text, axis=1)

vectorizer = CountVectorizer(max_features=1000)

X = vectorizer.fit_transform(train_texts)

cat_clf = MultinomialNB()
dep_clf = MultinomialNB()

cat_clf.fit(X, train["category_id"])
dep_clf.fit(X, train["department_id"])


with open("model.pkl", 'wb') as file:
    pickle.dump(vectorizer, file)
    pickle.dump(cat_clf, file)
    pickle.dump(dep_clf, file)
