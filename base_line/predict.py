import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

test = pd.read_csv("test.tsv", sep="\t")


def conver_to_text(row):
    return " ".join(map(str, row))


test_texts = test.apply(conver_to_text, axis=1)


with open("model.pkl", 'rb') as file:
    vectorizer = pickle.load(file)
    cat_clf = pickle.load(file)
    dep_clf = pickle.load(file)

X = vectorizer.transform(test_texts)

submission = pd.DataFrame()

submission["category_id"] = cat_clf.predict(X)
submission["department_id"] = dep_clf.predict(X)


submission.to_csv("prediction.csv", index=False)
