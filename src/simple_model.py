from string import punctuation
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk import ngrams
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import f1_score, make_scorer
from xgboost import XGBClassifier
import utils


nlp = spacy.load('en_core_web_sm')
bad_words = ["affirmed", "affirm", "reversed", "reverse", "remanded", "remand"]
nltk.download('stopwords')
translator = str.maketrans('', '', punctuation)
stoplist = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')


def tokenize_text(doc):
    "Input doc and return clean list of tokens"
    doc = doc.replace('\r', ' ').replace('\n', ' ')
    lower = doc.lower()  # all lower case
    nopunc = lower.translate(translator)  # remove punctuation
    words = nopunc.split()  # split into tokens
    nostop = [w for w in words if w not in stoplist and w not in bad_words]  # remove stopwords
    no_numbers = [w if not w.isdigit() else '#' for w in nostop]  # normalize numbers
    stemmed = [stemmer.stem(w) for w in no_numbers]  # stem each word
    return stemmed


def tokenize_single(text):
    text = text.replace('\r', ' ').replace('\n', ' ')
    lower = text.lower()  # all lower case
    nopunc = lower.translate(translator)  # remove punctuation
    return stemmer.stem(nopunc) if not nopunc.isdigit() else '#'


def get_ngrams(corpus, max_len, tokenizer_fn=tokenize_text):
    grams = []
    for i, d in enumerate(corpus):
        tokens = tokenizer_fn(d)
        grams.append([t for t in tokens])
        for n in range(2, max_len + 1):
            grams[i] += [" ".join(gram) for gram in ngrams(tokens, n)]
    return grams


class SimpleModel:
    def __init__(self, train_dataset, val_dataset):
        tfidf = TfidfVectorizer(min_df=0.01,
                                max_df=0.9,
                                max_features=10000,
                                # stop_words='english',
                                use_idf=True,
                                ngram_range=(1, 3),
                                tokenizer=tokenize_text)

        self.model = Pipeline([
            ("Vectorizer", tfidf),
            ("model", XGBClassifier(seed=utils.RANDOM_SEED))
        ])
        self.search_parameters = {
            "Vectorizer__min_df": [0, 0.05, 0.1],
            "Vectorizer__max_df": [0.9, 0.95, 1],
            "Vectorizer__max_features": [1000, 5000, 10000],
            "Vectorizer__ngram_range": [(1, 2), (1, 3), (1, 4)]
        }
        self.train_X = train_dataset["opinion"]
        self.train_y = train_dataset["outcome"]
        self.val_X = val_dataset["opinion"]
        self.val_y = val_dataset["outcome"]
        split = PredefinedSplit(np.concatenate((np.repeat(-1, len(self.train_y)), np.repeat(0, len(self.val_y)))))
        self.search = GridSearchCV(self.model, self.search_parameters, cv=split, scoring="f1",
                                   verbose=2, n_jobs=4
                                   )

    def fit(self):
        X = np.concatenate((self.train_X, self.val_X))
        y = np.concatenate((self.train_y, self.val_y))
        self.search.fit(X, y)
        self.model = self.search.best_estimator_
        return self.search.cv_results_

    def evaluate(self, dataset):
        return self.model.score(dataset["opinion"], dataset["outcome"])

    def load(self, result_dataset):
        params = result_dataset.sort_values("rank_test_score")
        params["param_Vectorizer__ngram_range"] = params["param_Vectorizer__ngram_range"]\
            .apply(lambda s: s.replace("(", "").replace(")", "").split(", "))\
            .apply(lambda q: (int(q[0]), int(q[1])))
        tfidf = TfidfVectorizer(min_df=params["param_Vectorizer__min_df"].values[0],
                                max_df=params["param_Vectorizer__max_df"].values[0],
                                max_features=params["param_Vectorizer__max_features"].values[0],
                                # stop_words='english',
                                use_idf=True,
                                ngram_range=params["param_Vectorizer__ngram_range"].values[0],
                                tokenizer=tokenize_text)
        self.model = Pipeline([
            ("Vectorizer", tfidf),
            ("model", XGBClassifier())
        ])
        self.model.fit(self.train_X, self.train_y)

    def predict(self, documents):
        if type(documents) == str:
            return self.model.predict_proba([documents])
        return self.model.predict_proba(documents)

    def get_model_vocabulary(self):
        return self.model.named_steps["Vectorizer"].vocabulary_

    def get_model_max_ngrams(self):
        return self.model.get_params()["Vectorizer__ngram_range"][1]