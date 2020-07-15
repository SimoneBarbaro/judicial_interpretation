from string import punctuation
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk import ngrams
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from xgboost import XGBClassifier

from . import utils

nlp = spacy.load('en_core_web_sm')
bad_words = ["affirmed", "affirm", "reversed", "reverse"]
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


def get_ngrams(corpus, max_len, tokenizer_fn=tokenize_text):
    grams = []
    for i, d in enumerate(corpus):
        tokens = tokenizer_fn(d)
        grams.append(tokens)
        for n in range(2, max_len):
            grams[i] += list(ngrams(tokens, n))


class SimpleModel:
    def __init__(self):
        tfidf = TfidfVectorizer(min_df=0.01,
                                max_df=0.9,
                                max_features=10000,
                                # stop_words='english',
                                use_idf=True,
                                ngram_range=(1, 3),
                                tokenizer=tokenize_text)

        self.model = Pipeline([
            ("Vectorizer", tfidf),
            ("model", XGBClassifier())
        ])
        self.search_parameters = {
            "Vectorizer__min_df": [0, 0.01, 0.05, 0.1],
            "Vectorizer__max_df": [0.9, 0.95, 0.99, 1],
            "Vectorizer__max_features": [1000, 2000, 5000, 10000],
            "Vectorizer__ngram_range": [(1, 1), (1, 2), (1, 3), (1, 4)]
        }
        self.search = GridSearchCV(self.model, self.search_parameters, n_jobs=-1)

    def fit(self, dataset):
        self.search.fit(dataset["opinion"], dataset["outcome"])
        return self.search.cv_results_

    def evaluate(self, dataset):
        self.search.score(dataset["opinion"], dataset["outcome"])

    def predict(self, documents):
        return self.search.predict_proba(documents)
