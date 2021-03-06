import pandas as pd
import numpy as np
from lime.lime_text import LimeTextExplainer
import gensim
from gensim import corpora
from gensim.models.ldamodel import LdaModel, CoherenceModel

from simple_model import get_ngrams, tokenize_single
from sklearn.metrics import accuracy_score
import utils


class LimeExplainer:
    def __init__(self):
        self.explainer = LimeTextExplainer(class_names=[0, 1])
        self.exps = []
        self.lime_dataset = None

    def build_explanations(self, model_predict_fn, data):
        self.exps = []
        id = []
        keys = []
        scores = []
        self.exps = data["opinion"].apply(self.explainer.explain_instance,
                                          classifier_fn=model_predict_fn)
        lime_pred = self.exps.apply(lambda e: np.argmax(e.predict_proba)).values
        print("lime accuracy to model:")
        print(accuracy_score(np.argmax(model_predict_fn(data["opinion"]), axis=-1), lime_pred))
        for i, ex in enumerate(self.exps):
            for word, score in ex.as_list():
                id.append(i)
                keys.append(word)
                scores.append(score)
        self.lime_dataset = pd.DataFrame(data={"doc": id, "word": keys, "score": scores})
        self.lime_dataset["importance"] = self.lime_dataset["score"].apply(lambda x: abs(x))

        return self.lime_dataset

    def load(self, lime_dataset):
        self.lime_dataset = lime_dataset

    def get_word_importances(self):
        importances = self.lime_dataset.set_index(["word", "doc"]).mean(level=0).sort_values("importance",
                                                                                             ascending=False)
        return importances


class LdaExplainer:
    def __init__(self, ngrams=1, dictionary=None):
        self.ngrams = ngrams
        self.dictionary = dictionary
        self.num_words = 1000
        if self.dictionary is not None:
            self.num_words = len(self.dictionary)
        self.lda = None
        self.id2word = None

    def clean(self, data):
        doc_clean = get_ngrams(data, self.ngrams)
        if self.dictionary is None:
            return doc_clean
        cleaned_doc = []
        for doc in doc_clean:
            cleaned = []
            for w in doc:
                if w in self.dictionary.keys():
                    cleaned.append(w)
            cleaned_doc.append(cleaned)

        return cleaned_doc

    def get_corpus(self, data):
        return [self.id2word.doc2bow(doc) for doc in self.clean(data["opinion"])]

    def search_lda(self, data_train, data_val, min_topics=5, max_topics=20):
        doc_train = self.clean(data_train["opinion"])
        doc_val = self.clean(data_val["opinion"])
        self.id2word = corpora.Dictionary(doc_train)
        train_corpus = self.get_corpus(data_train)
        lda_search = []
        coherences = []
        for t in range(min_topics, max_topics + 1):
            lda = gensim.models.wrappers.LdaMallet("../../mallet-2.0.8/bin/mallet",
                                                   corpus=train_corpus,
                                                   id2word=self.id2word,
                                                   num_topics=t,
                                                   random_seed=utils.RANDOM_SEED)
            lda_search.append(lda)
            coherence_model = CoherenceModel(lda, texts=doc_val, dictionary=self.id2word, coherence='c_v')
            coherence = coherence_model.get_coherence()
            coherences.append(coherence)
        self.lda = lda_search[np.argmax(coherences)]

        return lda_search, coherences

    def load_config(self, file, data_train):
        doc_train = self.clean(data_train["opinion"])
        self.id2word = corpora.Dictionary(doc_train)
        with open(file) as f:
            num_topics = int(f.read())
            self.lda = gensim.models.wrappers.LdaMallet("../../mallet-2.0.8/bin/mallet",
                                                        corpus=self.get_corpus(data_train),
                                                        id2word=self.id2word,
                                                        num_topics=num_topics,
                                                        random_seed=utils.RANDOM_SEED)

    def get_lda_dataset(self, data):
        corpus = self.get_corpus(data)

        id = []
        words = []
        topics = []
        topic_props = []
        word_props = []

        for i, row in enumerate(self.lda[corpus]):
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            for (topic_num, prop_topic) in row:
                wp = self.lda.show_topic(topic_num, self.num_words)
                for word, prop in wp:
                    id.append(i)
                    words.append(word)
                    topics.append(topic_num)
                    topic_props.append(prop_topic)
                    word_props.append(prop)
        return pd.DataFrame(
            data={"doc": id, "word": words, "topic": topics, "topic_prop": topic_props, "word_prop": word_props})


class Explainer:
    def __init__(self, data, model, lime_explainer: LimeExplainer, lda_explainer: LdaExplainer):
        self.lime_dataset = lime_explainer.lime_dataset.set_index(["doc", "word"]).sort_index()
        self.lda_dataset = lda_explainer.get_lda_dataset(data).set_index(["doc", "word"]).sort_index()
        self.data = data
        self.model = model

    def explain_document(self, doc):
        topic_scores = np.zeros_like(self.lda_dataset["topic"].drop_duplicates(), dtype=np.float)
        for word, row in self.lime_dataset.loc[doc].iterrows():
            if (doc, tokenize_single(word)) in self.lda_dataset.index:
                best_fit = self.lda_dataset.loc[(doc, tokenize_single(word))].sort_values("word_prop").head(1)
                topic_scores[best_fit["topic"]] += best_fit["topic_prop"] * best_fit["word_prop"] * row["score"]
        return topic_scores / np.sum(topic_scores)

    def explain_iterator(self):
        for doc in self.lda_dataset["topic"].drop_duplicates():
            yield self.data.iloc[doc], np.argmax(
                self.model.predict(self.data.iloc[doc]["opinion"])), self.explain_document(doc)

    def get_aggregated_topic_importance(self):
        correct_topic_importance = []
        wrong_topic_importance = []
        for doc, prediction, explanation in self.explain_iterator():
            if doc["outcome"] == prediction:
                correct_topic_importance.append(explanation)
            else:
                wrong_topic_importance.append(explanation)
        return np.mean(correct_topic_importance, axis=0), np.mean(wrong_topic_importance, axis=0)
