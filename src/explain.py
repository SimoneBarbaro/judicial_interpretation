import pandas as pd
from lime.lime_text import LimeTextExplainer
import gensim

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
        for i, d in enumerate(data["opinion"]):
            ex = self.explainer.explain_instance(d, model_predict_fn)
            self.exps.append(ex)
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
        importances = self.lime_dataset.set_index(["word", "doc"]).mean(level=0).sort_values("importance", ascending=False)
        return importances
