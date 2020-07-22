import os
import pandas as pd
import numpy as np

from data_load import get_train_val_test_splits, get_data
from simple_model import SimpleModel
from explain import LimeExplainer, LdaExplainer, Explainer

data = get_data()
data_train, data_val_model, data_val_interpretation, data_test = get_train_val_test_splits(data)
data_val = pd.concat([data_val_model, data_val_interpretation])
model = SimpleModel(data_train, data_val_model)
if not os.path.exists("../data/simple_model_fit.csv"):
    fit = model.fit()
    pd.DataFrame(fit).to_csv("../data/simple_model_fit.csv", index=False, header=True)
else:
    fit = pd.read_csv("../data/simple_model_fit.csv")
    model.load(fit)

lime_explainer = LimeExplainer()
"""
if not os.path.exists("../data/lime_data.csv"):
    lime_dataset = lime_explainer.build_explanations(model.predict, data_val_interpretation)
    lime_dataset.to_csv("../data/lime_data.csv", header=True, index=False)
lime_dataset = pd.read_csv("../data/lime_data.csv")
lime_explainer.load(lime_dataset)
"""
if not os.path.exists("../data/lime_data_test.csv"):
    lime_dataset = lime_explainer.build_explanations(model.predict, data_test)
    lime_dataset.to_csv("../data/lime_data_test.csv", header=True, index=False)
else:
    lime_dataset = pd.read_csv("../data/lime_data.csv")
    lime_explainer.load(lime_dataset)
lda_explainer = LdaExplainer(ngrams=model.get_model_max_ngrams(), dictionary=model.get_model_vocabulary())

if not os.path.exists("../data/lda_search_result.txt"):
    min_topics = 5
    max_topics = 20
    _, coherences = lda_explainer.search_lda(data_train, data_val, min_topics, max_topics)
    with open("../data/lda_search_result.txt", "w") as f:
        f.write(str(np.argmax(coherences) + min_topics))
else:
    lda_explainer.load_config("../data/lda_search_result.txt", data_train)

explainer = Explainer(data_test, model, lime_explainer, lda_explainer)

good_importance, bad_importance = explainer.get_aggregated_topic_importance()

print(good_importance)
print(bad_importance)

for doc, prediction, explanation in explainer.explain_iterator():
    pass
