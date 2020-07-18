import os
import pandas as pd

from data_load import get_train_val_test_splits, get_data
from simple_model import SimpleModel, get_ngrams
from explain import LimeExplainer
import gensim
from gensim.models.ldamodel import LdaModel, CoherenceModel
from gensim import corpora

data = get_data()
data_train, data_val_model, data_val_interpretation, data_test = get_train_val_test_splits(data)
data_val = pd.concat([data_val_model, data_val_interpretation])
model = SimpleModel(data_train, data_val_model)
if not os.path.exists("../data/simple_model_fit2.csv"):
    fit = model.fit()
    pd.DataFrame(fit).to_csv("../data/simple_model_fit2.csv", index=False, header=True)
fit = pd.read_csv("../data/simple_model_fit2.csv")
model.load(fit)
print(pd.DataFrame(fit))
print(model.evaluate(data_val_model))

exit(0)

explainer = LimeExplainer()
if not os.path.exists("../data/lime_data.csv"):
    lime_dataset = explainer.build_explanations(model.predict, data_val_interpretation)
    lime_dataset.to_csv("lime_data.csv", header=True, index=False)
lime_dataset = pd.read_csv("../data/lime_data.csv")
explainer.load(lime_dataset)
print(explainer.get_word_importances())

doc_clean = get_ngrams(data_train["opinion"], 4)
doc_clean_val = get_ngrams(data_val["opinion"], 4)
dictionary = corpora.Dictionary(doc_clean)

# creating the document-term matrix
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

lda_search = []
coherences = []
for t in range(10, 101, 10):
    lda = gensim.models.wrappers.LdaMallet("../mallet-2.0.8/bin/mallet",
                                           corpus=doc_term_matrix,
                                           id2word=dictionary,
                                           num_topics=t)
    lda.save('../data/lda{}.model'.format(t))
    lda_search.append(lda)
    coherence_model = CoherenceModel(lda, texts=doc_clean_val, dictionary=dictionary, coherence='c_v')
    coherence = coherence_model.get_coherence()
    coherences.append(coherence)
    print(coherence)
