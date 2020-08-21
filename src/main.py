import os
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from string import punctuation
from lime.lime_text import LimeTextExplainer
from numpy.random import randint
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix

from data_load import get_train_val_test_splits, get_data
from simple_model import SimpleModel
from explain import LimeExplainer, LdaExplainer, Explainer


def main(args):
    data = get_data(args.data_path, args.is_new_data)

    plt.clf()
    data["opinion"].apply(len).hist()
    plt.xlabel("document lenght")
    plt.ylabel("frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(args.result_dir, 'data_len_hist.png'))
    translator = str.maketrans('', '', punctuation)

    def unigrams_distinct(text):
        return len(set(text.replace('\r', ' ').replace('\n', ' ').lower().translate(translator).split()))

    plt.clf()
    data["opinion"].apply(unigrams_distinct).hist()
    plt.xlabel("distinct unigrams")
    plt.ylabel("frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(args.result_dir, 'data_unigrams_distinct.png'))

    data_train, data_val_model, data_val_interpretation, data_test = get_train_val_test_splits(data)
    data_val = pd.concat([data_val_model, data_val_interpretation])
    model = SimpleModel(data_train, data_val_model)
    if not os.path.exists(os.path.join(args.result_dir, "simple_model_fit.csv")):
        fit = model.fit()
        pd.DataFrame(fit).to_csv(os.path.join(args.result_dir, "simple_model_fit.csv"), index=False, header=True)
    else:
        fit = pd.read_csv(os.path.join(args.result_dir, "simple_model_fit.csv"))
        model.load(fit)
    print("Model validation f1 score:")
    print(model.evaluate(data_val_model))
    print("Model test f1 score:")
    print(model.evaluate(data_test))

    print("Model test confusion matrix:")
    print(confusion_matrix(data_test["outcome"].values, np.argmax(model.predict(data_test["opinion"]), axis=1)))

    explainer = LimeTextExplainer(class_names=[0, 1])
    model_predict_fn = model.predict
    e = data_test["opinion"].head(1).apply(explainer.explain_instance, classifier_fn=model_predict_fn).iloc[0]
    plt.clf()
    e.as_pyplot_figure()
    plt.tight_layout()
    plt.xlabel("LIME score")
    plt.ylabel("word")
    plt.tight_layout()
    plt.savefig(os.path.join(args.result_dir, 'lime_example.png'))

    lime_explainer = LimeExplainer()
    print("lime on validation data:")
    if not os.path.exists(os.path.join(args.result_dir, "lime_data_val.csv")):
        lime_dataset = lime_explainer.build_explanations(model.predict, data_val_interpretation)
        lime_dataset.to_csv("../data/lime_data_val.csv", header=True, index=False)
    else:
        lime_dataset = pd.read_csv(os.path.join(args.result_dir, "lime_data_val.csv"))
        lime_explainer.load(lime_dataset)
    print("lime on test data:")
    if not os.path.exists(os.path.join(args.result_dir, "lime_data_test.csv")):
        lime_dataset = lime_explainer.build_explanations(model.predict, data_test)
        lime_dataset.to_csv(os.path.join(args.result_dir, "new_lime_data_test.csv"), header=True, index=False)
    else:
        lime_dataset = pd.read_csv(os.path.join(args.result_dir, "lime_data_test.csv"))
        lime_explainer.load(lime_dataset)

    lda_explainer = LdaExplainer(ngrams=model.get_model_max_ngrams(), dictionary=model.get_model_vocabulary())

    if not os.path.exists(os.path.join(args.result_dir, "lda_search_result.txt")):
        min_topics = 5
        max_topics = 20
        _, coherences = lda_explainer.search_lda(data_train, data_val, min_topics, max_topics)
        with open(os.path.join(args.result_dir, "lda_search_result.txt"), "w") as f:
            f.write(str(np.argmax(coherences) + min_topics))

        plt.clf()
        plt.plot(range(5, 21), coherences)
        plt.xlabel("num topics")
        plt.ylabel("coherences")
        plt.tight_layout()
        plt.savefig(os.path.join(args.result_dir, "lda_coherences.png"))
    else:
        lda_explainer.load_config(os.path.join(args.result_dir, "lda_search_result.txt"), data_train)

    cols = np.linspace(0, 360, lda_explainer.lda.num_topics)
    for i, weights in lda_explainer.lda.show_topics(num_topics=-1,
                                                    num_words=100,
                                                    formatted=False):
        maincol = cols[i]

        def colorfunc(word=None, font_size=None,
                      position=None, orientation=None,
                      font_path=None, random_state=None):
            color = randint(maincol - 10, maincol + 10)
            if color < 0:
                color = 360 + color
            return "hsl(%d, %d%%, %d%%)" % (color, randint(65, 75) + font_size / 7, randint(35, 45) - font_size / 10)

        wordcloud = WordCloud(background_color="white",
                              ranks_only=False,
                              max_font_size=120,
                              color_func=colorfunc,
                              height=600, width=800).generate_from_frequencies(dict(weights))

        plt.clf()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(os.path.join(args.result_dir, "word_clouds{}.png".format(i)))

    plt.clf()
    lime_explainer.get_word_importances().head(20)["score"].plot.bar()
    plt.ylabel("LIME score")
    plt.tight_layout()
    plt.savefig(os.path.join(args.result_dir, "best_scores.png"))

    explainer = Explainer(data_test, model, lime_explainer, lda_explainer)
    affirmed_topic_importance, reversed_topic_importance = explainer.get_aggregated_topic_importance()
    plt.clf()
    pd.DataFrame({"topic_importance": affirmed_topic_importance}).plot.bar()
    plt.xlabel("Topic")
    plt.ylabel("Normalized score")
    plt.tight_layout()
    plt.savefig(os.path.join(args.result_dir, "affirmed_topic_importance.png"))
    plt.clf()
    pd.DataFrame({"topic_importance": reversed_topic_importance}).plot.bar()
    plt.xlabel("Topic")
    plt.ylabel("Normalized score")
    plt.tight_layout()
    plt.savefig(os.path.join(args.result_dir, "reversed_topic_importance.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="../data/sc_opinions_meta_00_11_case_level.csv",
                        help="Where to find the data")
    parser.add_argument("--is_new_data", type=bool, default=False,
                        help="Whether the data is on the old format. It's required for knowing how to read it.")
    parser.add_argument("--result_dir", type=str, default="../results/",
                        help="Where to store the run results.")
    args = parser.parse_args()
    main(args)
