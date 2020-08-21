import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import utils


def get_data(data_path, is_new_data):
    if is_new_data:
        data = pd.read_csv(data_path)
        data = data[data["outcome"] != -1]
    else:
        data = pd.read_csv(data_path, sep="|").drop(columns="Unnamed: 0")
        data["outcome"] = data["caseDisposition"].apply(lambda x: utils.OUTCOME_DICT.get(x, np.nan))
    return data[["opinion", "outcome"]].dropna()


def get_train_val_test_splits(data):
    data_train, data_tmp = train_test_split(data, test_size=1.0 - utils.TRAIN_SPLIT, random_state=utils.RANDOM_STATE)
    data_val, data_test = train_test_split(data_tmp, test_size=utils.TEST_SPLIT, random_state=utils.RANDOM_STATE)
    data_val_model, data_val_interpretation = train_test_split(data_val, test_size=utils.INTERPRETATION_VAL_SPLIT,
                                                               random_state=utils.RANDOM_STATE)
    return data_train, data_val_model, data_val_interpretation, data_test


def split_paragraphs(data):
    paragraphs = []
    for i, row in data.iterrows():
        text = row["opinion"]
        outcome = row["outcome"]
        vec = text.split("\n\n ")
        paragraphs.append(pd.DataFrame({"opinion": vec, "outcome": [outcome] * len(vec)}))
    return pd.concat(paragraphs, ignore_index=True)
