import numpy as np
import tensorflow as tf
from transformers import glue_convert_examples_to_features
from sklearn.utils import class_weight

OUTCOME_DICT = {
    1: 1,
    2: 1,
    3: 0,
    4: 0,
    5: 0,
    9: 0,
}
RANDOM_SEED = 11
RANDOM_STATE = np.random.RandomState(RANDOM_SEED)
TRAIN_SPLIT = 0.7
TEST_SPLIT = 0.33
INTERPRETATION_VAL_SPLIT = 0.5
BERT_SEQ_LENGTH = 512
SEQ_EXTENTION = 73
MAX_SEQ_LENGTH = BERT_SEQ_LENGTH * SEQ_EXTENTION


def get_class_weights(labels):
    return dict(enumerate(class_weight.compute_class_weight('balanced',
                                                            np.unique(labels),
                                                            labels)))

def data_to_tf_dataset(data):
    def gen():
        for i, row in data.iterrows():
            yield {"idx": i, "label": row["outcome"], "sentence": row["opinion"]}

    return tf.data.Dataset.from_generator(gen, ({"idx": tf.int64, "label": tf.int32, "sentence": tf.string}))


def get_tokenized_data(data, tokenizer, lenght):
    return glue_convert_examples_to_features(data_to_tf_dataset(data), tokenizer, max_length=72 * lenght, task='cola')