import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

from . import utils
from .data_load import get_data, get_train_val_test_splits


def extend_data(inputs, y):
    x = {"input_ids": tf.reshape(inputs["input_ids"], shape=(72, 1, utils.BERT_SEQ_LENGTH)),
         "attention_mask": tf.reshape(inputs["attention_mask"], shape=(72, 1, utils.BERT_SEQ_LENGTH)),
         "token_type_ids": tf.reshape(inputs["token_type_ids"], shape=(72, 1, utils.BERT_SEQ_LENGTH))}
    return tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(x), tf.data.Dataset.from_tensor_slices(tf.repeat(y, 72))))


def make_features(data, bert, file_name):
    supersplit_data = data.flat_map(extend_data)
    features = []
    j = 0
    for d in supersplit_data.as_numpy_iterator():
        if j % 72 == 0:
            if j > 0:
                f = np.expand_dims(np.concatenate(f, axis=0), axis=0)
                features.append(f)
            f = []
        f.append(bert.predict(d[0])[1])
        j = j + 1

    f = np.expand_dims(np.concatenate(f, axis=0), axis=0)
    features.append(f)

    features = np.concatenate(features, axis=0)
    np.save(file_name, features)
    return features


def build_features():
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    bert = TFBertModel.from_pretrained('bert-base-uncased')
    data = get_data()
    data_train, data_val_model, data_val_interpretation, data_test = get_train_val_test_splits(data)
    original_train_dataset = glue_convert_examples_to_features(utils.data_to_tf_dataset(data_train), tokenizer,
                                                               max_length=utils.MAX_SEQ_LENGTH, task='cola')
    original_valid_dataset = glue_convert_examples_to_features(utils.data_to_tf_dataset(data_val_model), tokenizer,
                                                               max_length=utils.MAX_SEQ_LENGTH, task='cola')
    return make_features(original_train_dataset, bert, "train_features.npy"), make_features(original_valid_dataset, bert, "valid_features.npy")

