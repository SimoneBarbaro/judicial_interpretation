import numpy as np
from tensorflow.keras import Model
import tensorflow as tf
import tensorflow_hub as hub
from bert.tokenization.bert_tokenization import FullTokenizer
from . import utils


class TimeDitributedBert(tf.keras.layers.Wrapper):
    def __init__(self, bert_layer, **kwargs):
        super(TimeDitributedBert, self).__init__(bert_layer, **kwargs)

    def call(self, inputs, training=None, mask=None):
        a = tf.keras.layers.Lambda(lambda x: x[:, 0])(inputs)
        b = tf.keras.layers.Lambda(lambda x: x[:, 1])(inputs)
        c = tf.keras.layers.Lambda(lambda x: x[:, 2])(inputs)
        pooled_output, sequence_output = self.layer([a, b, c], training=training)
        return pooled_output

    def compute_output_shape(self, input_shape):
        return None, 768


def get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens) > max_seq_length:
        # Cutting down the excess length
        tokens = tokens[0:max_seq_length]
        return [1] * len(tokens)
    else:
        return [1] * len(tokens) + [0] * (max_seq_length - len(tokens))


def get_segments(tokens, max_seq_length):
    if len(tokens) > max_seq_length:
        # Cutting down the excess length
        tokens = tokens[:max_seq_length]
        segments = []
        current_segment_id = 0
        for token in tokens:
            segments.append(current_segment_id)
            if token == "[SEP]":
                current_segment_id = 1
        return segments
    else:
        segments = []
        current_segment_id = 0
        for token in tokens:
            segments.append(current_segment_id)
            if token == "[SEP]":
                current_segment_id = 1
        return segments + [0] * (max_seq_length - len(tokens))


def get_ids(tokens, tokenizer, max_seq_length):
    if len(tokens) > max_seq_length:
        tokens = tokens[:max_seq_length]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        return token_ids
    else:
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = token_ids + [0] * (max_seq_length - len(token_ids))
        return input_ids


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class TfHubBert(Model):
    def __init__(self, max_seq_length, trainable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                                         trainable=trainable)
        vocab_file = self.bert_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = self.bert_layer.resolved_object.do_lower_case.numpy()
        self.tokenizer = FullTokenizer(vocab_file, do_lower_case)
        self.max_seq_length = max_seq_length
        """
        self.input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                                    name="input_word_ids")
        self.input_mask_ = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                                 name="input_mask")
        self.segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                                 name="segment_ids")
        """

    def prep(self, s, get='id'):
        stokens = self.tokenizer.tokenize(s)
        stokens = ["[CLS]"] + stokens + ["[SEP]"]
        if get == 'id':
            input_ids = get_ids(stokens, self.tokenizer, self.max_seq_length)
            return input_ids
        elif get == 'mask':
            input_masks = get_masks(stokens, self.max_seq_length)
            return input_masks
        else:
            input_segments = get_segments(stokens, self.max_seq_length)
            return input_segments

    def call(self, inputs, training=None, mask=None):
        pooled_output, _ = self.bert_layer([inputs[0], inputs[1], inputs[2]],
                                           training=training)
        return pooled_output

    def text_to_bert_input(self, text):
        stokens1 = self.tokenizer.tokenize(text)

        tokens = list(chunks(stokens1, utils.BERT_SEQ_LENGTH))

        input_ids1 = []
        input_masks1 = []
        input_segments1 = []
        for stok in tokens:
            stokens1 = ["[CLS]"] + stok + ["[SEP]"]
            input_ids1.append(get_ids(stokens1, self.tokenizer, self.max_seq_length))
            input_masks1.append(get_masks(stokens1, self.max_seq_length))
            input_segments1.append(get_segments(stokens1, self.max_seq_length))
        return input_ids1, input_masks1, input_segments1

    def dataframe_to_bert_input(self, df):
        input_word_ids = []
        input_mask = []
        segment_ids = []
        ys = []
        for i, row in df.iterrows():
            a, b, c = self.text_to_bert_input(row["opinion"])
            ys = ys + [row["outcome"]] * len(a)
            """
            input_word_ids.append(a)
            input_mask.append(b)
            segment_ids.append(c)
            """
            input_word_ids = input_word_ids + a
            input_mask = input_mask + b
            segment_ids = segment_ids + c

        input_word_ids = np.array(input_word_ids)
        input_mask = np.array(input_mask)
        segment_ids = np.array(segment_ids)
        return [input_word_ids, input_mask, segment_ids], np.array(ys)

    def get_predictor(self):
        def bert_predict(text):
            if isinstance(text, list):
                res = []
                for t in text:
                    a, b, c = self.text_to_bert_input(t)
                    x = [np.array([a]), np.array([b]), np.array([c])]
                    res.append(self.predict(x)[0])
                return np.array(res)
            else:
                a, b, c = self.text_to_bert_input(text)
                x = [np.array([a]), np.array([b]), np.array([c])]
                return self.predict(x)

        return bert_predict

    def compile(self,
                optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=None,
                loss_weights=None,
                sample_weight_mode=None,
                weighted_metrics=None,
                **kwargs):
        super().compile(optimizer=optimizer, loss=loss, metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])


class TfHubSimpleBert(TfHubBert):
    def __init__(self, max_seq_length, trainable, num_layers=2, cell_size=64,
                 *args, **kwargs):
        super().__init__(max_seq_length, trainable, *args, **kwargs)
        self.dense_layers = []
        for i in range(num_layers):
            self.dense_layers.append(tf.keras.layers.Dense(cell_size, activation="relu"))
        self.output_layer = tf.keras.layers.Dense(2, activation="softmax")

    def call(self, inputs, training=None, mask=None):
        x = super().call(inputs, training, mask)
        for layer in self.dense_layers:
            x = layer(x, training=training, mask=mask)
        return self.output_layer(x, training=training, mask=mask)


class TfHubRecurrentBert(TfHubBert):
    def __init__(self, max_seq_length, trainable, num_layers=2, num_dense_layers=0,
                 bidirectional=True, type_layer="LSTM", cell_size=64,
                 *args, **kwargs):
        super().__init__(max_seq_length, trainable, *args, **kwargs)

        self.bert_stuff = tf.keras.layers.TimeDistributed(TimeDitributedBert(self.bert_layer))

        self.recurrent_layers = []
        self.dense_layers = []
        for i in range(num_layers):
            return_sequence = i < num_layers - 1
            layer = tf.keras.layers.GRU(cell_size, return_sequences=return_sequence)
            if type_layer == "LSTM":
                layer = tf.keras.layers.LSTM(cell_size, return_sequences=return_sequence)
            if bidirectional:
                layer = tf.keras.layers.Bidirectional(layer)
            self.recurrent_layers.append(layer)
        for i in range(num_dense_layers):
            self.dense_layers.append(tf.keras.layers.Dense(cell_size, activation="relu"))
        self.output_layer = tf.keras.layers.Dense(2, activation="softmax")

    def call(self, inputs, training=None, mask=None):
        word_ids_reshaped = tf.keras.layers.Reshape((73, 512, 1))(inputs[0])
        input_mask_reshaped = tf.keras.layers.Reshape((73, 512, 1))(self.input_mask_)
        segment_ids_reshaped = tf.keras.layers.Reshape((73, 512, 1))(self.segment_ids_)
        inputs = tf.keras.layers.Concatenate(axis=-1)([word_ids_reshaped, input_mask_reshaped, segment_ids_reshaped])
        x = self.bert_stuff(inputs)
        for layer in self.recurrent_layers:
            x = layer(x, training=training, mask=mask)
        for layer in self.dense_layers:
            x = layer(x, training=training, mask=mask)
        return self.output_layer(2, activation="softmax")(x, training=training, mask=mask)
