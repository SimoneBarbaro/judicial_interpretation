import numpy as np
from tensorflow.keras import Model
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer, glue_convert_examples_to_features
from . import utils


class TimeDitributedBert(tf.keras.layers.Wrapper):
    def __init__(self, bert_layer, **kwargs):
        super(TimeDitributedBert, self).__init__(bert_layer, **kwargs)

    def call(self, inputs, training=None, mask=None):
        a = tf.keras.layers.Lambda(lambda x: x[:, 0])(inputs)
        b = tf.keras.layers.Lambda(lambda x: x[:, 1])(inputs)
        c = tf.keras.layers.Lambda(lambda x: x[:, 2])(inputs)
        pooled_output, sequence_output = self.layer({"input_ids": a,
                                                     "attention_mask": b,
                                                     "token_type_ids": c},
                                                    training=training)
        return pooled_output

    def compute_output_shape(self, input_shape):
        return None, 768


def data_to_tf_dataset(data):
    def gen():
        for i, row in data.iterrows():
            yield {"idx": i, "label": row["outcome"], "sentence": row["opinion"]}

    return tf.data.Dataset.from_generator(gen, ({"idx": tf.int64, "label": tf.int32, "sentence": tf.string}))


class HuggingFaceBert(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bert_layer = TFBertForSequenceClassification.from_pretrained('bert-base-cased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    def call(self, inputs, training=None, mask=None):
        return self.bert_layer.call(inputs, training=None)

    def compile(self,
                optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=None,
                loss_weights=None,
                sample_weight_mode=None,
                weighted_metrics=None,
                **kwargs):
        super().compile(optimizer=optimizer, loss=loss,
                        metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

    def get_dataset(self, data, max_length=utils.BERT_SEQ_LENGTH):
        return glue_convert_examples_to_features(data_to_tf_dataset(data), self.tokenizer, max_length=max_length,
                                                 task='cola')

    def train(self, training_data, epochs, batch_size, val_data=None):
        steps_per_epoch = len(training_data["opinion"]) // batch_size
        dataset = self.get_dataset(training_data)
        val_dataset = None
        validation_steps = None
        if val_data is not None:
            val_dataset = self.get_dataset(val_data)
            validation_steps = len(val_data["opinion"]) // batch_size
        return self.fit(dataset.shuffle(utils.RANDOM_SEED).batch(batch_size).repeat(-1),
                        epochs=epochs, steps_per_epoch=steps_per_epoch,
                        class_weight=utils.get_class_weights(training_data["opinion"]),
                        validation_data=val_dataset, validation_steps=validation_steps)

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


class HuggingFaceRecurrentBert(HuggingFaceBert):
    def __init__(self, num_layers=2, num_dense_layers=0,
                 bidirectional=True, type_layer="LSTM", cell_size=64,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

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

        word_ids_reshaped = tf.keras.layers.Reshape((73, 512, 1))(inputs["input_ids"])
        input_mask_reshaped = tf.keras.layers.Reshape((73, 512, 1))(inputs["attention_mask"])
        segment_ids_reshaped = tf.keras.layers.Reshape((73, 512, 1))(inputs["token_type_ids"])
        inputs = tf.keras.layers.Concatenate(axis=-1)([word_ids_reshaped, input_mask_reshaped, segment_ids_reshaped])
        x = self.bert_stuff(inputs)
        for layer in self.recurrent_layers:
            x = layer(x, training=training, mask=mask)
        for layer in self.dense_layers:
            x = layer(x, training=training, mask=mask)
        return self.output_layer(x, training=training, mask=mask)
