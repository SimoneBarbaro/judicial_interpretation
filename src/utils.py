import numpy as np
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
MAX_SEQ_LENGTH = BERT_SEQ_LENGTH * 73


def get_class_weights(labels):
    return dict(enumerate(class_weight.compute_class_weight('balanced',
                                                            np.unique(labels),
                                                            labels)))
