import pandas as pd

from .data_load import get_train_val_test_splits, get_data
from .simple_model import SimpleModel

data = get_data()
data_train, data_val_model, data_val_interpretation, data_test = get_train_val_test_splits(data)
data_val = pd.concat([data_val_model, data_val_interpretation])
model = SimpleModel()
res = model.fit(data_train)
pd.DataFrame(res).to_csv("simple_model_fit.csv")
