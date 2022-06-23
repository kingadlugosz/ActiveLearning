import pandas as pd
import numpy as np


def load_dataset(path, num=False):
    df = pd.read_csv(path, header=None).to_numpy()
    x = df[:, :-1]
    y_names = df[:, -1]
    if num:
        y = y_names
    else:
        label_names = np.unique(y_names)
        labels = np.arange(label_names.size)
        dict_labels = dict(np.column_stack((label_names, labels)))
        y = []
        for elem in y_names:
            y.append(dict_labels[elem])
    return x, np.array(y)
