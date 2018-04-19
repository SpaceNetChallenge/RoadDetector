import numpy as np
from config import Config
import pandas as pd


def get_csv_folds(path, d):
    df = pd.read_csv(path, index_col=0)
    m = df.max()[0] + 1
    train = [[] for i in range(m)]
    test = [[] for i in range(m)]

    folds = {}
    for i in range(m):
        fold_ids = list(df[df['fold'].isin([i])].index)
        folds.update({i: [n for n, l in enumerate(d) if l in fold_ids]})

    for k, v in folds.items():
        for i in range(m):
            if i != k:
                train[i].extend(v)
        test[k] = v

    return list(zip(np.array(train), np.array(test)))

def update_config(config, **kwargs):
    d = config._asdict()
    d.update(**kwargs)
    print(d)
    return Config(**d)