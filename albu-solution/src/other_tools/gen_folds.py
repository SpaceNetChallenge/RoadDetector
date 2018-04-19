import pandas as pd
import os
import random
from random import shuffle
random.seed(42)

files = os.listdir(r'/data/train/images')
shuffle(files)
s = {k[:5] for k in files}
d = {k: [v for v in files if v.startswith(k)] for k in s}
folds = {}

idx = 0
for v in d.values():
    for val in v:
        folds[val] = idx % 4
        idx+=1

df = pd.Series(folds, name='fold')
df.to_csv('folds.csv', header=['fold'], index=True)
