
import pandas as pd
import numpy as np
import csv
import sys

file = sys.argv[1]
df = pd.read_csv(file)
text = df['review'].values
labels = df['sentiment'].values

# generate random indexs
idx = np.random.permutation(len(text))
text = text[idx]
labels = labels[idx]

df_shuffle = pd.DataFrame()
df_shuffle['review'] = text
df_shuffle['sentiment'] = labels
df_shuffle.to_csv("shuffled.csv")
