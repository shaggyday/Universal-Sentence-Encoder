# Runs on IMBD reviews
# v1.x: treats each reviews as an individual sentence
# v1.2: decreases dense layer neurons: 256/64 --> 256

import sys
# import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import pickle
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout
from keras import backend as K
from USE_dependencies import encode_sentence, plotem
np.random.seed(10)

# Get the data ready
text = pickle.load(open("review_level_embeddings.p","rb"))
labels = pickle.load(open("labels.p","rb"))

# building the model 
dropout = 0.4
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(512,)))
model.add(Dropout(dropout))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train'em
train_text = text[:40000]
train_labels = labels[:40000]
test_text = text[40000:]
test_labels = labels[40000:]
epochs = 20
batch_size = 64
history = model.fit(train_text, train_labels,
            validation_data=(test_text, test_labels),
            epochs=epochs,
            batch_size=batch_size)

# plot'em
plotem('USE+IMBD+keras_v2.0.plot.png', history)
