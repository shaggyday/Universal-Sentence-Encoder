# requires data to be PRE-EMBEDDED

import sys
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import pickle
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
from keras import backend as K
from keras.optimizers import Adam
from USE_dependencies import shuffle_arrays, plotem
np.random.seed(10)

# Get the data ready
text = pickle.load(open("IMDB_reviewLevel_reviews_em.p","rb"))
labels = pickle.load(open("IMDB_reviewLevel_labels.p","rb"))
text, labels = shuffle_arrays(text, labels)

# building the model 
dropout = 0.4
model = Sequential()
model.add(LSTM(128, input_shape=(None ,512)))
# model.add(Dense(128, activation='relu', input_shape=(512,)))
# model.add(Dropout(dropout))
# model.add(Dense(16, activation='relu'))
# model.add(Dropout(dropout))
model.add(Dense(1, activation='sigmoid'))
model.summary()
# optimizer = Adam(lr=0.0005)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train'em
train_text = text[:30000]
train_labels = labels[:30000]
val_text = text[30000:35000]
val_labels = labels[30000:35000]
test_text = text[35000:]
test_labels = labels[35000:]
epochs = 50
batch_size = 64
history = model.fit(train_text, train_labels,
            validation_data=(val_text, val_labels),
            epochs=epochs,
            batch_size=batch_size)

# test
score = model.evaluate(test_text, test_labels, verbose=1)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])

# plot'em
plotem('128.png', history)
