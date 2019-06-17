# includes embedding lamba layer in the model
# requires data to NOT be pre embedded
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import sys
import pickle
import keras
import keras.layers as layers
from keras.models import Model
from keras import backend as K
from USE_dependencies import plotem, shuffle_arrays
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(10)

# Tune the hyperparamters here!!!
epochs = 10
batch_size = 100
neurons = 16
dropout = 0.1
plot_title = "out.png"

# Get the data ready
text = pickle.load(open("IMDB-sentences.p","rb"))
labels = pickle.load(open("IMDB-labels.p","rb"))
# text, labels = shuffle_arrays(text, labels)
num_sample = len(text)
split = int(num_sample*0.8)

train_text = np.array(text[:split])[:, np.newaxis]
train_labels = np.array(labels[:split])[:, np.newaxis]
test_text = np.array(text[split:])[:, np.newaxis]
test_labels = np.array(labels[split:])[:, np.newaxis]

# Get USE ready
module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" 
#@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
embed = hub.Module(module_url)
embed_size = embed.get_output_info_dict()['default'].get_shape()[1].value
tf.logging.set_verbosity(tf.logging.ERROR)

# wrapping USE around lambda layer
def UniversalEmbedding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)), 
      signature="default", as_dict=True)["default"]

# building the model with lambda layer
input_text = layers.Input(shape=(1,), dtype=tf.string)
embedding = layers.Lambda(UniversalEmbedding)(input_text)
Dense = layers.Dense(neurons)(embedding)
# dropout = layers.Dropout(dropout)(dense)
pred = layers.Dense(1, activation='sigmoid')(Dense)
model = Model(inputs=[input_text], outputs=pred)
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train'em
with tf.Session() as session:
  K.set_session(session)
  session.run(tf.global_variables_initializer())
  session.run(tf.tables_initializer())
  history = model.fit(train_text, train_labels,
            validation_data=(test_text, test_labels),
            epochs=epochs,
            batch_size=batch_size)

# score = model.evaluate(test_text, test_labels, verbose=1)
# print("\nTest score:", score[0])
# print('Test accuracy:', score[1])

# plot'em
plotem(plot_title, history)