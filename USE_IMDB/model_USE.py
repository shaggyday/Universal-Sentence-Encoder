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
import keras
import keras.layers as layers
from keras.models import Model
from keras import backend as K
from USE_dependencies import plotem, shuffle_df
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(10)

# Get the data ready
file = sys.argv[1]
df = pd.read_csv(file)
df['sentiment'] = [1 if sentiment == 'positive' else 0 for sentiment in df['sentiment'].values]
df = shuffle(df)
text = df['review'].values
labels = df['sentiment'].values
num_sample = len(text)
split = int(num_sample*0.8)

train_text = np.array(text[:split])[:, np.newaxis]
train_labels = np.array(labels[:split])[:, np.newaxis]
test_text = np.array(text[split:])[:, np.newaxis]
test_labels = np.array(labels[split:])[:, np.newaxis]
# train_text=np.array(train_text)[:, np.newaxis]
# train_labels = np.array(train_labels)[:, np.newaxis]
# test_text=np.array(test_text)[:, np.newaxis]
# test_labels = np.array(test_labels)[:, np.newaxis]

epochs = 10
batch_size = 5000

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
dropout = 0.4
input_text = layers.Input(shape=(1,), dtype=tf.string)
embedding = layers.Lambda(UniversalEmbedding, output_shape=(embed_size,))(input_text)
dense = layers.Dense(16, activation='relu')(embedding)
dropout = layers.Dropout(dropout)(dense)
pred = layers.Dense(1, activation='sigmoid')(dropout)
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
plotem("aug", history)