# Runs on IMBD reviews
# v1.x: treats each reviews as an individual sentence

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import keras.layers as layers
from keras.models import Model
from keras import backend as K

np.random.seed(10)
# Get the data ready
dataset = pd.read_csv('movie_reviews_review_level.csv.bz2', compression='bz2')
dataset['sentiment'] = [1 if sentiment == 'positive' else 0 for sentiment in dataset['sentiment'].values]
text = dataset['review'].values
label = dataset['sentiment'].values

train_text = text[:40000]
train_label = label[:40000]
test_text = text[40000:]
test_label = label[40000:]
train_text=np.array(train_text)[:, np.newaxis]
train_label = np.array(train_label)[:, np.newaxis]
test_text=np.array(test_text)[:, np.newaxis]
test_label = np.array(test_label)[:, np.newaxis]


# Get USE ready
module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" 
#@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)
tf.logging.set_verbosity(tf.logging.ERROR)
embed_size = embed.get_output_info_dict()['default'].get_shape()[1].value

# wrapping USE around lambda layer
def UniversalEmbedding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)), 
      signature="default", as_dict=True)["default"]

# building the model with lambda layer
dropout = 0.4
input_text = layers.Input(shape=(1,), dtype=tf.string)
embedding = layers.Lambda(UniversalEmbedding, output_shape=(embed_size,))(input_text)
dense1 = layers.Dense(512, activation='relu')(embedding)
dropout1 = layers.Dropout(dropout)(dense1)
dense2 = layers.Dense(128, activation='relu')(dropout1)
dropout2 = layers.Dropout(dropout)(dense2)
pred = layers.Dense(1, activation='sigmoid')(dropout2)
model = Model(inputs=[input_text], outputs=pred)
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 20
batch_size = 64

with tf.Session() as session:
  K.set_session(session)
  session.run(tf.global_variables_initializer())
  session.run(tf.tables_initializer())
  history = model.fit(train_text, train_label,
            validation_data=(test_text, test_label),
            epochs=epochs,
            batch_size=batch_size)

# plot'em
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss.png', bbox_inches='tight')

acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('acc.png', bbox_inches='tight')