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
dataset = pd.read_csv('movie_reviews_processed.csv')
dataset['sentiment'] = [1 if sentiment == 'positive' else 0 for sentiment in dataset['sentiment'].values]
text = dataset['review'].values
label = dataset['sentiment'].values

train_text = text[:40000]
train_label = label[:40000]

test_text = text[40000:]
test_label = label[40000:]

# Get USE ready
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3" 
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
# dropout = 0.2
# recurrent_dropout = dropout
input_text = layers.Input(shape=(1,), dtype=tf.string)
embedding = layers.Lambda(UniversalEmbedding, output_shape=(embed_size,))(input_text)
dense1 = layers.Dense(512, activation='relu')(embedding)
dense2 = layers.Dense(128, activation='relu')(dense1)
pred = layers.Dense(1, activation='softmax')(dense2)
model = Model(inputs=[input_text], outputs=pred)
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

train_text=np.array(train_text)[:, np.newaxis]
train_label = np.array(train_label)
test_text=np.array(test_text)[:, np.newaxis]
test_label = np.array(test_label)

epochs = 15
batch_size = 32

with tf.Session() as session:
  K.set_session(session)
  session.run(tf.global_variables_initializer())
  session.run(tf.tables_initializer())
  history = model.fit(train_text, train_label,
            validation_data=(test_text, test_label),
            epochs=epochs,
            batch_size=batch_size)
  # model.save_weights('./model.h5')


# new_text = ["In what year did the titanic sink ?", "What is the highest peak in California ?", "Who invented the light bulb ?"]
# new_text = np.array(new_text, dtype=object)[:, np.newaxis]
# with tf.Session() as session:
#   K.set_session(session)
#   session.run(tf.global_variables_initializer())
#   session.run(tf.tables_initializer())
#   model.load_weights('./model.h5')  
#   predicts = model.predict(new_text, batch_size=32)
# categories = df_train.label.cat.categories.tolist()
# predict_logits = predicts.argmax(axis=1)
# predict_labels = [categories[logit] for logit in predict_logits]
# print(predict_labels)