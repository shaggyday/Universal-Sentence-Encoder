import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def encode_sentence(text):
	# Get USE ready
	module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" 
	#@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
	embed = hub.Module(module_url)
	tf.logging.set_verbosity(tf.logging.ERROR)

	print('Embedding...')
	# encode em
	with tf.Session() as session:
	  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
	  text_embeddings = session.run(embed(text))
	
	return text_embeddings

def plotem(title, history):
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	acc = history.history['acc']
	val_acc = history.history['val_acc']
	epochs = range(1, len(loss) + 1)
	plt.plot(epochs, loss, 'r', label='Training loss')
	plt.plot(epochs, val_loss, 'g', label='Validation loss')
	plt.plot(epochs, acc, 'r', label='Training acc')
	plt.plot(epochs, val_acc, 'g', label='Validation acc')
	plt.title('model accuracy & loss')
	plt.ylabel('loss                                 accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig(title, bbox_inches='tight')

def shuffle_arrays(text, labels):
	# generate random indexs
	idx = np.random.permutation(len(text))
	text_shuffled = text[idx]
	labels_shuffled = labels[idx]
	return (text_shuffled, labels_shuffled)

def shuffle_df(df):
	text = df['review'].values
	labels = df['sentiment'].values

	text_shuffled, labels_shuffled = shuffle_arrays(text, labels)

	df_shuffle = pd.DataFrame()
	df_shuffle['review'] = text_shuffled
	df_shuffle['sentiment'] = labels_shuffled
	return df_shuffle

