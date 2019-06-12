import tensorflow as tf
import tensorflow_hub as hub
import os
import pandas as pd
import re
import seaborn as sns
import pickle
import sys
# from nltk import tokenize
import numpy as np
import time
from USE_dependencies import shuffle_arrays
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Get USE ready
module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" 
#@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
embed = hub.Module(module_url)
tf.logging.set_verbosity(tf.logging.ERROR)

# Get the data ready
file = sys.argv[1]
dataset = pd.read_csv(file)

dataset['sentiment'] = [1 if sentiment == 'positive' else 0 for sentiment in dataset['sentiment'].values]
label = dataset['sentiment'].values
pickle.dump(label, open("labels_out.p","wb"))

#text
text = dataset['review'].values
text = np.array(text)[:, np.newaxis]
text_out = []

print('Embedding...')
# embed em
with tf.Session() as session:
	session.run([tf.global_variables_initializer(), tf.tables_initializer()])
	# give estimated time
	start = time.time()
	test_out = []
	test_num = 50
	for i in range(test_num):
		test_embeddings = session.run(embed(text[i])) 
		test_out.append(test_embeddings)
	end = time.time()
	time_sample = end-start
	time_est = time_sample / test_num * len(text) / 60
	print("Estimated time: " + str(time_est) +'min')
	dis = str(input("Continue embedding? [y/n]"))
	if dis == "y":
		j = 0
		for review in text:
			print(j)
			j += 1
			text_embeddings = session.run(embed(review))
			text_out.append(text_embeddings)
		# dump'em
		print('dumping...')
		pickle.dump(text_out, open("text_out.p", "wb"))
