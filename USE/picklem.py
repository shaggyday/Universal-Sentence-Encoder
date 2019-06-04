import tensorflow as tf
import tensorflow_hub as hub
import os
import pandas as pd
import re
import seaborn as sns
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Get the data ready
print('Decompressing data...')
dataset = pd.read_csv('movie_reviews_review_level.csv.bz2', compression='bz2')
# dataset['sentiment'] = [1 if sentiment == 'positive' else 0 for sentiment in dataset['sentiment'].values]
# label = dataset['sentiment'].values
# pickle.dump(label, open("label.p","wb"))

text = dataset['review'].values

# Get USE ready
module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" 
#@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
embed = hub.Module(module_url)
tf.logging.set_verbosity(tf.logging.ERROR)

print('Embedding...')
# embed em
with tf.Session() as session:
  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
  text_embeddings = session.run(embed(text))

print('dumping...')
# dump em
pickle.dump(text_embeddings, open("sentence_embeddings.p", "wb"))