import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.flow as naf

from nlpaug.util import Action

import pickle
import os
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import re
import seaborn as sns

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" 
embed = hub.Module(module_url)
tf.logging.set_verbosity(tf.logging.ERROR)

dataset = pd.read_csv('movie_reviews_sentence_level.csv.bz2')


text = dataset['review'].values