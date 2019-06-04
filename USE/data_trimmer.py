import numpy as np
import pandas as pd
import csv
from text_wrangling import pre_process_corpus

dataset = pd.read_csv('movie_reviews_raw.csv.bz2', compression='bz2')
reviews = dataset['review'].values

print('pre processing data...')
reviews = pre_process_corpus(reviews)
wtr = csv.writer(open ('out.csv', 'w'), delimiter=',', lineterminator='\n')

print('writing into csv...')
for x in reviews : 
   wtr.writerow ([x])