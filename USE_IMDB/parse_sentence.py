import pandas as pd
import sys
import pickle
from nltk import tokenize
	
file = sys.argv[1]
df = pd.read_csv(file)
text = df['review'].values
sent_text = []

i = 0
for review in text:
	print(i)
	i += 1
	sentences = tokenize.sent_tokenize(review)
	sent_text.append(sentences)

pickle.dump(sent_text, open("sent_text.p", "wb"))
print(sent_text.shape)