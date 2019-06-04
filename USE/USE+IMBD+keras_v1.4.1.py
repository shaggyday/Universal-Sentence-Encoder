# Runs on IMBD reviews
# v1.x: treats each reviews as an individual sentence
# v1.2: decreases dense layer neurons: 256/64 --> 256

import matplotlib.pyplot as plt
import numpy as np
import pickle
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout
from keras import backend as K
np.random.seed(10)

# Get the data ready
text = pickle.load(open("review_level_embeddings.p","rb"))
label = pickle.load(open("labels.p","rb"))

train_text = text[:40000]
train_label = label[:40000]
test_text = text[40000:]
test_label = label[40000:]

# building the model 
dropout = 0.4
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(512,)))
model.add(Dropout(dropout))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train'em
epochs = 20
batch_size = 32
history = model.fit(train_text, train_label,
            validation_data=(test_text, test_label),
            epochs=epochs,
            batch_size=batch_size)

# plot'em
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('model accuracy & loss')
plt.ylabel('loss                               accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('USE+IMBD+keras_v1.4.plot.png', bbox_inches='tight')