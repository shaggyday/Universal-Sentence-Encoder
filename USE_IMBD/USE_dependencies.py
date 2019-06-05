import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

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
	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.plot(epochs, acc, 'bo', label='Training acc')
	plt.plot(epochs, val_acc, 'b', label='Validation acc')
	plt.title('model accuracy & loss')
	plt.ylabel('loss                                 accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig(title, bbox_inches='tight')
	return