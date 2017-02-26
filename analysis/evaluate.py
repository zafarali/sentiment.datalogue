import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


def _onehot_to_label(arr):
	return np.argmax(arr, axis=1)

def confusion(true, predictions):
	"""
		@params:
			predictions: one-hot normalized probabilities of each class
			true: the true one-hot encoding
		@returns sklearn.metrics.confusion_matrix
	"""
	# true = _onehot_to_label(true)
	# predictions = _onehot_to_label(predictions)
	return confusion_matrix(true, predictions)

def accuracy(true, predictions):
	# true = _onehot_to_label(true)
	# predictions = _onehot_to_label(predictions)
	return accuracy_score(true, predictions)


def evaluate(true, predictions, title='Training'):
	print( title + ' Accuarcy: ' + str(accuracy(true, predictions)) )
	print( title + ' Confusion: ')
	print(confusion(true, predictions))