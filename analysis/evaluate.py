import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc


def _onehot_to_label(arr):
	return np.argmax(arr, axis=1)
def _proba_to_class(arr):
	return (arr > 0.5).astype('int32')


def confusion(true, predictions):
	"""
		@params:
			true: true labels
			predictions: probability of label
		@returns sklearn.metrics.confusion_matrix
	"""
	predictions = _proba_to_class(predictions)
	return confusion_matrix(true, predictions)

def accuracy(true, predictions):
	"""
		Calculates the accuracy of the predictions
		@params:
			true: true labels
			predictions: probability of label
		@returns sklearn.metrics.accuracy_score
	"""
	predictions = _proba_to_class(predictions)
	return accuracy_score(true, predictions)

def roc_auc(true, prediction_probas):
	"""
		Calculates the area under the recieving operating characteristic curve
		the ROC is the plot of true positive vs false positive rate
		the AUC is the area under this curve. 
		@params:
			true: true labels
			prediction_probas: the probability of getting a true label
		@returns 
			(list) false_positive_rate, (list) true_positve_rate, (float) area_under_curve
		reference: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html

	"""
	fpr, tpr, _ = roc_curve(true, prediction_probas) #true positive, false positive rates
	auc_ = auc(fpr, tpr)
	return fpr, tpr, auc_


def evaluate(true, predictions, title='Training'):
	print( title + ' Accuarcy: ' + str(accuracy(true, predictions)) )
	print( title + ' Confusion: ')
	print(confusion(true, predictions))
	print( title + 'AUC ROC: ' + str(roc_auc(true, predictions)[-1]))

	return accuracy(true, predictions)

