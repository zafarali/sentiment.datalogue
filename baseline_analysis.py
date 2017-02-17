import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB

import models
from data.utils import load_sparse_csr, generate_folder

MODELS = [ ('Logistic Regression', LogisticRegression), ('NaiveBayes-Bernoulli', BernoulliNB), ('NaiveBayes-Gaussian', GaussianNB) ]

results = {}

for (FOLDER, folder_title) in generate_folder('./data', folder_titles=True):
	X_train = load_sparse_csr(FOLDER+'X_train.csr.npz')
	Y_train = np.load(FOLDER+'Y_train.npy')
	X_test = load_sparse_csr(FOLDER+'X_test.csr.npz')
	Y_test = np.load(FOLDER+'Y_test.npy')

	# shuffle:
	X_train, Y_train = shuffle(X_train, Y_train)

	# need to convert back from sparse because gaussianNB doesn't accept this
	X_train = X_train.toarray()
	X_test = X_test.toarray()

	results[folder_title + '(train)'] = {}
	results[folder_title + '(test)'] = {}

	for (model_name, model) in MODELS:
		print('Model: '+model_name)
		print('Features:'+FOLDER)


		#fit the model
		m = model()
		m.fit(X_train, Y_train)

		Y_train_pred = m.predict(X_test)
		Y_test_pred = m.predict(X_train)

		print('Training Accuracy:')
		train_acc = m.score(X_train, Y_train)
		print(train_acc)

		print('Testing Accuracy:')
		test_acc = m.score(X_test, Y_test)
		print(test_acc)

		results[folder_title + '(train)'][model_name] = train_acc
		results[folder_title + '(test)'][model_name] = test_acc

		# print('Training Confusion')
		# print(confusion_matrix(m.predict(X_train), Y_train))

		# print('Testing Confusion')
		# print(confusion_matrix(m.predict(X_test), Y_test))



pd.DataFrame(results).to_pickle('./baseline_results.pickle')