"""
	Best Logistic Regression Model
	vocabsize = 100000
	ngram = (1,1)
	tfidf
"""
import csv
import json
import os
from sklearn.linear_model import LogisticRegression
from data.utils import load_sparse_csr
from analysis import evaluate
import numpy as np
from sklearn.utils import shuffle

FOLDER = './data/processed/tfidf_10000_1-1grams/'

X_train = load_sparse_csr(FOLDER + 'X_train.csr.npz')
X_test = load_sparse_csr(FOLDER + 'X_test.csr.npz')

Y_train = np.load(FOLDER + 'Y_train.npy')
Y_test = np.load(FOLDER + 'Y_test.npy')


X_train, Y_train = shuffle(X_train, Y_train)


lr = LogisticRegression(penalty='l2')
lr.fit(X_train, Y_train)

Y_train_preds = lr.predict_proba(X_train)
Y_test_preds = lr.predict_proba(X_test)


train_acc = evaluate.evaluate(Y_train, evaluate._onehot_to_label(Y_train_preds), title='Training')
test_acc = evaluate.evaluate(Y_test, evaluate._onehot_to_label(Y_test_preds), title='Testing')

fpr_train, tpr_train, auc_train = evaluate.roc_auc(Y_train, np.argmax(Y_train_preds, axis=1))
fpr_val, tpr_val, auc_val = evaluate.roc_auc(Y_test, np.argmax(Y_test_preds, axis=1))

o = {
	'fpr': fpr_val.tolist(),
	'tpr': tpr_val.tolist(),
	'auc': auc_val.tolist()
}

json.dump(o, open('./LR.auc.json', 'w'))
print('Done')

data_point = ['LR', 'vocabsize=10000, ngram=(1,1)', 'tfidf', train_acc, test_acc, auc_val.tolist(), tpr_val.tolist(), fpr_val.tolist()]
with open('./results.csv', 'a') as f:
	writer = csv.writer(f)
	writer.writerow(data_point)
