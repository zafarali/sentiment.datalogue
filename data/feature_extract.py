"""
	Extracts features from the data so that we can use it directly for classification later
"""
import nltk
import string
import pandas as pd
import numpy as np
from utils import create_folder, save_sparse_csr
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

stopwords = nltk.corpus.stopwords
STOPWORDS = stopwords.words('english') + list(string.punctuation) + ['``', '...', '--', ',', ':', 'br', '\'s', '\'', 'n\'t', '\'\'']
DATA_FOLDER = './processed/'
CORPUS = pd.read_pickle( DATA_FOLDER + 'pd.DF.train_both.pickle' )['text']

for n_features in [ 5000 , 10000 ]:

	# create the hashing vectorizer
	hv = HashingVectorizer(stop_words=STOPWORDS, n_features=n_features)
	hv.fit(CORPUS)

	# create a pipeline for tfidf calculations
	hv2 = HashingVectorizer(stop_words=STOPWORDS, n_features=n_features, norm=None)
	tf = TfidfTransformer()
	tfidf = Pipeline([("hash", hv2), ("tf", tf)])

	tfidf.fit(CORPUS)

	# transform all the data we have
	for dataset in ['train', 'test']:
		neg_corpus = pd.read_pickle( DATA_FOLDER + 'pd.DF.' + dataset + '_neg.pickle')['text']
		pos_corpus = pd.read_pickle( DATA_FOLDER + 'pd.DF.' + dataset + '_pos.pickle')['text']

		# loop over transformers
		for (transformer_name, transformer) in zip( ['hashing', 'tfidf'] , [ hv , tfidf ] ):

			FOLDER = DATA_FOLDER + './' + transformer_name + '_' + str(n_features)
			
			neg = transformer.transform(neg_corpus)
			pos = transformer.transform(pos_corpus)

			X = sparse.vstack( [ neg, pos ] ) # shuffle later
			Y = np.hstack([ np.zeros(neg.shape[0]), np.ones(pos.shape[0]) ])

			assert Y.shape[0] == neg.shape[0] + pos.shape[0], 'Y did not have expected size'
			assert X.shape[0] == neg.shape[0] + pos.shape[0], 'X did not have expected size'

			create_folder(FOLDER)

			save_sparse_csr(  FOLDER + '/X_' + dataset + '.csr', X)
			# np.save(FOLDER + '/X_' + dataset + '.npy', X.toarray())
			np.save(FOLDER + '/Y_' + dataset + '.npy', Y)







