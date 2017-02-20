import embeddings
import pandas as pd
import numpy as np
import nltk
from utils import create_folder
from keras.preprocessing.sequence import pad_sequences

DATA_FOLDER = './processed/'
CORPUS = pd.read_pickle( DATA_FOLDER + 'pd.DF.train_both.pickle' )['text']
DIMENSIONS = [ 50 , 100 , 200 , 300 ]
TOP_Ns = [ 500, 1000, 5000, 9000, 10000, None ]

# for testing:
# DIMENSIONS = [ 50 , 100 ]
# TOP_Ns = [ 500 ]

for dim in DIMENSIONS:

	glove_mapping = embeddings.load_glove_embeddings(dimensions = dim)

	for top_n in TOP_Ns:
		word_mapping = embeddings.create_word_embedding(corpus = CORPUS, top_n = top_n)

		# sentence to word idxs mapper
		s_to_w = lambda w: embeddings.sentence_to_word_idx(w, vocabulary = word_mapping)

		# find the largest sentence in our copus and pad to that
		corpus_mapped = CORPUS.map(s_to_w)
		corpus_mapped = pad_sequences(corpus_mapped, padding='post', maxlen=1000)
		max_pad = corpus_mapped.shape[1]

		print('Saving Glove Dimension:' + str(glove_mapping) + ', Mapping Dimension:' +str(top_n) )

		# transform all the data we have
		for dataset in ['train', 'test']:
			# first store indices
			neg_corpus = pd.read_pickle( DATA_FOLDER + 'pd.DF.' + dataset + '_neg.pickle')['text']
			pos_corpus = pd.read_pickle( DATA_FOLDER + 'pd.DF.' + dataset + '_pos.pickle')['text']

			# pad sequences
			neg_mapped = pad_sequences(neg_corpus.map(s_to_w), maxlen=max_pad, padding='post')
			pos_mapped = pad_sequences(pos_corpus.map(s_to_w), maxlen=max_pad, padding='post')

			FOLDER = DATA_FOLDER + 'glove'+str(dim)+'_'+'top'+str(top_n)

			X = np.vstack( [ neg_mapped, pos_mapped ] ) # shuffle later
			Y = np.hstack([ np.zeros(neg_mapped.shape[0]), np.ones(pos_mapped.shape[0]) ])

			assert Y.shape[0] == neg_mapped.shape[0] + pos_mapped.shape[0], 'Y did not have expected size'
			assert X.shape[0] == neg_mapped.shape[0] + pos_mapped.shape[0], 'X did not have expected size'

			create_folder(FOLDER)

			np.save(FOLDER + '/X_' + dataset + '.npy', X)
			np.save(FOLDER + '/Y_' + dataset + '.npy', Y)

			embedding_matrix = embeddings.vocabulary_to_glove_matrix(word_mapping, glove_mapping)
			np.save(FOLDER + '/embedding_matrix.npy' , embedding_matrix)


