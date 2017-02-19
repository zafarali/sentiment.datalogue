import string
import nltk
import numpy as np
import pandas as pd
from collections import Counter
from keras.preprocessing.text import text_to_word_sequence

GLOVE_FOLDER = './glove.6B'

# SKIP_WORDS = nltk.corpus.stopwords.words('english')
CHAR_EMBEDDING = dict(zip(*(string.ascii_lowercase + ' ', range(0, 27))))
CORPUS = pd.read_pickle('./processed/pd.DF.train_both.pickle')['text']

def load_glove_embeddings(folder=GLOVE_FOLDER, dimensions=100):
    """
        Loads Glove embeddings into a lookup table.

        Reference: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    """
    assert dimensions in [ 50, 100, 200, 300 ], 'Dimension was unrecognized.'

    embeddings = {}
    with open(GLOVE_FOLDER+'/glove.6B.'+str(dimensions)+'d.txt') as f:
        for line in f:
            values = line.split()
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings[word] = embedding

    return embeddings


def create_word_embedding(corpus=CORPUS, top_n=None):
	"""
		Creates a word embedding using a corpus
		returns a dictionary that you can look up indices of a onehot encoding
	"""
	word_set = Counter(text_to_word_sequence(' '.join(corpus)))
	top_n = top_n if top_n else len(word_set)
	return dict([(w,i) for i, (w, count) in enumerate(word_set.most_common(top_n))])


def sentence_to_char_one_hot(sentence):
	"""
		Converts a sentence into a one hot embedding using the characters
		@params:
			sentence (str) to be encoded
		@returns:
			embedded sentence (np.array with shape (len(embedding), len(sentence)))
	"""
    # +1 to store unknown character, ideally this should not happen...
	embedded = np.zeros( ( len(CHAR_EMBEDDING) + 1 , len(sentence) ) )
	for i, char in enumerate(sentence):
		idx = CHAR_EMBEDDING.get(char)
		if idx is not None:
			embedded[idx, i] = 1
	return embedded

def sentence_to_word_idx(sentence, vocabulary):
	"""
		Converts a sentence into an encoding representing indicies of an embedding
		@params:
			sentence: the sentence to encode
			vocabulary: the word to idx mapping
	"""
	sentence = text_to_word_sequence(sentence)
	words_encoded = np.zeros(len(sentence))
	for i, word in enumerate(sentence):
		encoded = vocabulary.get(word, len(vocabulary) + 1 )
		words_encoded[i] = encoded
	return words_encoded


def vocabulary_to_glove_matrix(vocabulary, lookup):
	"""
		Returns a matrix of size (len(vocabulary), len(embedding))
		that can be used in an keras.layers.Embedding layer
		@params:
			vocabulary: the word to int mapping
			lookup: the word to embedding matrix
	"""
	embedding_matrix = np.zeros( ( len(vocabulary) + 1 , len(lookup['the']) ) )

	for word, i in vocabulary.items():
		embedding_vector = lookup.get(word)
		if embedding_vector is not None:
			# words not found in embedding index will be all-zeros.
			embedding_matrix[i] = embedding_vector

	return embedding_matrix

# class Sentence(object):
# 	def __init__(self, sentence):
# 		"""
# 			 class
# 		"""
# 		self.sentence = sentence

