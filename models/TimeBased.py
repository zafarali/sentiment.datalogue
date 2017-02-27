import numpy as np
from keras import layers
from keras.models import Model, Sequential


def LSTM_embedding(input_shape, dropW=0.2, dropU=0.2, n_dims=128, embedding=None):
	assert embedding is not None, 'must supply embedding!'
	model = Sequential()
	model.add( layers.InputLayer(input_shape = (input_shape,)) )
	model.add( embedding )
	model.add( layers.LSTM(n_dims, dropout_W=dropW, dropout_U=dropU) )
	return model