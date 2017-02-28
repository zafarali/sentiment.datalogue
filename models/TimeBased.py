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

def biLSTM_embedding(input_shape, dropW=0.2, dropU=0.2, n_dims=128, embedding=None):
	model = Sequential()
	model.add( layers.InputLayer(input_shape = (input_shape,)) )
	model.add( embedding )
	model.add( layers.Bidirectional( \
		layers.LSTM(n_dims, dropout_W=dropW, dropout_U=dropU)) )
	return model	

def LSTMCNN_embedding(input_shape, n_filters=64, filter_size=5, pool_length=4, \
	dropout=False, dropW=0.2, dropU=0.2, lstm_dims=128, embedding=None, \
	hiddens=[], notactivate=False, bidirectional=False):
	model = Sequential()
	# model.add( layers.InputLayer(input_shape = (input_shape,)) )
	model.add( embedding )

	if dropout:
		model.add( layers.Dropout(dropout) )

	model.add( layers.Conv1D(n_filters, filter_size, border_mode='valid', subsample_length=1) )
	
	if not notactivate:
		model.add( layers.Activation('relu') )
	
	model.add( layers.AveragePooling1D(pool_length) )

	if bidirectional:
		model.add( layers.Bidirectional( \
			layers.LSTM(lstm_dims, dropout_W=dropW, dropout_U=dropU) ) )
	else:
		model.add( layers.LSTM(lstm_dims, dropout_W=dropW, dropout_U=dropU) )

	for hidden in hiddens:
		model.add( layers.Dense(hidden) )
		if dropout:
			model.add( layers.Dropout(dropout) )
		model.add( layers.Activation('relu') )
	return model


def TimeAvgLSTM_embedding(input_shape, dropW=0.2, dropU=0.2, lstm_dims=128, embedding=None):
	model = Sequential()
	model.add( layers.InputLayer(input_shape = (input_shape,)) )
	model.add( embedding )
	model.add( layers.LSTM(lstm_dims, dropout_W=dropW, dropout_U=dropU) )
	model.add( layers.GlobalAveragePooling1D() )

	return model
