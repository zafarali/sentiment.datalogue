import numpy as np
from keras import layers
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.objectives import binary_crossentropy

def CNN(input_shape, filters=[64, 64], filter_sizes=[5, 5], dropout=True, batch_norm=True, maxpool_size=5, activation='relu'):
	"""
		Basic convoultion neural network without an embedding layer
	"""
	model = Sequential()
	model.add( layers.InputLayer(input_shape = (input_shape,)) )
	model.add( layers.Reshape((input_shape, 1)) )

	model = basic_CNN(model, filters=filters, filter_sizes=filter_sizes, dropout=dropout, batch_norm=batch_norm, activation=activation)

	model.add( layers.Flatten() )
	model.add( layers.Dense(128) )
	model.add( layers.Activation(activation) )

	return model


def CNN_embedding(input_length, hiddens=[50], embedding=False, filters=[250], filter_sizes=[3], dropout=True, batch_norm=True, activation='relu', maxpool_size=None):
	"""
		Example script from: https://github.com/fchollet/keras/blob/master/examples/imdb_cnn.py
	"""
	if not embedding:
		raise ValueError('Must supply a keras.layers.Embedding layer.')

	model = Sequential()
	model.add( layers.InputLayer( input_shape = (input_length, ) ) )
	model.add( embedding )

	model = basic_CNN(model, noactivate=True, filters=filters, filter_sizes=filter_sizes, dropout=dropout, batch_norm=batch_norm, activation=activation)

	model.add( layers.GlobalMaxPooling1D() )
	model.add( layers.Dropout(dropout) )

	if hiddens and len(hiddens) > 0:
		model = basic_FNN(model, hiddens=hiddens, activations=['relu']*len(hiddens), dropout=dropout)
	

	return model


def basic_CNN(model, noactivate=False, filters=[250], filter_sizes=[3], dropout=True, maxpool=False, batch_norm=True, activation='relu'):
	"""
		Utility function to add multiple stacks of CNNs
	"""
	for n_filters, filter_size in zip(filters, filter_sizes):
		model.add( layers.Conv1D(n_filters, filter_size, border_mode='valid') )
		if not noactivate:
			model.add( layers.Activation(activation) )
		if dropout:
			model.add( layers.Dropout(dropout) )
		if batch_norm:
			model.add( layers.BatchNormalization() )

	return model


def basic_FNN(model, hiddens=[500, 200], dropout=True, activations=['relu', 'relu']):	
	"""
		Helper function to build layers of fully connected networks
	"""
	for h, a in zip(hiddens, activations):
		model.add( layers.Dense(h) )
		if dropout:
			model.add( layers.Dropout(dropout) )
		model.add( layers.Activation(a) )

	return model
