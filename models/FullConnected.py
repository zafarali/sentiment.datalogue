import numpy as np
from keras import layers
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.objectives import binary_crossentropy


def FC(input_shape, input_noise=False, hiddens=[500, 200], dropout=True, activations=['relu', 'relu']):
	"""
		A basic fully connected model without an embedding layer
	"""
	if len(hiddens) != len(activations):
		raise ValueError('Number of Hidden Layers must match the activations layer')

	model = Sequential()
	model.add( layers.InputLayer(input_shape = (input_shape,)) )

	if input_noise:
		model.add( layers.noise.GaussianNoise(input_noise) )

	return basic_FNN(model, hiddens=hiddens, dropout=dropout, activations=activations)

def FC_embedding(input_shape, embedding=False, hiddens=[500, 200], dropout=True, activations=['relu', 'relu']):
	"""
		For prototyping purposes a fully connected neural network with an embedding layer
	"""
	if len(hiddens) != len(activations):
		raise ValueError('Number of Hidden Layers must match the activations layer')
	if not embedding:
		raise ValueError('Can only use FC_embedding by supplying an keras.layers.Embedding() layer')

	model = Sequential()
	model.add( layers.InputLayer(input_shape = (input_shape,)) )
	model.add( embedding )
	model.add( layers.Flatten() )

	return basic_FNN(model, hiddens=hiddens, dropout=dropout, activations=activations)

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
