import numpy as np
from keras import layers
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.objectives import binary_crossentropy


def FC(input_shape, hiddens=[500, 200], dropout=True, activations=['relu', 'relu']):
	if len(hiddens) != len(activations):
		raise ValueError('Number of Hidden Layers must match the activations layer')

	model = Sequential()
	model.add( layers.InputLayer(input_shape = (input_shape,)) )

	for h, a in zip(hiddens, activations):
		model.add( layers.Dense(h) )
		if dropout:
			model.add( layers.Dropout(dropout) )
		model.add( layers.Activation(a) )

	return model
