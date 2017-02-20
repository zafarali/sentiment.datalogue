import numpy as np
from keras import layers
from data.utils import load_sparse_csr
from keras.models import Model, Sequential
from sklearn.cross_validation import train_test_split
from keras.objectives import binary_crossentropy
from keras.optimizers import Adam

X_train = load_sparse_csr('./data/processed/tfidf_10000_1-1grams/X_train.csr.npz').toarray()
Y_train = np.load('./data/processed/tfidf_10000_1-1grams/Y_train.npy')
# X_train = X_train.reshape(X_train.shape+(1,))
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1)
BATCH_SIZE = 100


FEATURE_SIZE = X_train.shape[1]
print(FEATURE_SIZE)


model = Sequential()
model.add( layers.InputLayer(input_shape = (FEATURE_SIZE,) ) )
model.add( layers.Dense(512) )
model.add( layers.Activation('relu') )
model.add( layers.Dropout(0.75) )
model.add( layers.Dense(256) )
model.add( layers.Activation('relu') )
model.add( layers.BatchNormalization() )
model.add( layers.Dropout(0.5) )
model.add( layers.Dense(128) )
model.add( layers.Activation('relu') )
model.add( layers.Dense(1) )
model.add( layers.Activation('sigmoid') )


adam = Adam(0.01)
model.compile(optimizer=adam, loss=binary_crossentropy, metrics=['accuracy'])

# callbacks = [ EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto') ]

try:
    model.fit(X_train, Y_train, validation_split=0.1, batch_size=BATCH_SIZE, nb_epoch=1000)
except KeyboardInterrupt as e:
    print('\n Stopped early')


print('train accuracy: '+str(model.evaluate(X_train, Y_train, verbose=False)[1]))
print('val accuracy: '+str(model.evaluate(X_val, Y_val, verbose=False)[1]))
#train accuracy: 0.982266666667
#val accuracy: 0.8688
