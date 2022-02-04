import numpy
import matplotlib.pyplot as plt

from numpy.random import RandomState

from sklearn.datasets import fetch_olivetti_faces
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

from tensorflow.keras.utils import img_to_array
from keras.models import Model
from keras.preprocessing import image

# Load faces data
rng = RandomState(0)
faces = fetch_olivetti_faces(return_X_y=False, shuffle=True, random_state=rng)

# global centering
faces_centered = faces.images - faces.images.mean(axis=0)

# local centering
#faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)

data = faces_centered

#Test/Train split (40 individual faces in the test set). Simple for loop to extract first instance of every target label from 0 to 39. Notice that it does NOT matter which image of the same person we take so long as it is the only image of that person available in the test set
data_test = []
target_test = []
target_shrunk = faces.target.tolist()
data_shrunk= data.tolist()
i=0

while len(target_test)<40:
	for z in range(0, 400):
		if faces.target[z] == i:
			target_test.append(faces.target[z])
			del target_shrunk[z-i]
			data_test.append(data[z])
			del data_shrunk[z-i]
			i = i+1
			continue

data_shrunk = numpy.array(data_shrunk)
data_test = numpy.array(data_test)

# reshape to be [samples][channels][width][height]
X_train = data_shrunk.reshape(data_shrunk.shape[0], 1, 64, 64).astype('float32')
X_test = data_test.reshape(data_test.shape[0], 1, 64, 64).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(target_shrunk)
y_test = np_utils.to_categorical(target_test)
num_classes = y_test.shape[1]
# define a simple CNN model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Convolution2D(32, (5, 5), padding='same', input_shape=(1, 64, 64), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(400, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
# build the model
model = baseline_model()
# Fit the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=400, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))

#summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
