# MNIST CLASSIFIER FOR KAGGLE

# Soil classification with pretrained algorithm.
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import metrics
from keras import losses
from keras import backend as K
from keras.datasets import mnist
import pandas as pd
import os
os.chdir(r'C:/Users/Tim/pythonscripts')

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout
import numpy as np
# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.5))

# Adding a second convolutional layer
classifier.add(Convolution2D(64, (3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.5))

classifier.add(Convolution2D(128, (3, 3), activation="relu"))
classifier.add(Convolution2D(128, (1, 1), activation="relu"))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.5))

classifier.add(Convolution2D(128, (1, 1), activation="relu"))
classifier.add(Convolution2D(128, (1, 1), activation="relu"))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.5))
# Step 3 - Flattening
#classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units=512, activation="relu"))
classifier.add(Dropout(0.5))
classifier.add(Dense(units=1028, activation="relu"))
classifier.add(Dropout(0.5))
classifier.add(Dense(units=2056, activation="relu"))
classifier.add(Dropout(0.5))
classifier.add(Dense(units=10, activation="softmax"))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# Import the MNIST data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
X_train = train.iloc[:, 1:]

X_test = test.iloc[:, :]
y_train = train.iloc[:, :1]
y_test = test.iloc[:, :1]

X_train = np.array(X_train)
X_test = np.array(X_test)

y_train = np.array(y_train)
y_test = np.array(y_test)

# Feature scaling on X
from sklearn.preprocessing import StandardScaler
sc_X_train = StandardScaler()
X_train = sc_X_train.fit_transform(X_train)

# Reshape X
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_train = y_train.reshape(-1, 1)

onehot = OneHotEncoder(sparse=False)
y_train = onehot.fit_transform(y_train)

# Data generator
from keras.preprocessing.image import ImageDataGenerator

# Create the generators for datasets.
train_datagen = ImageDataGenerator(
                                   shear_range = 0.1,
                                   zoom_range = 0.1,
                                   horizontal_flip = True,
                                   rotation_range = 0,
                                   width_shift_range = 0.1,
                                   height_shift_range = 0.1)

test_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.1,
                                   zoom_range = 0.1,
                                   horizontal_flip = True,
                                   rotation_range = 0,
                                   width_shift_range = 0.1,
                                   height_shift_range = 0.1)

training_set = train_datagen.flow(X_train, y_train, batch_size=128)

classifier.fit(X_train, y_train, validation_split=0.2)

test = np.array(test)
test = test.reshape(test.shape[0], 28, 28, 1)
results = classifier.predict(test)
results = np.argmax(results, axis=1)
results = pd.Series(results, name='Label')

submission = pd.concat([pd.Series(range(1, 28001), name='ImageId'), results], axis=1)
submission.to_csv('cnn_mnist_scores2.csv', index=False)

#classifier.fit_generator(training_set, epochs=10, steps_per_epoch=25)