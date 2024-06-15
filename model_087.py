import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.src.utils import np_utils
from keras.callbacks import ModelCheckpoint

# Load the dataset
data = pd.read_csv("data/data.csv")
dataset = np.array(data)
np.random.shuffle(dataset)

# Split features and labels
X = dataset[:, :1024]  # Features (images)
Y = dataset[:, 1024]   # Labels (character classes)

# Normalize and split data into training and testing sets
X_train, X_test = X[:70000] / 255., X[70000:72001] / 255.
Y_train, Y_test = Y[:70000], Y[70000:72001]

# Reshape data for CNN input
image_x, image_y = 32, 32
X_train = X_train.reshape(X_train.shape[0], image_x, image_y, 1)  # (num_samples, image_x, image_y, 1)
X_test = X_test.reshape(X_test.shape[0], image_x, image_y, 1)

# One-hot encode labels
Y_train = np_utils.to_categorical(Y_train, num_classes=37)  # 37 classes for Devanagari characters
Y_test = np_utils.to_categorical(Y_test, num_classes=37)

# Define and compile the CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), input_shape=(image_x, image_y, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(5, 5)))
model.add(Flatten())
model.add(Dense(37, activation='softmax'))  # Output layer with softmax activation for multiclass classification
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define a callback to save the best model based on validation accuracy
filepath = "best_model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Train the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=8, batch_size=64, callbacks=callbacks_list)

# Evaluate the model
scores = model.evaluate(X_test, Y_test, verbose=0)
print("CNN Error: %.2f%%" % (100 - scores[1] * 100))
print("Model Accuracy : %.2f%%" % (scores[1]*100))

# Save the model
model.save('model.h5')
