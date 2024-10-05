import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from MusicGenreClassification import plot_history

DATASET_PATH = "processed/music_genres.json"

def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)
    # Convert lists into numpy arrays
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])
    return inputs, targets

def prepare_dataset(test_size, validation_size):
    # load data
    X, y = load_data(DATASET_PATH)
    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    # create test/validation split
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=validation_size)
    # CNN expects a 3d array -> (130, 13, 1)
    X_train = X_train[..., np.newaxis] # 4d array -> (num_samples, 130, 13, 1)
    X_test = X_test[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    return X_train, X_validation, X_test, y_train, y_validation, y_test

def build_model(input_shape):
    # create model
    model = Sequential()
    # 1st conv layer
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D((3, 3), strides=(2, 2), padding="same"))
    model.add(BatchNormalization())
    # 2nd conv layer
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(MaxPooling2D((3, 3), strides=(2, 2), padding="same"))
    model.add(BatchNormalization())
    # 3rd conv layer
    model.add(Conv2D(32, (2, 2), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), padding="same"))
    model.add(BatchNormalization())
    # flatten the output and feed it into dense layer
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    # output layer
    model.add(Dense(10, activation="softmax"))

    return model

def predict(model, X, y):
    X = X [np.newaxis, ...]
    predictions = model.predict(X) # X -> (1, 130, 13, 1), predictions [[0.2, 0.5, ...]]
    # extract index with max value
    predicted_index = np.argmax(predictions, axis=1)
    print("Expected index:{}, Predicted index:{}".format(y, predicted_index))

if __name__ == "__main__":
    # create train, validation and test sets
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_dataset(0.25, 0.2)

    # build the CNN net
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape)

    # compile the network
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.000009)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])
    
    # train the CNN
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=700)
    # plot graphs
    plot_history(history)

    # evaluate the CNN on the test set
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy on test set is: {}".format(test_accuracy))

    # make predictions on a sample
    X = X_test[100]
    y = y_test[100]
    predict(model, X, y)

    model.save('saved_models/GenreClassification_CNN_500.keras')