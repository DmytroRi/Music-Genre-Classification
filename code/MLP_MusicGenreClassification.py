import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf  # Import only from tensorflow
from tensorflow.keras.optimizers import Adam  # type: ignore # Use tensorflow.keras instead of keras
from tensorflow.keras import layers  # type: ignore # Also import layers from tensorflow.keras
import matplotlib.pyplot as plt

DATASET_PATH = "processed/music_genres.json"

def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)
    # Convert lists into numpy arrays
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])
    return inputs, targets

def plot_history(history):
    fig, axs = plt.subplots(2)
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="lower right")
    axs[1].set_title("Accuracy eval")

    plt.show()


if __name__ == "__main__":
    # Load data
    inputs, targets = load_data(DATASET_PATH)

    # Split the data into train and test sets
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs,
                                                                              targets,
                                                                              test_size=0.3)
    
    # Build the network architecture
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),

        # 1st hidden layer
        tf.keras.layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.3),
        # 2nd hidden layer
        tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.3),
        # 3rd hidden layer
        tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.3),
        # Output layer
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    
    # Compile network
    optimiser = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    # Display model architecture
    model.summary()
    
    # Train network 
    history = model.fit(inputs_train, targets_train, 
              validation_data=(inputs_test, targets_test),
              epochs=150,
              batch_size=32)

    plot_history(history)

    # evaluate the CNN on the test set
    test_error, test_accuracy = model.evaluate(inputs_test, targets_test, verbose=1)
    print("Accuracy on test set is: {}".format(test_accuracy))

    # make predictions on a sample
    X = inputs_test[100]
    y = targets_test[100]
    prediction = model.predict(X[np.newaxis,  ...])
    predicted_index = np.argmax(prediction, axis=1)
    print("Expected index:{}, Predicted index:{}".format(y, predicted_index))

    model.save('saved_models/GenreClassification_MLP.keras')