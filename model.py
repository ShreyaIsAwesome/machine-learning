import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout


images = np.load("./static/data/images.npy")
labels = np.load("./static/data/labels.npy")
one_hot_encoding_to_label_dict = {np.argmax(ohe):label for ohe, label in zip(np.array(pd.get_dummies(labels)), labels)}

class Model:
    def __init__(self, load_existing_model):
        self.x = images/225.0
        self.y = np.array(pd.get_dummies(labels))

        if load_existing_model:
            print("Loading existing model...")
            self.epochs = 0
            self.created_model = load_model("./static/model/model.keras")
        else:
            print("Creating a new model...")
            self.epochs = 30
            self.created_model = self.create_model()
            load_existing_model=True

        self.created_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42)
        self.train_model(self.epochs)

    def create_model(self):
        model = Sequential()

        model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.15))

        model.add(Dense(8, activation='softmax'))
    
        return model

    def train_model(self, epochs):
        self.history = self.created_model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=32, validation_data=(self.x_test, self.y_test))
        self.created_model.save("./static/model/model.keras")
        print("Model trained and saved successfully.")
    
    def predict(self, image):
        class_num = np.argmax(self.created_model.predict(image))
        class_name = one_hot_encoding_to_label_dict[class_num]
        return class_name
        
# model.train_model(epochs=5)