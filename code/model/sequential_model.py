# -*- coding: utf-8 -*-

import numpy as np

from datetime import datetime 

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint 
from sklearn.preprocessing import LabelEncoder

class SequentialModel():
    
    def __init__(self, num_labels):
        
        self.model = Sequential()
        self.model.add(Dense(256, input_shape=(40,)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        
        self.model.add(Dense(256))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        
        self.model.add(Dense(num_labels))
        self.model.add(Activation('softmax'))
        
        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
        
        
    def pre_train(self, x_test, y_test):
        # Display model architecture summary 
        self.model.summary()
        
        # Calculate pre-training accuracy 
        score = self.model.evaluate(x_test, y_test, verbose=0)
        accuracy = 100*score[1]
        
        print("Pre-training accuracy: %.4f%%" % accuracy)


    def train(self, x_train, y_train, x_test, y_test):
        
        num_epochs = 100
        num_batch_size = 32
    
        checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.basic_mlp.hdf5', 
                                       verbose=1, save_best_only=True)
        
        start = datetime.now()
        
        self.model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=1)
        
        
        duration = datetime.now() - start
        print("Training completed in time: ", duration)
        
        
        #Evaluating the model on the training and testing set
        score = self.model.evaluate(x_train, y_train, verbose=0)
        print("Training Accuracy: ", score[1])
        
        score = self.model.evaluate(x_test, y_test, verbose=0)
        print("Testing Accuracy: ", score[1])
        
        
    def predict(self, prediction_feature):
        
        le = LabelEncoder()
        
        predicted_vector = self.model.predict_classes(prediction_feature)
        predicted_class = le.inverse_transform(predicted_vector) 
        print("The predicted class is:", predicted_class[0], '\n') 
    
        predicted_proba_vector = self.model.predict_proba(prediction_feature) 
        predicted_proba = predicted_proba_vector[0]
        for i in range(len(predicted_proba)): 
            category = le.inverse_transform(np.array([i]))
            print(category[0], "\t\t : ", format(predicted_proba[i], '.32f') )