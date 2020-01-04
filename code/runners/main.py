# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import librosa

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 

from keras.utils import to_categorical

from code.pre_processing.pre_processor import Preprocessor
from code.model.sequential_model import SequentialModel

def run():
    preprocessor = Preprocessor()
    metadata = pd.read_csv("./metadata/UrbanSound8K.csv")
    

    features = []
    for index, row in metadata.iterrows():
        file_name = './audio/' + 'fold' + str(row['fold'])+'/' + str(row["slice_file_name"])
        label = row["class"]
        feature = preprocessor.prepare(file_name, label)
        
        if feature is not None:
            features.append(feature)
    
    featuresdf = pd.DataFrame(features, columns=['feature','class_label'])
    print('Finished feature extraction from ', len(featuresdf), ' files')
    
    # Convert features and corresponding classification labels into numpy arrays
    X = np.array(featuresdf.feature.tolist())
    y = np.array(featuresdf.class_label.tolist())
    
    # Encode the classification labels
    le = LabelEncoder()
    yy = to_categorical(le.fit_transform(y))
    
        
    ### Splitting the dataset
    # split the dataset 
    x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)
    
    # Model
    num_labels = yy.shape[1]
    sequential_model = SequentialModel(num_labels)
    
    # Pre train model
    sequential_model.pre_train(x_test, y_test)
    
    # Train the model
    sequential_model.train(x_train, y_train, x_test, y_test)
    
    
    
    
    class_samples = [
        ('Air conditioner', './audio/fold5/100852-0-0-0.wav'),
        ('Car horn', './audio/fold10/100648-1-1-0.wav'),   
        ('Children playing', './audio/fold5/100263-2-0-117.wav'),   
        ('Dog bark', './audio/fold5/100032-3-0-0.wav'),   
        ('Drilling', './audio/fold3/103199-4-1-0.wav'),  
        ('Engine Idling', './audio/fold9/103249-5-0-0.wav'),   
        ('Gunshot', './audio/fold4/135528-6-4-1.wav'),   
        ('Jackhammer', './audio/fold6/132021-7-0-5.wav'),   
        ('Siren', './audio/fold8/133473-8-0-0.wav'),   
        ('Street music', './audio/fold6/132162-9-1-73.wav')   
    ]
    

    class_samples2 = [
        './sampleaudio/dogbark1.wav',
        './sampleaudio/dogbark2.wav',
        './sampleaudio/dogbark3.wav',
        './sampleaudio/engineidling1.wav',
        './sampleaudio/engineidling2.wav',
        './sampleaudio/engineidling3.wav',
        './sampleaudio/gunshot1.wav',
        './sampleaudio/gunshot2.wav',
        './sampleaudio/gunshot3.wav',
        './sampleaudio/siren1.wav',
        './sampleaudio/siren2.wav',
        './sampleaudio/siren3.wav',
    ]
        
    
    def extract_featue(file_name):
        try:
            audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
            mfccsscaled = np.mean(mfccs.T,axis=0)
            
        except Exception:
            print("Error encountered while parsing file: ")
            return None, None

        return np.array([mfccsscaled])
    
    
    def print_prediction(file_name):
        prediction_feature = extract_featue(file_name) 
    
        predicted_vector = sequential_model.model.predict_classes(prediction_feature)
        predicted_class = le.inverse_transform(predicted_vector) 
        print("The predicted class is:", predicted_class[0], '\n') 
    
        predicted_proba_vector = sequential_model.model.predict_proba(prediction_feature) 
        predicted_proba = predicted_proba_vector[0]
        for i in range(len(predicted_proba)): 
            category = le.inverse_transform(np.array([i]))
            print(category[0], "\t\t : ", format(predicted_proba[i], '.32f'))
    
    
    for entry in class_samples:
        sample_name = entry[0]
        sample_audio_file_location = entry[1]
        
        print('Making a prediction for: ' + sample_name)
        print_prediction(sample_audio_file_location)
        print('\n\n\n')
        
        
    for file in class_samples2:
        print('Making a prediction for: ' + file)
        print_prediction(file)
        print('\n\n\n')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    