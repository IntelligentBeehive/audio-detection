# -*- coding: utf-8 -*-

import numpy as np

import librosa
import librosa.display

from code.helpers.wavfilehelper import WavFileHelper


class Preprocessor():
    
    def __init__(self):
        self.wavfile_helper = WavFileHelper()
        
    
    def prepare(self, file_name, label):
        
        # Converts bitrate to 16 bits if it's higher then 16 bits
        data = self.wavfile_helper.read_file_properties(file_name)
        if(data[2] >= 24):
            print(f'Converting {data[2]} bits file to 16 bits file: {file_name}')
            self.wavfile_helper.convert_to_16bit(file_name)
            
        # Normalizes sample rate
        try:
            audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            mfccs_scaled = np.mean(mfccs.T, axis=0)
        except Exception:
            print("File couldn't be processed: " + file_name)
            return None
        
        
        # return extracted feautures
        return (mfccs_scaled, label)
       
        
    def extract_feature(self, file_name):
        try:
            audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
            mfccsscaled = np.mean(mfccs.T,axis=0)
            
        except Exception:
            return None, None
        
        return np.array([mfccsscaled])
    