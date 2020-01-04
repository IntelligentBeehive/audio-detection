# -*- coding: utf-8 -*-

import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav


def show_audio_samples():
    
    samples = [
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
    
    for sample in samples:
        sample_name = sample[0]
        sample_audio_file_location = sample[1]
        
        plt.figure(figsize=(12,4))
        plt.title(sample_name)
        plt.xlabel('Time in seconds')
        plt.ylabel('Amplitude')
        data,sample_rate = librosa.load(sample_audio_file_location)
        _ = librosa.display.waveplot(data,sr=sample_rate)
        

def show_mono_stereo_difference():
    filename = './audio/fold6/132162-9-1-73.wav'

    librosa_audio, librosa_sample_rate = librosa.load(filename)
    scipy_sample_rate, scipy_audio = wav.read(filename)
    
    plt.figure(figsize=(12, 4))
    plt.title('stereo')
    plt.plot(scipy_audio)
    
    plt.figure(figsize=(12, 4))
    plt.title('mono')
    plt.plot(librosa_audio)
    


def show_mfcc_extration():
    filename = './audio/fold6/132162-9-1-73.wav'

    librosa_audio, librosa_sample_rate = librosa.load(filename)
    mfccs = librosa.feature.mfcc(y=librosa_audio, sr=librosa_sample_rate, n_mfcc=40)
    print(mfccs.shape)
    
    librosa.display.specshow(mfccs, sr=librosa_sample_rate, x_axis='time')
    
    