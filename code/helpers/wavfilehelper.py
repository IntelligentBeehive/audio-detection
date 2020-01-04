# -*- coding: utf-8 -*-

import soundfile

from wavinfo import WavInfoReader

class WavFileHelper():
    
    def read_file_properties(self, filename):
        
        wav_file = WavInfoReader(filename)
        
        sample_rate = wav_file.fmt.sample_rate
        num_channels = wav_file.fmt.channel_count
        bit_rate = wav_file.fmt.bits_per_sample
        
        return (num_channels, sample_rate, bit_rate)


    def convert_to_16bit(self, filename):
        print('converting: ' + filename)
        data, samplerate = soundfile.read(filename)
        soundfile.write(filename, data, samplerate, subtype='PCM_16')
        print('done converting: ' + filename)
        

    def get_channel_distribution(self, array_of_channels):
        
        # Get distribution of mono and stereo in values between 0 and 1
        distribution = array_of_channels.value_counts(normalize=True)
        
        # Multiple distribution x100 to get percentage and round to 2 decimals
        decimal_places = 2
        percentage_mono = round(distribution[1] * 100, decimal_places)
        percentage_stereo = round(distribution[2] * 100, decimal_places)

        # Return key-value pair of the percentages
        channel_data = {
            'percentage_mono': percentage_mono,
            'percentage_stereo': percentage_stereo
        }
        
        return channel_data
    
    
    def get_samplerate_distribution(self, array_of_samplerates):
        # Get distribution of samples_rates in values between 0 and 1
        distribution = array_of_samplerates.value_counts(normalize=True)
        
        # Loop over each entry and assign it to a key-value pair where 
        # the value is added as percentage 
        samplerate_data = {}
        for index, value in distribution.items():
            decimal_places = 2
            value_in_percentage = round(value * 100, decimal_places)
            samplerate_data[index] = value_in_percentage
        
        # Return key-value pair of the percentages
        return samplerate_data
    

    def bit_depth_distribution(self, array_of_bit_depths):
        # Get distribution of samples_rates in values between 0 and 1
        distribution = array_of_bit_depths.value_counts(normalize=True)
        
        # Loop over each entry and assign it to a key-value pair where 
        # the value is added as percentage 
        bit_depth_data = {}
        for index, value in distribution.items():
            decimal_places = 2
            value_in_percentage = round(value * 100, decimal_places)
            bit_depth_data[index] = value_in_percentage
        
        # Return key-value pair of the percentages
        return bit_depth_data