import numpy as np
import mne
import os
import matplotlib.pyplot as plt                    
import argparse
import easygui

from datetime import datetime    
from tkinter import messagebox
from scipy import signal
from pathlib import Path

#Print the system information
mne.sys_info()

def file_name(path):
    name=os.path.splitext(os.path.basename(path))[0]
    if '_2020' in name:
        ind = str.index(name,'_2020')
        name=name[:ind]

    name=name + datetime.now().strftime("_%Y%B%d_%H-%M") 
    return name

def load_brainvision_vhdr(path):
    # Import the BrainVision data into an MNE Raw object
    mne.set_log_level("WARNING")
    print('Reading raw file...')
    print('')
    raw= mne.io.read_raw_brainvision(path, 
            preload=True, 
            eog=('EOG1_1','EOG2_1'),
            misc=('EMG1_1','EMG2_1'),
            verbose=True)
    raw.rename_channels(lambda s: s.strip("."))

    # Specify this as the emg channel (channel type)
    raw.set_channel_types({'EMG1_1': 'emg','EMG2_1': 'emg'}) 
    print('')
    print('Done!')

    return raw    

def show_info(raw):  #Brainvision files
    raw.rename_channels(lambda s: s.strip("."))    # strip channel names of "." characters
    print()
    print('------------------------------ Show Info -----------------------------')
    print('File:', __file__)
    print('')
    _, times = raw[:, :] 
    print('Data type: {}\n\n{}\n'.format(type(raw), raw))
    # Give the size of the data matrix
    print('%s channels x %s samples' % (len(raw.info['ch_names']), len(raw.times)))
    # Give the sample rate
    print('Sample rate:', raw.info['sfreq'], 'Hz')
    
    #Give Channels
    print('Channels:',raw.info['ch_names'])
    print('EEG: ', list(raw.copy().pick_types(eeg=True).info['ch_names']))
    print('EOG: ', raw.copy().pick_types(eog=True).info['ch_names'])
    #Brainvision EMG son misc pero lo cambie a emg
    print('EMG: ', raw.copy().pick_types(emg=True).info['ch_names'])     
    print('Time min: %s seg. Time max: %s seg. ' % (raw.times.min(), raw.times.max()))
    print()

def set_sleep_stages(raw,path_stages): #Colored sleep stages
    stages= np.loadtxt(path_stages,delimiter =' ', usecols =(0) )
    n_anot= stages.shape[0]
    epoch_length=30
    onset = np.zeros((n_anot))        
    duration = np.zeros((n_anot))    
    description = np.zeros((n_anot))  
    start=0

    for i in range(n_anot):
        onset[i] = start
        duration[i] = epoch_length 
        description[i] = stages[i]
        start= start + epoch_length
    
    stages_anot= mne.Annotations(onset,duration,description, orig_time=raw.annotations.orig_time)    
    reraw = raw.copy().set_annotations(stages_anot)

    return reraw, stages_anot

def LPF(senal):  #Low pass filter (Butterworth - Fc = 30 Hz - Order = 5)

    order = 5
    sampling_freq = 200
    cutoff_freq = 30
    sampling_duration = 30
    number_of_samples = sampling_freq * sampling_duration
    time = np.linspace(0, sampling_duration, number_of_samples, endpoint=False)

    normalized_cutoff_freq = 2*30/sampling_freq #Normalized_cutoff_freq = 2 * cutoff_freq / sampling_freq
    order_filter=5

    numerator_coeffs, denominator_coeffs  = signal.butter(order_filter, normalized_cutoff_freq)  #Return two arrays: numerator and denominator coefficients of the filter.
    
    filtered_signal = signal.lfilter(numerator_coeffs, denominator_coeffs, senal)
    
    return filtered_signal

def BP(senal): #Band pass filter (Chebyshev II - Fc1 = 0.125 Hz - Fc2 = 6 Hz - Order = 16)

    order = 16
    rs= 40   #The minimum attenuation required in the stop band. Specified in decibels, as a positive number.
    Wn= [0.125, 6]   #Critical frequencies (For type II filters, this is the point in the transition band at which the gain first reaches -rs. 
    #For digital filters, Wn are in the same units as fs. Wn is in half-cycles / sample.

    sos = signal.cheby2(order, rs, Wn, btype='bandpass', analog=False, output='sos', fs=100)
    
    filtered = signal.sosfilt(sos, senal)

    return filtered

def filtering(raw,name_channel,sf):
    ###Extracting data, sampling frequency and channels names
    data, sf, chan=raw._data, raw.info['sfreq'], raw.info['ch_names']
    n = data.shape[1]   # Number of samples
    channel = (raw.ch_names).index(name_channel)
    data=data[channel][:]    
    data=data.tolist()
    
    pos = 0 
    data_out = np.zeros([1,len(data)])

    while pos < n-1 :  #Windows of five seconds to filter the signal
        
        if pos + sf * 5 < n-1 :  #Each 5 second window instead of the last one
           
            vector = data[pos:pos+int(5*sf)]

            # 1° Downsampling 200 Hz --> 100 Hz
            data_downsampled = signal.decimate(vector, 2, n=None, ftype='iir', axis=- 1, zero_phase=True)

            # 2° Signal - avg(signal)
            data_referenced =  data_downsampled - np.mean(data_downsampled)

            # 3° Low-pass Fc = 30 Hz
            data_LP = LPF(data_referenced)

            # 4° Band-pass Cheby II 0.5-4 Hz
            x = BP(data_LP)

            # 5° Oversampling 100 Hz --> 200 Hz   //// To plot this signal, otherwise it can't be plotted with mne tools
            dat = signal.resample(x, len(vector), t=None, axis=0, window=None) 

            data_out[0,pos:pos+int(5*sf)] = dat

        else:   #The last part of the signal to filter (which could be less than 5 seconds)

            vector = data[pos:n-1]

            # 1° Downsampling 200 Hz --> 100 Hz
            data_downsampled = signal.decimate(vector, 2, n=None, ftype='iir', axis=- 1, zero_phase=True)

            # 2° Signal - avg(signal)
            data_referenced =  data_downsampled - np.mean(data_downsampled)

            # 3° Low-pass Fc = 30 Hz
            data_LP = LPF(data_referenced)

            # 4° Band-pass Cheby II 0.5-4 Hz
            x = BP(data_LP)

            # 5° Oversampling 100 Hz --> 200 Hz   //// In order to plot this signal, otherwise it can't be plotted with mne tools
            dat = signal.resample(x, len(vector), t=None, axis=0, window=None) 
            data_out[0,pos:n-1] = dat 

        pos = pos + int(5*sf)

    return data_out

def extract_td(x):  #Extract the position and duration of the labeled 'KC'
    
    vector = np.zeros([1,2])

    for i in range(len(x)-1):

        if x[i] == ',' and x[i+1] != 'K':

            start =  float(x[0:i])
            vector[0,0] = start           
            duration = float(x[i+1:len(x)-4])
            vector[0,1] = duration

    return vector
            
def KCs(raw,name_channel, sf, filetxt): #Return a vector with only KCs labeled (everything else is zero)

    data, sf, chan=raw._data, raw.info['sfreq'], raw.info['ch_names']   
    n = data.shape[1]   # Number of samples  
    channel = (raw.ch_names).index(name_channel)
    data=data[channel][:]    
    data=data.tolist()

    data_out = np.zeros([1,len(data)])

    for x in filetxt:
        if x[len(x)-3] == 'K' and x[len(x)-2] == 'C':
            vector = extract_td(x)  #vector[0,0] = position of KC // vector[0,1] = duration of KC
            start = int(vector[0,0]*sf)
            end = int((vector[0,0] + vector[0,1])*sf)
            data_out[0,start:end] = data[start:end]

    return data_out

def pulse(time_shape,sfreq): #To generate 0.5 sec x-axes
    t = np.linspace(1, round(time_shape/sfreq), time_shape, endpoint=False)    # Create artificial signal with a 0.5 sec pulse
    pulso = signal.square(2 * np.pi * 1 * t) # Pulse signal
    return pulso

def subtraction_eog(raw): #Substraction of EOGs signals
    eog1= raw.get_data(picks='EOG1_1') 
    eog2= raw.get_data(picks='EOG2_1')  
    sub_eog = eog1-eog2
    return sub_eog

def subtraction_emg(raw): #Substraction of EMGs signals
    emg1= raw.get_data(picks='EMG1_1') 
    emg2= raw.get_data(picks='EMG2_1')   
    sub_emg = emg1-emg2
    return sub_emg

def re_esctructure(raw, filetxt): #Re-estructure data
    data,sfreq =raw.get_data(),raw.info['sfreq']  
    time_shape = data.shape[1]
    
    sub_eog=subtraction_eog(raw)
    sub_emg =subtraction_emg(raw)

    pos_c3 = (raw.ch_names).index('C3_1')
    c3_1 = data[pos_c3,:]
    pos_c4 =(raw.ch_names).index('C4_1')
    c4_1 = data[pos_c4,:]

    step=int(sfreq/4) #every half second

    new_data=data.copy()
    new_data[0]= sub_eog
    new_data[1]= sub_emg
    new_data[2]= pulse(time_shape,sfreq)
    new_data[3] = KCs(raw, 'C4_1', 200, filetxt)
    new_data[4]= c4_1    
    new_data[5]= filtering(raw,'C4_1',200)

    new_data=new_data[[0,1,2,3,4,5], :]

    new_ch_names = ['EOG', 'EMG', 'Pulse', 'KCs', 'C4','Filtered']  
    new_chtypes = ['eog'] + ['emg']+ ['misc'] + 3 *['eeg']         # Remake channels
    
    # Initialize an info structure      
    new_info = mne.create_info(new_ch_names, sfreq=sfreq, ch_types=new_chtypes)
    new_info['meas_date'] = raw.info['meas_date']      # Record timestamp for annotations
    
    new_raw=mne.io.RawArray(new_data, new_info)        # Build a new raw object 
    new_raw.set_annotations(raw.annotations)         
    
    return new_raw

def plot(raw,n_channels,scal,order): 
    """To visualize the data"""
    raw.plot(show_options=True,
    title='Etiquetado',
    start=0,                        # Initial time to show
    duration=30,                    # Time window (sec) to plot in a given time
    n_channels=n_channels, 
    scalings=scal,                  # Scaling factor for traces.
    block=True,
    order=order)

############# Main function ##########################################################
def main():  # Wrapper function
    messagebox.showinfo(message ="This program allows you to tag a specific event.", title="Info")

    path = easygui.fileopenbox(title ='Select VHDR file.')
    raw=load_brainvision_vhdr(path) 
    show_info(raw)
    path_states = easygui.fileopenbox(title='Select the hypnogram (file with extension txt).') # To select txt file of previous anotations
    raw,_ = set_sleep_stages(raw,path_states)
    
    path_KC = easygui.fileopenbox(title ='Select .txt file with KCs positions and durations.')
    file_KC = open(path_KC, "r")  # Read txt file
    
    raw = re_esctructure(raw, file_KC)
    
    show_info(raw)
    
    #For actual EEG/EOG/EMG/STIM data different scaling factors should be used.
    scal = dict(eeg=20e-5, eog=150e-5,emg=15e-4, misc=1e-3) #, stim=10e-5)
    n_channels = 6
    order=[2,4,3,5,1,0]  # pulse - C4 - KCs - filtered C4 - EMG - EOG

    # Plot it!
    plot(raw,n_channels,scal,order)

    # WARNING: This script don't include any line to save data, it is just to visualize different steps. 
    # To save data you can include the following lines: 
    #aw.annotations.save(file_name(path)+ ".txt")
    #raw.save(file_name(path)+  ".fif",overwrite=True)  
    #print('Scoring was completed and the data was saved.')

if __name__ == '__main__':
    main()