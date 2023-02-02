#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 13:16:47 2022

@author: daraghkneeshaw
"""

import librosa as lr
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import math

fs = 44100 # sample rate
basedir='./WavFiles/' #for reading in wavs

#%% helper functions

# Read in wav file and return signal array and time
def getSignal (signal,fs):
    y, fs = sf.read(basedir+signal) 
    time = np.arange(0,len(y)/fs, 1/fs) 
    return y, time

# get time array for signal to plot on x axis
def getTimeArr(sig,fs):
    time = np.arange(0,len(sig)/fs, 1/fs) 
    return time

# write signal to wav file
def writeToWav(signal, fs):
    name = input("what would you like to call this?")
    sf.write('{}.wav'.format(name), signal,fs)

#  normalise any array by dividing each amplitude by the absolute maximum
#  gives us an absolute max aplitude of 1 to avoid clipping, as convolution
#  greatly amplifies the signals
def normalise(array):
    arraynp = np.array(array)
    normarr = arraynp/np.abs(arraynp).max()
    return normarr

# convolve signal with impulse in time domain
def timeConvolve(signal,impulse):
    y = [] # empty array for convolved signal
       
# get amount of 0s needed for adding to beggning and end of original signal
    padding = np.zeros(len(impulse)-1)
    
# add 0 padding to start and end of original signal
# this allows us to perform the dot product of the impulse and original at
# every point
    sigpad = np.concatenate((padding,signal,padding))
   
#  reverse the impulse signal
#  enables us to perform convolution as formula is sum of x(k)h(n-k)
    imprev = np.flip(impulse)

# length of convolved signal is sum of lenght of both signals - 1
    len_result = len(signal) + len(impulse) - 1
    
# iterate through the length of the new signal,
#  adading the dot product of the impulse and original signal as it moves along
    for i in range(0, (len_result)):
        y.append(np.dot(imprev,sigpad[i:(i+len(imprev))]))
    
    # normalise convolved sig
    normY = normalise(y)

    return normY

# add signals together
# amp = the amplitude of "wet" (convolved) signal
# we must add padding to the original so we can add the signals
def addsigs(original, convolved,amp):
# get the difference in length between the signals
    diff = len(convolved) - len(original)
# make array of 0s for padding
    padding = np.zeros(diff)
# add padding to end of origina and sum arrays
    neworig = np.concatenate((original,padding))
    added = neworig + amp*convolved
    addednorm = normalise(added)
    
    return addednorm
 

# # # FFT Convolution

# find the closest power of two that's >= n
# FFT algorithm works more efficiently on arrays whose length is a power of 2
# log to the base two of n and 2 to the power of the next number up
def next_power_of_2(n):
    next = pow(2, math.ceil(math.log2(n)))
    return next

# add 0 padding to an array
def addPadding(orig,length):
    padding = np.zeros(length)
    newsig = np.concatenate((orig,padding))
    return newsig
    

# perform convolution in Frequency domain (pointwise multiplication)
def FFTConvolution(signal,impulse):
    
    siglen = len(signal)
    implen = len(impulse)
    # length of convolved array will be length of signal + length of impulse -1
    convlen = siglen + implen - 1
    # find next power of two as above
    convlen2 = next_power_of_2(convlen)
    
    # add padding to end of both arrays
    # both will be of length convlen2 with 0s at the end
    sigpadding = addPadding(signal, convlen2 - siglen)
    imppadding = addPadding(impulse, convlen2 - implen)
    
    # perform FFT on both signals
    sigfft = np.fft.fft(sigpadding, n=convlen2)
    impfft = np.fft.fft(imppadding, n=convlen2)
    
    # convolve by pointwise multiplication of the arrays
    convFFT = sigfft*impfft
    
    # convert the convolved signal back to time domain with inverse FFT
    # take only the real values
    invFFT = np.real(np.fft.ifft(convFFT))
    # print(len(invFFT))
    # take only to length of convlen, as we don't need the extra that were
    # added to make FFT more efficient
    invFFT = invFFT[0:convlen-1]
    # print(len(invFFT))
    
    # normalise the signal
    normalsig = normalise(invFFT)
    
    return normalsig
 
#%%    
# # # Functions for producing full mixed signals
    
# input both wav files, the SR and amplitude of wet signal you want to add
def fullTimeConv(sig,imp,fs,amp):
    
    # read in both wav files
    mainsig, maintime = getSignal(sig, fs)
    impulse, imptime = getSignal(imp, fs)
    
    # convolve usin above function
    y = timeConvolve(mainsig,impulse)
    
    #  add signals with above function
    fullsig = addsigs(mainsig,y,amp)
    
    # get time array for plotting
    time = getTimeArr(fullsig, fs)
    
    #  write to a wav file
    writeToWav(fullsig, fs)
    
    return fullsig, time

#  uncomment to test function
# fullTimeConv('guitar.wav', 'artificial.wav', fs, .4)

# input both wav files, the SR and amplitude of wet signal you want to add  
def fullFFTConv(sig,imp,fs,amp):

    # read in both wav files
    mainsig, maintime = getSignal(sig, fs)
    impulse, imptime = getSignal(imp, fs)
   
    # convolve using above FFT convolve function
    y = FFTConvolution(mainsig,impulse)
   
    #  add signals with above function
    fullsig = addsigs(mainsig,y,amp)
    
    # get time array for plotting
    time = getTimeArr(fullsig, fs)
    
    #  write to a wav file
    writeToWav(fullsig, fs)
    return fullsig, time

#  uncomment to test function
# fullFFTConv('guitar.wav', 'church.wav', fs, .4)

#%%

# # # Functions for plotting 

# input the singal you want to plot, the sampling rate and how much of the 
# signal you want to show
def plotintime(signal,fs,xlimit=[]):
    # eg xlim([0,2]) - shows first 2 seconds
    
    # read in wav
    sig,sigtime = getSignal(signal, fs)
    
    # plot the signal
    plt.plot(sigtime,sig)
    #label the axes
    plt.xlabel("time (secs)")
    plt.ylabel("Amplitude")
    # show certain amoung of signal
    if xlimit != []:
        plt.xlim(xlimit)
    return

# plt.figure()
# plotintime('guitar.wav', fs,[])

# input the wav and the sampling rate
def spectrogram(signal,fs):
    # read in wav
    sig,sigtime = getSignal(signal, fs)
    # get spectrogram of short time fourier transform
    stft = np.abs(lr.stft(sig))
    # convert the power spectogram to db
    log = lr.power_to_db(np.abs(stft**2),ref=np.max)
    # plot it
    librosa.display.specshow(log, x_axis='time', y_axis='log', sr = fs)

    #  show legend for decibel level
    plt.colorbar()
    return

# plt.figure()
# spectrogram('guitar.wav', fs)



