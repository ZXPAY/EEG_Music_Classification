import numpy as np
from scipy.signal import butter, lfilter
from scipy.signal import freqz
import matplotlib.pyplot as plt

# Band pass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a


def BandPassFilter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Lowpass filter
def butter_lowpass(cutOff, fs, order=5):
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    b, a = butter(order, normalCutoff, btype='low')
    return b, a

def butter_lowpass_filter(data, cutOff, fs, order=4):
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Plot Data on Frequency domain -- Function
def PlotDataOnFreqDomain(FFTData, SamplingFrequency, LowerFreq=8, HigherFreq=30, title=None):
    PlotFreq = np.linspace(LowerFreq, HigherFreq, FFTData.shape[0])
    plt.plot(PlotFreq, FFTData, '*')
    if title is None:
        plt.title('Frequency Domain Plot')
    else:
        plt.title(title)
    plt.xlabel('Frequency(Hz)')
    plt.ylabel('micro volt (mV)')
    plt.grid(True)
    plt.show()

def BandPassListFilter(ListData, lowcut, highcut, fs, order=5):
    ListDataCollect = []
    for i in range(len(ListData)):
        ListDataCollect.append(BandPassFilter(ListData[i], lowcut, highcut, fs, order=5))
    return ListDataCollect
    
    
