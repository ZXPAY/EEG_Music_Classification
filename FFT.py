import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt

def nextpow2(n):
    m_f = np.log2(n)
    m_i = int(np.ceil(m_f))
    return int(2**m_i)

def FFT(data, sample_freq, data_length):
    n = nextpow2(data_length)
    Y = np.abs(fft.fft(data, n)/n)
    Y = 2 * Y[0:int(n / 2)]
    x = sample_freq / 2 * np.linspace(0, 1, n / 2)
    return  x, Y

# Plot Raw Data on Time domain -- Function
def PlotDataOnTimeDomain(RawData, SamplingFrequency, LowerYLim=4000, HigherYLim=5000, title=None):
    SamplesPerMinute = SamplingFrequency*60   # Unit : samples/ minute
    DataLength = RawData.shape[0]
    ConsumingTime = DataLength/SamplesPerMinute   # Unit : minites
    XTicksMinsList = []
    XTicksSamplesList = []
    for i in range((int(ConsumingTime)+1)):
        XTicksMinsList.append(str((i))+'min')
        XTicksSamplesList.append(i*SamplesPerMinute)
    PlotXAxis = np.linspace(0, DataLength, DataLength)
    plt.plot(PlotXAxis, RawData)
    plt.xticks(XTicksSamplesList, XTicksMinsList, rotation=45)    # format the ticks
    if title is None:
        plt.title('Signal on Time Domain')
    else:
        plt.title(title)
    plt.ylim(LowerYLim, HigherYLim)
    plt.ylabel('micro voltage')
    plt.xlabel('Time(min)')
    plt.grid(True)
    plt.show()


'''Example:
import matplotlib.pyplot as plt
Fs = 1000  # sample frequency
T = 1/Fs
L = 1000  # data length

t = np.arange(L)*T
y = 1*np.cos(2*np.pi*200*t)

x, Y = FFT(y, Fs, L)
plt.plot(x,Y)
plt.grid(True)
plt.show()

'''
'''Example:
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

# Sample rate and desired cutoff frequencies (in Hz).
fs = 5000.0
lowcut = 500.0
highcut = 1250.0

# Plot the frequency response for a few different orders.
plt.figure(1)
plt.clf()
for order in [3, 6, 9]:
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    w, h = freqz(b, a, worN=2000)
    plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
         '--', label='sqrt(0.5)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.grid(True)
plt.legend(loc='best')

# Filter a noisy signal.
T = 0.05
nsamples = T * fs
t = np.linspace(0, T, nsamples, endpoint=False)
a = 0.02
f0 = 600.0
x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
x += a * np.cos(2 * np.pi * f0 * t + .11)
x += 0.03 * np.cos(2 * np.pi * 2000 * t)
plt.figure(2)
plt.clf()
plt.plot(t, x, label='Noisy signal')

y = butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
plt.xlabel('time (seconds)')
plt.hlines([-a, a], 0, T, linestyles='--')
plt.grid(True)
plt.axis('tight')
plt.legend(loc='upper left')

plt.show()

'''