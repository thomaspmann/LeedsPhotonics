from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import time
import numpy as np

from tqdm import *
from picoscope import ps5000a
from scipy.optimize import curve_fit


def func(x, a, c, d):
    return a*np.exp(-x/c)+d


def fitMonoExponential(x, y):
    popt, pcov = curve_fit(func, x, y, p0=(max(y)-min(y), 10, min(y)))

    print('\nLifetime is: %.2f' % popt[1])

    return popt

class decayMeasure():
    def __init__(self):
        self.ps = ps5000a.PS5000a(connect=False)

    def openScope(self):
        self.ps.open()

        bitRes = 16
        self.ps.setResolution(str(bitRes))
        print("Resolution =  %d Bit" % bitRes)

        self.ps.setChannel("A", coupling="DC", VRange=20.0E-3, VOffset=0.0, enabled=True)
        self.ps.setChannel("B", coupling="DC", VRange=5.0, VOffset=0, enabled=False)
        self.ps.setSimpleTrigger(trigSrc="External", threshold_V=2.0, direction="Falling", timeout_ms=5000)

        waveformDuration = 100E-3
        obsDuration = 1*waveformDuration

        sampleFreq = 50E6
        sampleInterval = 1.0 / sampleFreq

        res = self.ps.setSamplingInterval(sampleInterval, obsDuration)
        print("Sampling frequency = %.3f MHz" % (1E-6/res[0]))
        print("Sampling interval = %.f ns" % (res[0] * 1E9))
        print("Taking  samples = %d" % res[1])
        print("Maximum samples = %d" % res[2])
        self.res = res

    def closeScope(self):
        self.ps.close()

    def armMeasure(self):
        self.ps.runBlock()

    def measure(self):
        # print("Waiting for trigger")
        while not self.ps.isReady():
            time.sleep(0.01)
        # print("Sampling Done")

        dataA = self.ps.getDataV("A")

        return dataA

    def accumulate(self, sweep_no):
        # Sum loops of data
        data = np.array(0)
        for i in tqdm(range(0, sweep_no)):
            # print("Measurement %d" % i)
            dm.armMeasure()
            data = data + dm.measure()
        plt.figure()
        dataTimeAxis = np.arange(self.res[1]) * self.res[0] * 1E3

        # Average datapoints
        x = 10000

        if self.res[1] % x != 0:
            raise ValueError('Error with number of datapoints to average.')

        dataTimeAxis = dataTimeAxis.reshape(-1, x).mean(axis=1)
        data = data.reshape(-1, x).mean(axis=1)

        # Remove first erronious data point
        dataTimeAxis = dataTimeAxis[1:]
        data = data[1:]

        popt = fitMonoExponential(dataTimeAxis, data)
        lifetime = str('%.2f' % popt[1])
        plt.plot(dataTimeAxis,data, label=lifetime)
        plt.plot(dataTimeAxis, func(dataTimeAxis, *popt), 'r--', label="Fitted Curve")
        plt.grid(True, which='major')
        plt.title("Picoscope 5000a waveforms")
        plt.ylabel("Voltage (V)")
        plt.xlabel("Time (ms)")
        plt.legend(loc="best")
        # plt.savefig('data.png', dpi=500)
        # plt.show(block=True)
        plt.show()
        plt.close()

        chip = 'T6'
        pulse = '50ms'
        ref = 'noFocus'
        title = time.strftime("%d%m%y_%H%M%S", time.gmtime())
        fname = "Data\\" + chip + '_' + ref +'_' + pulse + '_' + title + '.txt'
        dataTimeAxis = np.array(dataTimeAxis)
        saveData = np.c_[dataTimeAxis, data]
        np.savetxt(fname, saveData, newline='\r\n')

    def intensity(self):
        self.ps.setSimpleTrigger(trigSrc="External", threshold_V=2.0, direction="Falling", timeout_ms=5000,
                                 enabled=False)
        # Continually capture sweeps (press ctr+c to kill)
        try:
            while 1:
                dm.armMeasure()
                data = dm.measure()
                data = np.mean(data)
                print(data)
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    dm = decayMeasure()
    dm.openScope()
    dm.accumulate(100)
    # dm.intensity()
    dm.closeScope()
