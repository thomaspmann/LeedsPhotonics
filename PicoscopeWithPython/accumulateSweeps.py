# -*- coding: utf-8
# Adapted from example by Colin O'Flynn
#
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import time
import numpy as np
from picoscope import ps5000a


class decayMeasure():
    def __init__(self):
        self.ps = ps5000a.PS5000a(connect=False)

    def openScope(self):
        self.ps.open()
        bitRes = 15
        self.ps.setResolution(str(bitRes))
        self.ps.setChannel("A", coupling="DC", VRange=5.0, VOffset=0, enabled=True)
        self.ps.setChannel("B", enabled=False)

        waveformDuration = 10E-3
        obsDuration = 100*waveformDuration

        sampleFreq = 1E6
        sampleInterval = 1.0 / sampleFreq

        # Returns: res = (actualSampleInterval, noSamples, maxSamples)
        res = self.ps.setSamplingInterval(sampleInterval, obsDuration)

        print("Resolution =  %d Bit" % bitRes)
        print("Sampling interval = %.f ns" % (res[0] * 1E9))
        print("Sampling frequency = %.3f MHz" % (1E-6/res[0]))
        print("Taking  samples = %d" % res[1])
        print("Maximum samples = %d" % res[2])

        self.res = res

        #Use external trigger to mark when we sample
        self.ps.setSimpleTrigger(trigSrc="A", threshold_V=0.0, direction="Falling", timeout_ms=5000)

        self.ps.setSigGenBuiltInSimple(offsetVoltage=0, pkToPk=4, waveType="Sine",
                              frequency=1/waveformDuration, shots=1, triggerType="Rising", 
                              triggerSource="None")

    def closeScope(self):
        self.ps.close()

    def armMeasure(self):
        self.ps.runBlock()
        
    def measure(self):
        print("Waiting for trigger")        
        while(self.ps.isReady() == False): time.sleep(0.01)
        print("Sampling Done")

        dataA = self.ps.getDataV("A")

        return dataA

    def accumulate(self):

        fig = plt.figure()
        plt.ion()
        plt.show()

        dataTimeAxis = np.arange(self.res[1]) * self.res[0] * 1E3
        blockdata = np.array(0)
        for i in range (0, 5):
            print("Measurement %d" % i)
            dm.armMeasure()
            data = dm.measure()
            blockdata = blockdata + data

            plt.clf()
            plt.plot(dataTimeAxis, blockdata, label="Waveform")
            plt.grid(True, which='major')
            plt.title("Picoscope 5000a waveforms")
            plt.ylabel("Voltage (V)")
            plt.xlabel("Time (ms)")
            plt.legend()
            plt.draw()

        plt.show(block=True)

if __name__ == "__main__":
    dm = decayMeasure()
    dm.openScope()

    dm.accumulate()

    dm.closeScope()
