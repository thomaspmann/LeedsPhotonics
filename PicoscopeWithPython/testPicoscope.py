# -*- coding: utf-8
# Example by Colin O'Flynn
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
        self.ps.setResolution("15")
        self.ps.setChannel("A", coupling="DC", VRange=5.0, VOffset=0, enabled=True)
        self.ps.setChannel("B", enabled=False)


        waveform_desired_duration = 1  # in ms
        obs_duration = 4 * waveform_desired_duration
        sampling_interval = obs_duration / (4096)

        self.res = self.ps.setSamplingInterval(sampling_interval, obs_duration)
        print("Sampling interval = %.1f ns" % (self.res[0] * 1E9))
        print("Taking  samples = %d" % self.res[1])
        print("Maximum samples = %d" % self.res[2])

        # res = self.ps.setSamplingFrequency(sampleFreq=1E6, noSamples=5E6)
        # self.sampleRate = res[0]
        # print("Sampling @ %.4f MHz, %d samples"%(res[0]/1E6, res[1]))
        
        #Use external trigger to mark when we sample
        self.ps.setSimpleTrigger(trigSrc="A", threshold_V=0.0, direction="Falling", timeout_ms=5000)

        self.ps.setSigGenBuiltInSimple(offsetVoltage=0, pkToPk=4, waveType="Sine",
                              frequency=1/waveform_desired_duration, shots=1, triggerType="Rising", 
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
        dataTimeAxis = np.arange(self.res[1]) * self.res[0]
        
        plt.figure()
        plt.plot(dataTimeAxis, dataA, label="Waveform")
        plt.grid(True, which='major')
        plt.title("Picoscope 5000a waveforms")
        plt.ylabel("Voltage (V)")
        plt.xlabel("Time (ms)")
        plt.legend()
        # plt.savefig('data.png', dpi=500)
        plt.show(block=True)

        # print(data)
                             
if __name__ == "__main__":
    dm = decayMeasure()
    dm.openScope()
    
    # try:
    #     while 1:
    #         dm.armMeasure()
    #         dm.measure()    
    # except KeyboardInterrupt:
    #     pass
        
    dm.armMeasure()
    dm.measure()

    dm.closeScope()
