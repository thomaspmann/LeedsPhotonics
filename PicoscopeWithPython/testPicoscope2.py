"""
Demo... stuff can go here

"""
from __future__ import division

import time
from picoscope import ps5000a
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    print(__doc__)

    print("Attempting to open Picoscope 5000a...")

    # see page 13 of the manual to understand how to work this beast
    ps = ps5000a.PS5000a()

    # print(ps.getAllUnitInfo())

    # self.resolution = self.ADC_RESOLUTIONS["15"]

    waveform_desired_duration = 1  # in ms
    obs_duration = 4 * waveform_desired_duration
    sampling_interval = obs_duration / (4096)

    (actualSamplingInterval, nSamples, maxSamples) = ps.setSamplingInterval(sampling_interval,
                                                                            obs_duration)
    print("Sampling interval = %.1f ns" % (actualSamplingInterval * 1E9))
    print("Taking  samples = %d" % nSamples)
    print("Maximum samples = %d" % maxSamples)

    ps.setChannel('A', 'DC', VRange=5.0, VOffset=0.0, enabled=True)
    ps.setSimpleTrigger('A', 0.0, 'Rising', delay=0, timeout_ms=100, enabled=True)

    ps.setSigGenBuiltInSimple(offsetVoltage=0, pkToPk=4, waveType="Sine",
                              frequency=1 / waveform_desired_duration, shots=1,
                              triggerType="Rising", triggerSource="None")

    # take the desired waveform
    # This measures all the channels that have been enabled

    ps.runBlock()
    ps.waitReady()
    print("Done waiting for trigger")
    time.sleep(10)
    ps.runBlock()
    ps.waitReady()

    dataA = ps.getDataV('A', nSamples, returnOverflow=False)

    ps.stop()
    ps.close()

    dataTimeAxis = np.arange(nSamples) * actualSamplingInterval

    plt.ion()

    plt.figure()
    plt.hold(True)
    plt.plot(dataTimeAxis, dataA, label="Waveform")
    plt.grid(True, which='major')
    plt.title("Picoscope 5000a waveforms")
    plt.ylabel("Voltage (V)")
    plt.xlabel("Time (ms)")
    plt.legend()
    # plt.savefig('hi.png', dpi=500)
    plt.show(block=True)
