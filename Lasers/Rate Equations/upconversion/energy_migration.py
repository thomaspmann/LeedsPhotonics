"""
Simulate the effect of cooperative upconversion between Er ions on the PL decay time. 

it is confirmed that the strong increase of PL decay rate with increasing Er fluence is not 
explained by accumulation of beam-induced damage, but by an Er concentration quenching effect,
i.e. migration followed by quenching at a fixed density of quenching sites.

[1] Concentration quenching in erbium implanted alkali silicate glasses, Snoeks, E. Kik, P G, Polman, A
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi


def w(w_r=42, w_i=0, C=2.3E-39, N_q=0.81E19, N_er=1.5E21):
    """
    [1] Concentration quenching in erbium implanted alkali silicate glasses, Snoeks, E. Kik, P G, Polman, A
    
    Assumptions:
        (1) The absorption spectrum of a quenching site is identical to the absorption spectrum of an Er 3+ ion
            in sodalime glass. This simplification may be made in the case that the quenching mechanism is via 
            excitation of vibrations in -OH closely bound to a limited number of Er ions.
            
        (2) A constant density of quenching centers Nq, which implies that this density is independent of Er 
            fluence, and therefore these quenchers are not related to the damage created by the ion beam
            
        (3) All free OH contributes to quenching so that N_q = N_OH
    
    Note that W_r + W_i is the decay rate in absence of migration.
    
    :param w_r: Radiative PL decay rate (i.e. 1/tau) of
    :param w_i: Non-radiative decay rate ~= 0
    :param C: Er-Er interaction constant [cm6/s]
    :param N_q: Density of quenching sites [cm-3]
    :param N_er: Density of Er ions [cm-3]
    :return: Decay rate, w
    """
    return w_r + w_i + 8 * pi * C * N_q * N_er


if __name__ == "__main__":
    # Material parameters
    N_er_list = np.linspace(1E20, 15E20, num=10)

    w_list = []
    for N_er in N_er_list:
        w_list.append(w(N_er=N_er))

    fig, ax = plt.subplots()
    ax.plot(N_er_list, w_list)
    ax.set_xlabel(r'N_er (at./cm3)')
    ax.set_ylabel(r'w (/s)')
    # ax.ticklabel_format(useOffset=False)
    plt.show()
