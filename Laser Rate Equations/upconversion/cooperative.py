"""
Simulate the effect of cooperative upconversion between Er ions on the PL decay time. 

In this process, energy is transferred between two excited Er ions, leaving one of both ions de-excited.
This applies where a high fraction of excited Er ions is present.

[1] Concentration quenching in erbium implanted alkali silicate glasses, Snoeks, E. Kik, P G, Polman, A
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit


def decay_fn(t, a, tau, c):
    """Single-exponential decay fluorescence function."""
    return a * np.exp(-t / tau) + c


def dndt(n, t, tau=10, C=3E-24, N=1.5E19):
    """
    Rate equation for population inversion when upconversion is present. Default params for sodalime glass [1].
    
    :param n: fractional population inversion
    :param tau: Spontaneous de-excitation rate in absence of migration-quenching (tau_rad + tau_nrad) [ms]
    :param C:  Upconversion coefficient [m3/s].
    :param N: Er concentration [at./cm3]
    :return: dn/dt
    """
    return -n / tau - C * N * n ** 2


if __name__ == "__main__":
    # Material parameters
    tau = 10
    C = 3E-24
    N = 1E21  # 1.5E19

    # Time to simulate over
    t = np.linspace(0, 1000, num=100)

    # Starting population inversions
    n_list = np.linspace(1E-3, 1, num=20)

    tau_list = []
    tau_std_list = []
    for n in n_list:
        # Evaluate ODE to get y data
        y = odeint(dndt, y0=n, t=t, args=(tau, C, N))
        y = y[:, 0]

        # Fit a single exp. decay function using Levenberg-Marquardt algorithm
        popt, pcov = curve_fit(decay_fn, t, y, p0=[max(y), 10, min(y)])

        # Error in coefficients to 1 std.
        perr = np.sqrt(np.diag(pcov))

        tau_list.append(popt[1])
        tau_std_list.append(perr[1])

    fig, ax = plt.subplots()
    # ax.plot(n_list, tau_list)
    ax.errorbar(n_list, tau_list, yerr=tau_std_list)
    ax.set_xlabel(r'n(t=0)')
    ax.set_ylabel(r'tau (ms)')
    ax.ticklabel_format(useOffset=False)
    plt.show()
