"""
Simulate the effect of cooperative upconversion between Er ions on the PL decay time. 

In this process, energy is transferred between two excited Er ions, leaving one of both ions de-excited.
This applies where a high fraction of excited Er ions is present.

[1] Concentration quenching in erbium implanted alkali silicate glasses, Snoeks, E. Kik, P G, Polman
[2] Erbium implanted thin film photonic materials, Polman
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import h, c, pi
from scipy.integrate import odeint
from scipy.optimize import curve_fit


def decay_fn(t, a, tau, const):
    """Single-exponential decay fluorescence function."""
    return a * np.exp(-t / tau) + const


def w(w_r=42, w_i=0, c_erer=2.3E-39, n_q=8E18, n_er=1.4E20):
    """
    [1] Decay rate due to energy migration.
    
    Assumptions:
        (1) The absorption spectrum of a quenching site is identical to the absorption spectrum of an Er 3+ ion
            in sodalime glass. This simplification may be made in the case that the quenching mechanism is via 
            excitation of vibrations in -OH closely bound to a limited number of Er ions.

        (2) A constant density of quenching centers Nq, which implies that this density is independent of Er 
            fluence, and therefore these quenchers are not related to the damage created by the ion beam

        (3) All free OH contributes to quenching so that N_q = N_OH

    Note that W_r + W_i is the decay rate in absence of migration.

    :param w_r: Radiative PL decay rate (i.e. 1/tau) of
    :param w_i: Non-radiative decay rate (~= 0 when there are no implantation induced defects)
    :param c_erer: Er-Er interaction constant [cm6/s]
    :param n_q: Density of quenching sites [cm-3]
    :param n_er: Density of Er ions [cm-3]
    :return: Decay rate, w
    """
    return w_r + w_i + 8 * pi * c_erer * n_q * n_er


def dn2dt(n2, t, tau=10E-3, c_up=3E-18, n_er=1.4E20, lam_pump=980, i_p=0):
    """
    [2] Rate equation for population inversion when upconversion is present. 
    Default params for Er doped sodalime glass with a pump operating @ 980nm.

    :param n2: fractional population inversion
    :param tau: Spontaneous de-excitation rate [s]
    :param c_up:  Upconversion coefficient [cm3/s].
    :param n_er: Er concentration [at./cm3]
    :param lam_pump: Pump wavelength (either 980 or 1480) [nm]
    :param i_p: Intensity of the pump
    :return: dn2/dt
    """
    if lam_pump == 980:
        sig_a = 1E-21
        sig_e = 0
    elif lam_pump == 1480:
        sig_a = 1E-21
        sig_e = 0.5E-21
    else:
        raise ValueError('Pump wavelenght must be either 980 or 1480.')
    v_p = c / (lam_pump * 1E-9)

    r_up = i_p * sig_a / (h * v_p)
    r_down = i_p * sig_e / (h * v_p)
    return r_up * (1 - n2) - r_down * n2 - n2 / tau - c_up * n_er * n2 ** 2


# Plotting functions
def plot_migration():
    """Plot the decay rate as a function of Er concentration due to energy migration."""
    # Material parameters
    # N_er_list = np.linspace(1E19, 1.5E21, num=1000)
    N_er_list = np.linspace(1E17, 1.5E19, num=1000)

    w_list = []
    for N_er in N_er_list:
        w_list.append(w(n_er=N_er))

    fig, ax = plt.subplots()
    # ax.plot(N_er_list, w_list)
    # ax.set_ylabel(r'w (/s)')
    ax.plot(N_er_list, 1E3 / np.array(w_list))
    ax.set_ylabel(r'tau (ms)')
    ax.set_xlabel(r'N_er (at./cm3)')

    # ax.ticklabel_format(useOffset=False)
    plt.show()


def plot_quenching():
    """Plot the lifetime as a function of starting population inversion."""

    # Material parameters
    n_er = 2.7E20
    tau = 1 / w(n_er=n_er)  # Calculate lifetime (s)
    print(tau * 1E3)

    # Time to simulate over (s)
    t = np.linspace(0, 120E-3, num=1000)

    # Starting population inversion fraction
    n_list = np.linspace(1E-3, 1, num=20)

    tau_list = []
    tau_std_list = []
    for n in n_list:
        # Evaluate ODE to get y data
        y = odeint(dn2dt, y0=n, t=t, args=(tau, 3E-18, n_er,))
        y = y[:, 0]

        # Fit a single exp. decay function using Levenberg-Marquardt algorithm
        popt, pcov = curve_fit(decay_fn, t, y, p0=[max(y), 10E-3, min(y)])

        # Error in coefficients to 1 std.
        perr = np.sqrt(np.diag(pcov))

        tau_list.append(popt[1])
        tau_std_list.append(perr[1])

    # Convert to ms
    t *= 1E3
    tau_list = np.array(tau_list) * 1E3
    tau_std_list = np.array(tau_std_list) * 1E3

    fig, ax = plt.subplots()
    # ax.plot(n_list, tau_list)
    ax.errorbar(n_list, tau_list, yerr=tau_std_list, fmt='--')
    ax.set_xlabel(r'n(t=0)')
    ax.set_ylabel(r'tau (ms)')
    ax.ticklabel_format(useOffset=False)

    fig, ax = plt.subplots()
    ax.plot(n_list, tau_std_list)
    plt.show()


if __name__ == "__main__":
    plot_migration()
    # plot_quenching()
