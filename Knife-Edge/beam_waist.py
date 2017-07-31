import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm


def load_data():
    return xdata, ydata


# Eq 15
def f(z):
    # Polynomial coefficients
    a = [-6.71387E-3, -1.55115, -5.13306E-2, -5.49164E-2]
    # Polynomial expansion
    p = lambda z: a[0] + a[1] * z ** 1 + a[2] * z ** 2 + a[3] * z ** 3
    return 1 / (1 + np.exp(p(z)))


# Eq 10
def func(z, beta, x0):
    return z / (np.sqrt(2) * beta) + x0


if __name__ == "__main__":
    # Load x and y data
    xdata, ydata = load_data()

    # Normalise y data
    ydata /= max(ydata)

    # Calc z from y data (No need to use Eq15 approximation)
    zdata = norm.ppf(ydata)

    popt, pcov = curve_fit(func, zdata, ydata)
    print(popt)
