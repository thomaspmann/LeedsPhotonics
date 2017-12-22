from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

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

def R(xb, w, x0):
    beta = 1/w
    
    def integrand(x, beta, x0):
        return np.exp(-beta**2 * (x-x0)**2)

    I = quad(integrand, -np.inf, xb, args=(beta, x0))[0]
    return np.sqrt(np.pi/beta**2) * I


if __name__ == "__main__":
    # Load experimental data [position, intensity]
    data = np.genfromtxt('measurement.txt', delimiter='\t', skip_header=1)
    xdata = data[:, 0]
    ydata = data[:, 1]

    # Normalise y data
    ydata /= max(ydata)

    # Calc z from y data (No need to use Eq15 approximation)
    zdata = norm.ppf(ydata)

    # Fit
    popt, pcov = curve_fit(func, zdata, ydata)
    print('Fitted: w = {:g}, x0 = {:g}'.format(1/popt[0], popt[1]))
    
    # Plot fit and data
    fig, ax = plt.subplots()
    ax.plot(xb, y, '.', label='raw data')
    
    xb = np.linspace(min(x), max(x), num=50)
    R_vectorized = np.vectorize(R)
    yfit = R_vectorized(xb, 1/popt[0], popt[1])
    ax.plot(xb, yfit, label='fitted')
    plt.legend()
