import numpy as np
import matplotlib.pyplot as plt

from scipy.special import lambertw
from numpy import exp, log, e
 

def lambertDecayPopInv(t, *p):
  """
  Lifetime for population inversion regime.
  """
  sigma_21 = 1
  tau = 10
  alpha, n20 = p

  arg = -alpha*sigma_21*n20*exp(-(t+alpha*sigma_21*n20*tau)/tau)
  return -lambertw(arg).real/(alpha*sigma_21)


def lambertDecay(t, *p):
    """
    Lifetime for general regime.
    """
    alpha, n20 = p

    tau = 10
    n = 1
    sigma_12 = 1 
    sigma_21 = 1

    B = 1 + alpha*sigma_12*n
    A = (alpha*(sigma_12+sigma_21)) / B

    arg = -A*n20*exp(-(t/(B*tau))-(A*n20))

    # Check that result is real
    assert min(arg) >= -1/e, \
    'Lambert W Argument will give an imaginary result.'

    return -lambertw(arg).real/A


def minimizeFit():
    """
    http://tinyurl.com/jpjw6b5

    Theoretical fit of the population regime lambertW decay 
    using either scipy's minimize routine.

    Data is first created using p0 and then noise is added before
    the fit is performed.
    """

    import warnings
    warnings.filterwarnings('ignore')
    from scipy.optimize import minimize

    # generate noisy data with known coefficients
    alpha = 2
    n20 = 0.8 

    p0 = [alpha, n20]  # Guess for dependent data
    t = np.linspace(0,30,1000)
    y = lambertDecay(t,*p0)
    ynoised = y + np.random.normal(0, 0.05, np.size(t))

    ## mean squared error wrt. noisy data as a function of the parameters
    # err = lambda p: np.mean((model_func(t,*p)-data)**2)

    # Sum-Squared-Error Cost Function wrt. data as a function of the parameters
    err = lambda p: np.sum((ynoised-lambertDecay(t,*p))**2)

    # bounded optimization using scipy.minimize
    p_init = p0
    popt = minimize(
        err, # minimize wrt to the noisy data
        p_init, 
        bounds=[(0,None),(0.5,1)], # set the bounds
        method="L-BFGS-B" # this method supports bounds
    ).x

    # plot everything
    plt.figure()
    plt.scatter(t, ynoised, alpha=.2, label='f + noise')
    plt.plot(t, y, 'k-', label='f')
    plt.plot(t, lambertDecay(t, *popt), 'r--', lw=2, label="Fitted Curve")
    plt.axhline(1/np.e, ls='--', color='k', label='1/e')

    plt.title('Alpha: %.2f (%.2f), n20: %.2f (%.2f)' % \
        (popt[0], alpha, popt[1], n20))
    plt.legend(loc='best', title='alpha')
    plt.ylabel('$n_2$/max($n_2$)')
    plt.xlabel('time (ms)')
    plt.xlim(min(t),max(t))
    plt.yscale('log')
    plt.savefig('Images/MinimizeFitGeneral.png', dpi=300)
    plt.show()

    print(p0, popt)


def curveFit():
    """
    Theoretical fit of the population regime lambertW decay 
    using either the 

        1. Levenberg-Marquardt ('lm') algorithm
        2. tfr algorithm, which can take bounds

    Data is first created using p0 and then noise is added before
    the fit is performed.

    """
    from scipy.optimize import curve_fit

    # Create noisey data
    alpha = 2
    n20 = 0.8
    t = np.linspace(0,30,1000)
    p0 = [alpha, n20]  
    y = lambertDecay(t,*p0)
    ynoised = y + np.random.normal(0, 0.05, np.size(y))

    # Fit data to model using Levenberg-Marquardt algorithm
    p_init = p0 # Initial parameter guesses
    popt, pcov = curve_fit(lambertDecay, t, ynoised, p0=p_init, method='lm')
    # 'trf' can take bounds.
    # popt, pcov = curve_fit(lambertDecay, t, ynoised, p0=p0, 
    #   bounds=[(0,None),(0.5,1)], 
    #   method='trf')

    # Produce plots
    plt.figure()
    plt.plot(t, y, 'k-', label='f')
    plt.scatter(t, ynoised, alpha=.2, label='f + noise')
    plt.plot(t, lambertDecay(t, *popt), 'r-', label="Fitted Curve")
    plt.axhline(1/np.e, ls='--', color='k', label='1/e')

    plt.title('Alpha: %.2f (%.2f), n20: %.2f (%.2f)' % \
        (popt[0], alpha, popt[1], n20))
    plt.legend(loc='best')
    plt.xlabel('time (ms)')
    plt.xlim(min(t),max(t))
    plt.ylabel('$n_2$/max($n_2$)')
    plt.yscale('log')
    plt.savefig('Images/curveFitGeneral.png', dpi=300)
    plt.show()

    print(popt, p0)
    

if __name__ == "__main__":
    # curveFit()
    minimizeFit()