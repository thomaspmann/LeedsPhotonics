import numpy as np
import matplotlib.pyplot as plt

from scipy.special import lambertw
from numpy import exp, log, e
 

def minimize():
	"""

	http://stackoverflow.com/questions/16760788/python-curve-\
	fit-library-that-allows-me-to-assign-bounds-to-parameters

	Theoretical fit of the population regime lambertW decay 
	using either scipy's minimize routine.

	Data is first created using p0 and then noise is added before
	the fit is performed.

	"""

	import warnings
	warnings.filterwarnings('ignore')
	from scipy.optimize import minimize

	def model_func(t, *p):
		
		def lambertDecay(t, *p):
			sigma_21 = 1
			tau = 10
	#         n20 = 0.8
			alpha, n20 = p
			arg = -alpha*sigma_21*n20*exp(-(t+alpha*sigma_21*n20*tau)/tau)
			return -lambertw(arg)/(alpha*sigma_21)

		n = lambertDecay(t, *p)
		
		return n.real

	# generate noisy data with known coefficients
	alpha = 0.3
	n20 = 0.8 

	p0 = [alpha, n20]  # Guess for dependent data
	t = np.linspace(0,30,1000)
	n = model_func(t,*p0)
	data = n + np.random.normal(0, 0.02, np.size(n))

	## mean squared error wrt. noisy data as a function of the parameters
	# err = lambda p: np.mean((model_func(t,*p)-data)**2)

	# Sum-Squared-Error Cost Function wrt. noisy data as a function of the parameters
	err = lambda p: np.sum((data-model_func(t,*p))**2)

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
	plt.scatter(t, data, alpha=.2, label='f + noise')
	plt.plot(t, n, 'k-', label='f')
	plt.plot(t, model_func(t, *popt), 'r--', lw=2, label="Fitted Curve")

	plt.axhline(1/np.e, ls='--', color='k')
	plt.legend(loc='best', title='alpha')
	plt.ylabel('$n_2$/max($n_2$)')
	plt.xlabel('time (ms)')
	plt.xlim(min(t),max(t))
	plt.yscale('log')
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

	def model_func(t, *p):
		
		def lambertDecay(t, *p):
			sigma_21 = 1
			tau = 10
	#         n20 = 0.8
			alpha, n20 = p
			arg = -alpha*sigma_21*n20*exp(-(t+alpha*sigma_21*n20*tau)/tau)
			return -lambertw(arg)/(alpha*sigma_21)

		n = lambertDecay(t, *p)
		
		return n.real


	# Create noisey data
	alpha = 0.3
	n20 = 0.8
	p0 = [alpha, n20]
	t = np.linspace(0,30,1000)
	n = model_func(t,*p0)
	data = n + np.random.normal(0, 0.02, np.size(n))


	# Fit data to model using Levenberg-Marquardt algorithm
	popt, pcov = curve_fit(model_func, t, data, p0=p0, method='lm')
	# 'trf' can take bounds.
	# popt, pcov = curve_fit(model_func, t, data, p0=p0, bounds=[(0,None),(0.5,1)], method='trf')

	# Produce plots
	plt.figure()
	plt.plot(t, n, 'k-', label='f')
	plt.scatter(t, data, alpha=.2, label='f + noise')
	plt.plot(t, model_func(t, *popt), 'r-', label="Fitted Curve")
	plt.axhline(1/np.e, ls='--', color='k')
	plt.legend(loc='best', title='alpha')
	plt.ylabel('$n_2$/max($n_2$)')
	plt.xlabel('time (ms)')
	plt.xlim(min(t),max(t))
	plt.yscale('log')
	plt.show()

	print(popt, p0)

if __name__ == "__main__":
	# curveFit()
	minimize()