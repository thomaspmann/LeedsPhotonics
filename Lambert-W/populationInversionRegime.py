import numpy as np
import matplotlib.pyplot as plt

from scipy.special import lambertw
from numpy import exp, log, e

def lambertDecay(t, alpha, tau, sigma_21, n20):
	arg = -alpha*sigma_21*n20*exp(-(t+alpha*sigma_21*n20*tau)/tau)
	return -lambertw(arg).real/(alpha*sigma_21)


def differentAlphas():
	"""
	Fit the function with different Alpha values
	"""
	t = np.linspace(0,30,1000)

	tau = 10
	sigma_21 = 1
	n20 = 0.9  # Fraction of population inversion (>0.5)

	plt.figure()
	plt.axhline(1/np.e, ls='--', color='k')
	plt.ylabel('$n_2$/max($n_2$)')
	plt.xlabel('time (ms)')

	for alpha in [0.002, 0.1, 0.3, 1, 2, 5]:
		n = lambertDecay(t, alpha,tau,sigma_21,n20)

		# Select only part of curve that is completely real
	#     index = np.where(np.isreal(n))
	#     t = t[index]
	#     n = n[index].real

		plt.plot(t,n/max(n), label=alpha)

	plt.xlim(min(t),max(t))
	plt.yscale('log')
	plt.legend(loc='best', title='alpha')
	import os
	fname = os.path.join('Images', 'decaysVsAlphan20_0p9.png')
	plt.savefig(fname, dpi=300)
	plt.show()

def Warg():
	""" 
	Plot the argument of the lambert W function vs alpha or n20
	"""
	t = np.linspace(0,30,1000)

	tau = 10
	sigma_21 = 1
	n20 = 0.8  # Fraction of population inversion (>0.5)
	alpha = 1
	Warg = []
	n = np.linspace(0.5,1, 100)
	for n20 in n:
	    arg = -alpha*sigma_21*n20*exp(-(t+alpha*sigma_21*n20*tau)/tau)
	    Warg.append(min(arg))
	plt.plot(n,Warg, '.-')
	plt.axhline(-1/e, ls='--', color='k')
	plt.xlabel('$n_{2,0}$ value')
	plt.ylabel('Argument of the Lambert W function')
	import os
	fname = os.path.join('Images', 'Warg_n.png')
	plt.savefig(fname, dpi=300)
	plt.show()


def video():
	"""
	Save multiple plots which can then be converted to a video using ffmpeg.

	https://trac.ffmpeg.org/wiki/Create%20a%20video%20slideshow%20from%20images

	in ffmpeg bash type: 

	# On Linux
	ffmpeg -framerate 2 -i file%04d.png -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4

	# On Windows
	set PATH=%PATH%;C:\ffmpeg\bin
	ffmpeg -framerate 10 -i file%04d.png out.wmv

	## For 400 second film
	ffmpeg -t 400 -loop_input -i input.png output.wmv

	"""
	t = np.linspace(0,30,1000)
	tau = 10
	sigma_21 = 1
	n20 = 0.8  # Fraction of population inversion (>0.5)

	i = 0
	for alpha in np.linspace(0.1, 10, 500):
	    i +=1
	    n = lambertDecay(t, alpha,tau,sigma_21,n20)
	    
	    # Select only part of curve that is completely real
	#     index = np.where(np.isreal(n))
	#     t = t[index]
	#     n = n[index].real
	    
	    plt.figure()
	    plt.axhline(1/np.e, ls='--', color='k')
	    plt.ylabel('$n_2$/max($n_2$)')
	    plt.xlabel('time (ms)')
	    plt.xlim(min(t),max(t))
	    plt.yscale('log')
	    plt.plot(t,n/max(n), label=alpha)
	    plt.legend(loc='best', title='alpha')
	    plt.savefig('Images/file%04d.png' % i, dpi=300)
	    plt.close()


if __name__ == "__main__":
	differentAlphas()
	# Warg()
	# video()
