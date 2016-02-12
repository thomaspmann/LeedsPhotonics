import numpy as np
import matplotlib.pyplot as plt

from scipy.special import lambertw
from numpy import exp, log, e

def lambertDecay(t, alpha, tau, sigma_12, sigma_21, n, n20):
	
	B = 1 + alpha*sigma_12*n
	A = (alpha*(sigma_12+sigma_21)) / B

	arg = -A*n20*exp(-(t/tau)-(A*n20))
	return -lambertw(arg).real/A


def differentAlphas():
	"""
	Fit the function with different Alpha values
	"""
	t = np.linspace(0,30,1000)

	tau = 10
	sigma_12 = 1
	sigma_21 = 1
	n = 1
	n20 = 0.9

	plt.figure()
	plt.axhline(1/np.e, ls='--', color='k')
	plt.ylabel('$n_2$/max($n_2$)')
	plt.xlabel('time (ms)')

	for alpha in [0.002, 0.1, 0.3, 1, 2, 5]:
		y = lambertDecay(t, alpha, tau, sigma_12, sigma_21, n, n20)

		# Select only part of curve that is completely real
	#     index = np.where(np.isreal(n))
	#     t = t[index]
	#     n = n[index].real

		plt.plot(t,y/max(y), label=alpha)

	plt.xlim(min(t),max(t))
	plt.yscale('log')
	plt.legend(loc='best', title='alpha')
	import os
	fname = os.path.join('Images', 'decaysVsAlphaGeneraln20_0p9.png')
	# plt.savefig(fname, dpi=300)
	plt.show()

def Warg():
	""" 
	Plot the argument of the lambert W function vs alpha or n20
	"""
	t = np.linspace(0,30,1000)

	tau = 10
	sigma_12 = 1
	sigma_21 = 1
	n20 = 0.9
	n = 1
	alpha = 1
	Warg = []
	a = np.linspace(0.1,10, 1000)
	for alpha in a:
		B = 1 + alpha*sigma_12*n
		A = (alpha*(sigma_12+sigma_21)) / B

		arg = -A*n20*exp(-(t/tau)-(A*n20))
		Warg.append(min(arg))
	plt.plot(a,Warg, '-')
	plt.axhline(-1/e, ls='--', color='k')
	plt.xlabel('$\\alpha$ value')
	plt.ylabel('Argument of the Lambert W function')
	import os
	fname = os.path.join('Images', 'WargGeneraln20_0p9.png')
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
	sigma_12 = 1
	sigma_21 = 1
	n = 1
	n20 = 0.9

	i = 0
	for alpha in np.linspace(0.1, 10, 10):
		i +=1
		y = lambertDecay(t, alpha, tau, sigma_12, sigma_21, n, n20)
		
		plt.figure()
		plt.axhline(1/np.e, ls='--', color='k')
		plt.ylabel('$n_2$/max($n_2$)')
		plt.xlabel('time (ms)')
		plt.xlim(min(t),max(t))
		plt.yscale('log')
		plt.plot(t,y/max(y), label=alpha)
		plt.legend(loc='best', title='alpha')
		plt.savefig('Images/file%04d.png' % i, dpi=300)
		plt.close()


if __name__ == "__main__":
	# differentAlphas()
	Warg()
	# video()
