import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.special import lambertw
from numpy import exp, e


def lambertDecay(t, tau, sigma_12, sigma_21, n, n20, r, rho, d):

    # alpha = r/2
    alpha = (1+r/2)
    A = alpha*rho*d*(sigma_12+sigma_21)
    B = 1 + alpha*rho*d*sigma_12*n

    x = (t/(B*tau)) + (A*n20)
    arg = -A*n20*exp(-x)/B

    # Check that result is real
    assert min(arg) >= -1/e, \
        'Lambert W Argument will give an imaginary result.'

    return -B*lambertw(arg).real/A


def decayTime(t, tau, sigma_12, sigma_21, n, n20, r, rho, d):
    from scipy.optimize import curve_fit

    def model_func(t, a, b, c):
        return a*np.exp(-t/b)+c

    y = lambertDecay(t, tau, sigma_12, sigma_21, n, n20, r, rho, d)

    guess = [max(y)-min(y), tau, 0]  # Guess for a, b, c coefficients
    popt, pcov = curve_fit(model_func, t, y, guess)   # Fit using Levenberg-Marquardt algorithm
    decay = popt[1]         # Lifetime (ms)

    return decay


def varyFeedback(t, tau, sigma_12, sigma_21, n, n20, rho, d):
    """
    Fit the function with different r values
    """

    plt.figure()
    plt.axhline(1/e, ls='--', color='k')
    plt.ylabel('$n_2$/max($n_2$)')
    plt.xlabel('time (ms)')

    for r in [0.001, 0.5, 1]:
        y = lambertDecay(t, tau, sigma_12, sigma_21, n, n20, r, rho, d)
        # y = y/max(y)
        plt.plot(t, y, label=r)

    plt.title('$\\tau$ %.1f, $\\sigma_{12}$ %.1f, $\\sigma_{21}$ %.1f, n %.1f, d %.4f cm, $\\rho$ %.2f'
              % (tau, sigma_12, sigma_21, n, d, rho))
    plt.xlim(min(t), 25)
    # plt.ylim(0.01, 1)
    # plt.yscale('log')
    plt.legend(loc='best', title='r')
    plt.savefig('Images/varyFeedbackLog.png', dpi=900)
    plt.show()


def video(t, tau, sigma_12, sigma_21, n, n20, r, rho, d):
    """
    Save multiple plots which can then be converted to a video using ffmpeg.

    https://trac.ffmpeg.org/wiki/Create%20a%20video%20slideshow%20from%20images

    in ffmpeg bash type:

    # On Linux
    ffmpeg -framerate 2 -i file%04d.png -c:v libx264 -r 30 -pix_fmt yuv420p
     x.mp4

    # On Windows
    set PATH=%PATH%;C:\ffmpeg\bin
    ffmpeg -framerate 10 -i file%04d.png out.wmv

    ## For 400 second film
    ffmpeg -t 400 -loop_input -i input.png output.wmv
    """

    i = 0
    for r in tqdm(np.linspace(1E-4, 1, 200)):
        i += 1
        y = lambertDecay(t, tau, sigma_12, sigma_21, n, n20, r, rho, d)

        plt.figure()
        plt.axhline(1/np.e, ls='--', color='k')
        plt.ylabel('$n_2$/max($n_2$)')
        plt.xlabel('time (ms)')
        plt.xlim(min(t), 50)
        plt.ylim(1E-2, 1)
        plt.yscale('log')
        plt.plot(t, y/max(y), label=r)
        plt.title('tau %.1f, sigma_12 %.1f, sigma_21 %.1f, n %.1f, n20 %.1f' %
                  (tau, sigma_12, sigma_21, n, n20))
        plt.title('$\\tau$ %.1f, $\\sigma_{12}$ %.1f, $\\sigma_{21}$ %.1f, n %.1f, $n_{20}$ %.1f, d %.1f cm, '
                  '$\\rho$ %.2f' % (tau, sigma_12, sigma_21, n, n20, d, rho))
        plt.legend(loc='lower left', title='r')
        plt.savefig('Images/file%04d.png' % i, dpi=300)
        plt.close()


def threedPlot(t, tau, sigma_12, sigma_21, n, n20, r, rho, d):
    # from matplotlib import cm

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = np.linspace(0.01, 8, 200)
    y = np.linspace(0.1, 1, 200)
    X, Y = np.meshgrid(x, y)

    zs = np.array([decayTime(t, r, tau, sigma_12, sigma_21, n, n20)
                   for r, n20 in zip(np.ravel(X), np.ravel(Y))])

    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z)
    # cset = ax.contour(X, Y, Z, zdir='z', offset=-0, cmap=cm.coolwarm)
    # cset = ax.contour(X, Y, Z, zdir='x', offset=-8, cmap=cm.coolwarm)
    # cset = ax.contour(X, Y, Z, zdir='y', offset=1, cmap=cm.coolwarm)

    ax.set_xlabel('r')
    ax.set_ylabel('$n_{2,0}$')
    ax.set_zlabel('Decay Time')

    plt.show()


def contourPlot(t, tau, sigma_12, sigma_21, n, rho, d):
    plt.figure()
    x = np.linspace(1E-4, 1, 200)     # r
    y = np.linspace(1E-4, n, 200)     # n20
    X, Y = np.meshgrid(x, y)

    zs = np.array([decayTime(t, tau, sigma_12, sigma_21, n, n20, r, rho, d)
                   for r, n20 in zip(np.ravel(X), np.ravel(Y))])

    Z = zs.reshape(X.shape)

    origin = 'lower'
    # origin = 'upper'
    lv = 40  # Levels of colours
    CS = plt.contourf(X, Y, Z, lv,
                      # levels=np.arange(0, 100, 5),
                      cmap=plt.cm.plasma,
                      origin=origin)

    # CS2 = plt.contour(CS,
    #                   levels=CS.levels[::4],
    #                   colors='k',
    #                   origin=origin,
    #                   hold='on')

    plt.xlabel('r')
    plt.ylabel('$n_{2,0}/n$')
    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = plt.colorbar(CS)
    cbar.ax.set_ylabel('Measured monoexponential lifetime')
    # Add the contour line levels to the colorbar
    # cbar.add_lines(CS2)

    plt.title('$\\tau$ %.1f, $\\sigma_{12}$ %.1f, $\\sigma_{21}$ %.1f, n %.1f, d %.1f cm, $\\rho$ %.2f'
              % (tau, sigma_12, sigma_21, n, d, rho))
    plt.savefig('Images/contourfplot.png', dpi=900)

    plt.show()


if __name__ == "__main__":

    # Time to simulate over
    t = np.linspace(0, 100, 1000)

    # Define material parameters:
    rho = 0.217         # Density of Er ions (*1E21 cm^-3)
    tau = 12.54            # Radiative decay rate
    d = 0.98E-4*1          # Thickness of slab (cm)
    sigma_12 = 4.98     # Absorption cross-section (*1E-21 cm^2)
    sigma_21 = 5.02     # Emission cross-section (*1E-21 cm^2)
    n = 1               # Total number of active ions (i.e.,clustering)

    contourPlot(t, tau, sigma_12, sigma_21, n, rho, d)
    #
    # n20 = 0.2*n         # Fraction of excited ions at t=0
    # r = 0.5             # Reflectivity of top layer
    # varyFeedback(t, tau, sigma_12, sigma_21, n, n20, rho, d)
    # video(t, tau, sigma_12, sigma_21, n, n20, r, rho, d)
    # threedPlot(t, tau, sigma_12, sigma_21, n, n20, r, rho, d)
