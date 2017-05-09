import wgm.core as wgm
import numpy as np
import matplotlib.pyplot as plt


def plot_beta_fiber():
    rho_list = np.linspace(1.4, 2.8)
    beta_list = []
    for rho in rho_list:
        beta_list.append(wgm.beta_fiber(rho, lam0=1.55, N=1.44))

    fig, ax = plt.subplots()
    ax.plot(rho_list, beta_list)
    ax.set_xlabel('Fiber radius (um)')
    ax.set_ylabel('Propagation constant, beta (/um)')
    ax.set_ylim([5.4, 6])
    plt.show()


def test1():
    """Recreate the results in table 1 or 2 in [2]."""
    # Initialise wgm object (na and ns irrelevant as we will override the calculated m anyway)
    obj = wgm.WGM(na=1, ns=1, mode='TE')

    # Overide m_ratio to the value used in the tables
    obj.m_ratio = 1.363

    # Calculate and print the results
    m = 100  # Mode number
    nmax = 5  # Maximum order l to calculate up to
    print('Order l\tMode Number n={0:d}'.format(m))
    for n in range(1, nmax+1):
        result = obj.calc_size_param(n=n, m=m)
        print('{}\t\t{}'.format(n, result))


def test2():
    """Recreate Fig.1 in [1]"""
    n_sio2 = 1.44

    fig, ax2 = plt.subplots()
    ax1 = ax2.twiny()

    # Calculate the fiber
    rho_list = np.linspace(1.4, 2.5)
    beta_list = []
    for rho in rho_list:
        beta_list.append(wgm.beta_fiber(rho, lam0=1.55, N=n_sio2))

    ax1.plot(rho_list, beta_list, color='k')
    ax1.set_xlabel('Fiber radius (um)')
    ax1.set_ylabel('Propagation constant, beta (/um)')
    ax1.set_ylim([5.4, 6])

    # Calculate the sphere WGM
    na = 1       # Refractive index of medium surrounding sphere
    ns = n_sio2  # Refractive index of the sphere
    mode = 'TM'  # Mode of WGM ('TE' or 'TM)
    obj = wgm.WGM(na=na, ns=ns, mode=mode)  # Initialise the WGM object
    lam0 = 1.55  # Wavelength of interest (um)
    a_list = [50, 75, 100, 150, 200, 250]  # Radius of the spheres to calculate (um)

    nmax = 4  # Maximum order n to calculate up to
    for n in range(1, nmax+1):
        beta_list = []
        for a in a_list:
            l = 2 * a * np.pi  # Approximately equal to the sphere radius (2 x pi x r)
            m = l  # Interested in modes at the equator

            x = obj.calc_size_param(n=n, m=m)
            beta = wgm.beta_wgm(lam0=lam0, l=l, x=x)
            beta_list.append(beta)

        ax2.plot(a_list, beta_list, '.', label=n)
    ax2.set_xlabel('Sphere radius (um)')
    ax2.set_xlim([0, 300])
    ax2.legend(title='n')
    plt.show()


if __name__ == "__main__":
    # plot_beta_fiber()
    # test1()
    test2()

