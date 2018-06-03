import pandas as pd
import numpy as np
from numpy import pi
from uncertainties import ufloat
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


class D2LnF:
    def __init__(self, df):
        if 'anomaly' not in df:
            df['anomaly'] = 1
        df = self.prepare_df(df)
        self.data = df

    def prepare_df(self, df):
        # Add error to pulse energy measurement
        df['Pulse Energy (uJ)'] = [ufloat(i, 0.5) for i in df['Pulse Energy (uJ)']]
        # Calculate diameter squared from the max an minimum diameters measured
        df['r_m'] = (df['D_max'] + df['D_min']) / 4
        D = 2 * df['r_m']
        stdD = abs(df['D_max'] - df['D_min']) / 2
        df['Diameter^2'] = [ufloat(i, j) ** 2 for i, j in zip(D, stdD)]

        w = df.groupby('N').apply(calc_w)
        for key in w.keys():
            row_indexer = df.index[df['N'] == key].tolist()
            df.loc[row_indexer, 'w'] = w[key].nominal_value
        # Calculate average fluence assuming a fixed beam waist, plot and print F_th
        df['Fluence (J/cm2)'] = np.vectorize(F_avg)(df['Pulse Energy (uJ)'] * 1E-6, df['w'] * 1E-4)
        return df

    def calc_e_th(self, N, popt=False):
        bool = popt
        data = self.data.loc[self.data['anomaly'] == 1]
        data = data.loc[data['N'] == N]
        # Polynomial fit to D2 vs Ln(E) and extrapolate to D2=0 to find the threshold
        x, xerr = separate_nom_and_std(data['Pulse Energy (uJ)'])
        y, yerr = separate_nom_and_std(data['Diameter^2'])

        if len(x) > 4:
            popt, cov = np.polyfit(np.log(x), y, w=1 / yerr, deg=1, cov=True)
        else:
            popt = np.polyfit(np.log(x), y, w=1 / yerr, deg=1, cov=False)
            cov = [[0, 0], [0, 0]]
        m = ufloat(popt[0], np.sqrt(np.diag(cov))[0])
        c = ufloat(popt[1], np.sqrt(np.diag(cov))[1])
        E_th = np.e ** (-c / m)
        if popt:
            return E_th, popt
        else:
            return E_th

    def plot_e_th(self, N_list=None):
        fig, ax = plt.subplots()

        if N_list is None:
            N_list = self.data.N.unique()
        for N in N_list:
            data = self.data[self.data.N == N]
            x, xerr = separate_nom_and_std(data['Pulse Energy (uJ)'])
            y, yerr = separate_nom_and_std(data['Diameter^2'])
            res = ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='.', label=N)

            E_th, popt = self.calc_e_th(N, popt=True)
            xn = np.linspace(E_th.nominal_value, max(x))
            p = np.poly1d(popt)
            ax.plot(xn, p(np.log(xn)), color=res[0].get_color())
        ax.set_ylim([0, ax.get_ylim()[1]])
        ax.set_xscale('log')
        ax.set_ylabel('$D^2$ $[\mu m^2]$')
        ax.set_xlabel('Energy [uJ]')
        plt.legend()
        plt.show()
        return fig

    def calc_f_th(self, N, popt=False):
        bool = popt
        data = self.data.loc[self.data['anomaly'] == 1]
        data = data.loc[data['N'] == N]

        # Polynomial fit to D2 vs Ln(E) and extrapolate to D2=0 to find the threshold
        x, xerr = separate_nom_and_std(data['Fluence (J/cm2)'])
        y, yerr = separate_nom_and_std(data['Diameter^2'])

        if len(x) > 4:
            popt, cov = np.polyfit(np.log(x), y, w=1 / yerr, deg=1, cov=True)
        else:
            popt = np.polyfit(np.log(x), y, w=1 / yerr, deg=1, cov=False)
            cov = [[0, 0], [0, 0]]
        m = ufloat(popt[0], np.sqrt(np.diag(cov))[0])
        c = ufloat(popt[1], np.sqrt(np.diag(cov))[1])
        F_th = np.e ** (-c / m)
        if bool:
            return F_th, popt
        else:
            return F_th

    def plot_f_th(self, N_list=None):
        fig, ax = plt.subplots()

        if N_list is None:
            N_list = self.data.N.unique()
        for N in N_list:
            data = self.data[self.data.N == N]
            x, xerr = separate_nom_and_std(data['Fluence (J/cm2)'])
            y, yerr = separate_nom_and_std(data['Diameter^2'])
            res = ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='.', label=N)

            F_th, popt = self.calc_f_th(N, popt=True)
            xn = np.linspace(F_th.nominal_value, max(x))
            p = np.poly1d(popt)
            ax.plot(xn, p(np.log(xn)), color=res[0].get_color())
        ax.set_ylim([0, ax.get_ylim()[1]])
        ax.set_xscale('log')
        ax.set_xlabel('Fluence [J/cm$^2$]')
        ax.set_ylabel('$D^2$ $[\mu m^2]$')
        plt.legend()
        plt.show()
        return fig

    def incubation(self, model):
        data = self.data.loc[self.data['anomaly'] == 1]

        N_list = data.N.unique()
        Fth_list = []
        for N in N_list:
            Fth_list.append(self.calc_f_th(N))

        x = N_list
        y, yerr = separate_nom_and_std(Fth_list)
        fig, ax = plt.subplots()
        res = ax.errorbar(x, y, yerr=yerr, fmt='o', markerfacecolor='none', ls='')

        if model == 'glass':
            def fitfunc(x, f1, finf, k):
                return finf + (f1 - finf) * np.exp(-k * (x - 1))

        p0 = [max(y), min(y), 0.2]  # Initial guess for the parameters
        popt, pcov = curve_fit(fitfunc, x, y, p0=p0)
        perr = np.sqrt(np.diag(pcov))
        xn = np.linspace(min(x), max(x), num=10000)
        ax.plot(xn, fitfunc(xn, *popt), color=res[0].get_color())

        ax.set_xscale('log')
        ax.set_ylabel('$\mathrm{F_{th}}$ [J/cm$^2$]')
        ax.set_xlabel('$N$')
        return {'F_th(1)': ufloat(popt[0], perr[0]),
                'F_th(inf)': ufloat(popt[1], perr[1]),
                'k': ufloat(popt[2], perr[2])}, fig


class AFM(D2LnF):
    def prepare_df(self, df):
        # Add error to pulse energy measurement
        df['Pulse Energy (uJ)'] = [ufloat(i, 0.5) for i in df['Pulse Energy (uJ)']]
        df['Diameter^2'] = pow(2 * df['r_m'], 2)
        df['Diameter^2'] = [ufloat(i, 0) for i in df['Diameter^2']]

        # w = df.groupby('N').apply(calc_w)
        # for key in w.keys():
        #     row_indexer = df.index[df['N'] == key].tolist()
        #     df.loc[row_indexer, 'w'] = w[key].nominal_value
        # Calculate average fluence assuming a fixed beam waist, plot and print F_th
        df['w'] = 13.9
        df['Fluence (J/cm2)'] = np.vectorize(F_avg)(df['Pulse Energy (uJ)'] * 1E-6, df['w'] * 1E-4)
        return df

    def height(self, N=1):
        data = self.data.loc[self.data['anomaly'] == 1]
        data = data.loc[data['N'] == N]

        x, xerr = separate_nom_and_std(data['Fluence (J/cm2)'])
        y = data['Z_min']

        # Polynomial fit to evaluate the absorption coefficient
        popt, cov = np.polyfit(np.log(x), y, deg=1, cov=True)
        p = np.poly1d(popt)
        m = ufloat(popt[0], np.sqrt(np.diag(cov))[0])
        c = ufloat(popt[1], np.sqrt(np.diag(cov))[1])
        F_th = np.e ** (-c / m)
        alpha = 1 / m
        print('F_th = {:.2g}, alpha = {:.2g}, 1/alpha = {:.2g}'.format(F_th, alpha * 1E7, 1 / alpha))

        fig, ax = plt.subplots()
        res = ax.errorbar(x, y, fmt='.')
        xn = np.linspace(F_th.nominal_value, max(x), num=1000)
        ax.plot(xn, p(np.log(xn)), color=res[0].get_color())
        ax.set_ylim([0, ax.get_ylim()[1]])
        ax.set_xscale('log')
        ax.set_ylabel('$h_a$ [nm/pulse]')
        ax.set_xlabel('Laser Fluence, $F_0^{avg}$ [J/cm$^2$]')
        return fig

    def volume(self, N_list=None):
        data = self.data.loc[self.data['anomaly'] == 1]
        fig, ax = plt.subplots()

        if N_list is None:
            N_list = data.N.unique()
        for N in N_list:
            data = data.loc[data['N'] == N]
            x, xerr = separate_nom_and_std(data['Fluence (J/cm2)'])
            y = data['V_0']
            res = ax.plot(x, y, '.', label=N)
            xn = np.linspace(min(x), max(x))
            popt = np.polyfit(x, y, deg=1)
            p = np.poly1d(popt)
            ax.plot(xn, p(np.log(xn)), color=res[0].get_color())
        ax.set_ylim([0, ax.get_ylim()[1]])
        ax.set_xscale('log')
        ax.set_xlabel('Fluence [J/cm$^2$]')
        ax.set_ylabel('$D^2$ $[\mu m^2]$')
        plt.legend()
        plt.show()
        return fig

# Helper functions
def F_peak(E, w):
    """Convert pulse energy [J] and beam waist [cm] to peak fluence [J/cm2]"""
    return (2*E)/(pi * pow(w,2))


def F_avg(E, w):
    """Convert pulse energy [J] and beam waist [cm] to average fluence [J/cm2]"""
    return (E)/(pi * pow(w,2))


def Fpeak_to_E(Fpeak, w):
    """Convert peak fluence to pulse energy"""
    return Fpeak * pi * pow(w,2) / 2


def Favg_to_E(Favg, w):
    """Convert peak fluence to pulse energy"""
    return Favg * pi * pow(w,2)


def separate_nom_and_std(data):
    x = np.array([i.nominal_value for i in data])
    xerr = np.array([i.std_dev for i in data])
    return x, xerr


def calc_w(df):
    """
    Calculate beam waist from the gradient of linear regression to D2-LnE
    :param : i index to restrict the data thre regression is fitted to, e.g. slice(0, len(x))
    """
    data = df.loc[df['anomaly'] == 1]

    # Calculate beam waist in um from D2 vs energy linear fit
    x, xerr = separate_nom_and_std(data['Pulse Energy (uJ)'])
    y, yerr = separate_nom_and_std(data['Diameter^2'])

    # Polynomial fit to D2 vs Ln(E) to find the beam waist
    if len(x) > 4:
        popt, cov = np.polyfit(np.log(x), y, w=1 / yerr, deg=1, cov=True)
    else:
        popt = np.polyfit(np.log(x), y, w=1 / yerr, deg=1, cov=False)
        cov = [[0, 0], [0, 0]]
    m = ufloat(popt[0], np.sqrt(np.diag(cov))[0])
    c = ufloat(popt[1], np.sqrt(np.diag(cov))[1])
    w = pow(m / 2, 1 / 2)
    return w


# Update plot parameters for publication
def journalplotting():
    # Set figure size
    WIDTH = 325.215  # the number (in pt) latex spits out when typing: \the\linewidth
    FACTOR = .75  # the fraction of the width you'd like the figure to occupy
    fig_width_pt = WIDTH * FACTOR

    inches_per_pt = 1.0 / 72.27
    golden_ratio = (np.sqrt(5) - 1.0) / 2.0  # because it looks good

    fig_width_in = fig_width_pt * inches_per_pt  # figure width in inches
    fig_height_in = fig_width_in * golden_ratio  # figure height in inches
    fig_dims = [fig_width_in, fig_height_in]  # fig dims as a list

    # Update rcParams for figure size
    params = {
        'font.size': 11.0,
        'text.usetex': True,
        'font.family': 'times',  # serif
        #         'font.serif': 'cm',
        'savefig.dpi': 600,
        'savefig.format': 'eps',
        'savefig.bbox': 'tight',
        'figure.figsize': fig_dims,
    }
    plt.rcParams.update(params)
