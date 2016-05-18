#!/usr/bin/python
# By Samuel George; samuel@ras.ucalgary.ca #original 09/07/2010
# usage: rect.py #produces: bunch of png plots and an annimated gif.
import warnings

warnings.simplefilter("ignore", DeprecationWarning)
import matplotlib

matplotlib.use('Agg')  # avoid X11 issues.
import string, sys, math, scipy, numpy, pyfits, pylab, os, asciidata
from scipy.io import write_array
from scipy.io import read_array
from numpy import *
from pylab import *
from scipy import *
from scipy import integrate

# rc('text', usetex=True)
rc('axes', linewidth=2)
##tickmarks
majorLocator = MultipleLocator(20)
majorFormatter = FormatStrFormatter('%d')
minorLocator = MultipleLocator(2)
majorLocatory = MultipleLocator(2)
majorFormattery = FormatStrFormatter('%d')
minorLocatory = MultipleLocator(.5)

maxval = 100.0
foo = array(arange(0, maxval, 1.0))
qq = zeros(len(foo), float);
qq2 = zeros(len(foo), float)
a = 10
ii = 0
while ii < len(foo):
    if ii > 50 - a / 2 and ii < 50 + a / 2:
        qq[ii] = 1.0
    else:
        qq[ii] = 0.0
    ii += 1

maxlen = maxval / 2
out = zeros(maxlen * 2, float)
outkk = zeros(maxlen * 2, float)
print(len(out), len(foo))
kk = -maxlen
zz = 0
while kk < maxlen:
    ii = 0
    while ii < len(foo):
        if ii > (50 + kk) - a / 2 and ii < (50 + kk) + a / 2:
            qq2[ii] = 1.0  # /2.0
        else:
            qq2[ii] = 0.0

        ii += 1

    out[zz] = scipy.integrate.simps(qq2 * qq, dx=1)
    outkk[zz] = zz  # print out[zz], zz

    fig = plt.figure(figsize=(5, 2))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=0.97, wspace=None, hspace=0.25)
    ax1 = fig.add_subplot(111)
    ax1.plot(foo, qq, 'r-')
    ax1.plot(foo, qq2, 'b-')
    ax1.plot(outkk[:zz], out[:zz], 'g--')
    ax1.annotate('a = 9', xy=(80, 8), xytext=(80, 8))
    setp(gca().get_ymajorticklabels(), fontsize='large')
    setp(gca().get_xmajorticklabels(), fontsize='large')
    labels = setp(gca(), ylabel="h", xlabel="x")
    setp(labels, fontsize='large')
    ax1.xaxis.set_major_locator(majorLocator)
    ax1.xaxis.set_major_formatter(majorFormatter)
    ax1.xaxis.set_minor_locator(minorLocator)
    ax1.yaxis.set_major_locator(majorLocatory)
    ax1.yaxis.set_major_formatter(majorFormattery)
    ax1.yaxis.set_minor_locator(minorLocatory)
    for l in ax1.get_xticklines() + ax1.get_yticklines():
        l.set_markersize(6)
        l.set_markeredgewidth(1.2)
for l in ax1.yaxis.get_minorticklines() + ax1.xaxis.get_minorticklines():
    l.set_markersize(3)
    l.set_markeredgewidth(1.2)
ylim(-1, 10)
if zz < 10:
    pylab.savefig("rect0" + str(zz) + ".png")  # deal with issue of filenames in wrong order with convert.
else:
    pylab.savefig("rect" + str(zz) + ".png")
kk += 1
zz += 1

os.system("convert -delay 5 -loop 0 rect*.png rect_conv.gif")
