import numpy as np
from PIL import Image
from .GaussianFit import FitGauss2D, Gaussian2D
import matplotlib.pyplot as plt

# User input
fname = 'day2.jpg'
px_per_um = 0.215
um_per_px = 1/px_per_um

# Open image in grayscale and store as a numpy array
im = Image.open(fname).convert('L')
data = np.array(im)

# Fit a 2D Gaussian to the image around the maximum value
p, success = FitGauss2D(data,ip=None)
amplitude, xcenter, ycenter, xsigma, ysigma, rot, bkg = p

# Plot image with fitted contours over the image
fig, ax = plt.subplots()
ax.imshow(data, origin="lower", cmap='Greys')
x, y = np.indices(np.shape(data),dtype=np.float)
fn = Gaussian2D(amplitude, xcenter, ycenter, xsigma, ysigma, rot, bkg)
ax.contour(fn(x, y), alpha=0.8)
plt.savefig('day2')
plt.show()

# Beam area of ellipse
print('Gaussian radius: x = {:.0f} um and y = {:.0f} um'.format(xsigma * um_per_px / 2, ysigma * um_per_px / 2))
area = xsigma * um_per_px * ysigma * um_per_px * np.pi * 1E-8 / 4  # [cm2]
print('Area = {:e} cm^2'.format(area))


def calc_F(E, area):
    """Convert pulse energy [J] and beam area [cm^2] to average fluence [J/cm2]"""
    return E / area


print('Fluence = {:.2f} J/cm2'.format(calc_F(107E-6, area)))
