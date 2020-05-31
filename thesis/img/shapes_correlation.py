'''
======================
3D surface (color map)
======================

Demonstrates plotting a 3D surface colored with the coolwarm color map.
The surface is made opaque by using antialiased=False.

Also demonstrates using the LinearLocator and custom formatting for the
z axis tick labels.
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from pylab import *
from PIL import Image
from matplotlib.patches import Rectangle, Ellipse
from matplotlib.collections import PatchCollection
from scipy.signal import correlate

img = asarray(Image.open("shapes_in.png").convert('L'))
print(img.shape)
i=350
j=170
s=50
area = [Rectangle( (j-s, i-s), 2*s, 2*s)]
#gca().add_collection(PatchCollection(area, facecolor='none', edgecolor='r'))
imshow(img)
show()

print(img)

area = img[i-s:i+s, j-s:j+s]

imshow(area)
show()

fig = plt.figure()
ax = fig.gca()
print(img.shape)
print(area.shape)
img = img.astype('uint64')
area = area.astype('uint64')
#img = img - mean(img)
#area = area - mean(area)
cor = correlate(img, area, mode='full')
print(type(img))
print(type(cor))
print(area)
print(cor)


X = np.arange(cor.shape[1]) - s
Y = np.arange(cor.shape[0]) - s
#X, Y = np.meshgrid(X, Y)

# Plot the surface.
#surf = ax.plot_surface(X, Y, cor, cmap=cm.coolwarm, linewidth=0, antialiased=False)
imshow(cor, extent = [-2*s , cor.shape[1]+2*s, cor.shape[0]+2*s , -2*s])

# Customize the z axis.
#ax.set_zlim(-1.01, 1.01)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
