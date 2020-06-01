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


img = asarray(Image.open("INITIAL_x0y0.png").convert("L"))
i=350
j=170
s=50
area = [Rectangle( (j-s, i-s), 2*s, 2*s)]
gca().add_collection(PatchCollection(area, facecolor='none', edgecolor='r'))

area = img[i-s:i+s, j-s:j+s]
i-=10
j-=15
img = img[i-s:i+s, j-s:j+s]

imshow(img, cmap="gray")
#savefig("normalized_initial.png", bbox_inches = 'tight', pad_inches = 0, dpi=100)
show()



fig = figure(figsize=(1,1))
imshow(area, cmap="gray")
#fig.savefig("normalized_pattern.png", bbox_inches = 'tight', pad_inches = 0)
show()

fig = plt.figure()
ax = fig.gca()
print(img.shape)
print(area.shape)
img = img.astype('uint64')
area = area.astype('uint64')
cor = correlate(img, area, mode='full')
im = imshow(cor, extent = [-2*s , cor.shape[1]-2*s, cor.shape[0]-2*s , -2*s], cmap="gray")
fig.colorbar(im, shrink=0.5, aspect=5, orientation="horizontal")
#savefig("normalized_simple_corr.png", bbox_inches = 'tight', pad_inches = 0)
plt.show()


img = (img - mean(img))/(img.std()*img.shape[0]*img.shape[1])
area = (area - mean(area))/(img.std()*area.shape[0]*area.shape[1])
cor = correlate(img, area, mode='full')

print(np.unravel_index(np.argmax(cor, axis=None), cor.shape))

fig = plt.figure()
im = imshow(cor, extent = [-2*s , cor.shape[1], cor.shape[0], -2*s], cmap="gray")

# Customize the z axis.
#ax.set_zlim(-1.01, 1.01)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(im, shrink=0.5, aspect=5, orientation="horizontal")
#savefig("normalized_corr.png", bbox_inches = 'tight', pad_inches = 0)
plt.show()