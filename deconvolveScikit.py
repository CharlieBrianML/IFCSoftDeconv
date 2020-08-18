import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2
from skimage import color, data, restoration
import cv2
from skimage.exposure import rescale_intensity

data = cv2.imread('Images/verdeactrojomiosinaazulsinapto20Hz1.png')
psf = cv2.imread('PSF/PSF_BW4.png')
print("Imagen: ",data.shape)
print("Psf: ",psf.shape)
'''
astro = color.rgb2gray(data.astronaut())

#psf = np.ones((5, 5)) / 25
astro = conv2(astro, psf, 'same')
# Add Noise to Image
astro_noisy = astro.copy()
astro_noisy += (np.random.poisson(lam=25, size=astro.shape) - 10) / 255.
'''
# Restore Image using Richardson-Lucy algorithm
deconvolved_RL = restoration.richardson_lucy(data[:,:,0], psf[:,:,0], iterations=30)
print("deconvolved_RLdim: ",deconvolved_RL.shape,'\n',deconvolved_RL)
'''
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 5))
plt.gray()

for a in (ax[0], ax[1], ax[2]):
       a.axis('off')

ax[0].imshow(data)
ax[0].set_title('Original Data')

ax[1].imshow(data)
ax[1].set_title('Noisy data')

ax[2].imshow(deconvolved_RL[:,:,0], vmin=data.min(), vmax=data.max())
ax[2].imshow(deconvolved_RL,cmap='Spectral_r')
ax[2].set_title('Restoration using\nRichardson-Lucy')


fig.subplots_adjust(wspace=0.02, hspace=0.2,top=0.9, bottom=0.05, left=0, right=1)
plt.show()
'''
#outdeconv = rescale_intensity(deconvolved_RL, in_range=(0, 255))
#outdeconv = (outdeconv * 255).astype("uint8")
cv2.imshow("Resultado", deconvolved_RL)
#cv2.imshow("Resultado", deconvolved_RL[:,:,200])
cv2.waitKey(0)
cv2.destroyAllWindows()