#matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from scipy import ndimage, signal
from flowdec import data as fd_data
from flowdec import restoration as fd_restoration
from time import time
import cv2
from skimage.exposure import rescale_intensity


to=time()

data = cv2.imread('Images/310114 1hz-23_C003Z001.bmp',2)
data2 = cv2.imread('Images/310114_1hz-2_brightness_adjusted_FFM.bmp')
kernel = cv2.imread('PSF/PSF_BW6.bmp',2)

# Run the deconvolution process and note that deconvolution initialization is best kept separate from 
# execution since the "initialize" operation corresponds to creating a TensorFlow graph, which is a 
# relatively expensive operation and should not be repeated across multiple executions
algo = fd_restoration.RichardsonLucyDeconvolver(data.ndim).initialize()
res = algo.run(fd_data.Acquisition(data=data, kernel=kernel), niter=10).data

cv2.imshow("Convolve",data)
outdeconv = rescale_intensity(res, in_range=(0, 255))
outdeconv = (outdeconv * 255).astype("uint8")
cv2.imshow("Resultado", outdeconv)
cv2.imwrite("Deconvolutions/Deconvolve_310114_1hzC3.bmp",outdeconv)
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.show()

tf=time()
tt=tf-to
print("El tiempo total fue: ",tt/60, "minutos")
print("Ha terminado la ejecucion")
