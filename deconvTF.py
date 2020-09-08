from flowdec import data as fd_data
from flowdec import restoration as fd_restoration
#from time import time
#from imageFunctions import mostrarImagen
#from imageFunctions import normalizar
#from imageFunctions import imgReadCv2
#import cv2
'''
to=time()

data = imgReadCv2('Images/310114 1hz-23_C003Z001.bmp',2)
kernel = imgReadCv2('PSF/PSF_BW6.bmp',2)
'''
# Run the deconvolution process and note that deconvolution initialization is best kept separate from 
# execution since the "initialize" operation corresponds to creating a TensorFlow graph, which is a 
# relatively expensive operation and should not be repeated across multiple executions
def deconvolveTF(img,kernel):
	algo = fd_restoration.RichardsonLucyDeconvolver(data.ndim).initialize()
	res = algo.run(fd_data.Acquisition(data=data, kernel=kernel), niter=10).data
'''
mostrarImagen("Convolve",data,False)
deconv = normalizar(res)
mostrarImagen("Deconvolucion",deconv,True)

#cv2.imwrite("Deconvolutions/Deconvolve_310114_1hzC3.bmp",outdeconv)

tf=time()
tt=tf-to
print("El tiempo total fue: ",tt/60, "minutos")
print("Ha terminado la ejecucion")
'''