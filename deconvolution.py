from time import time
from imageFunctions import mostrarImagen
from imageFunctions import normalizar
from imageFunctions import imgReadCv2
from deconvTF import deconvolveTF

to=time()

img = imgReadCv2('Images/310114 1hz-23_C003Z001.bmp',2)
psf = imgReadCv2('PSF/PSF_BW6.bmp',2)

deconv = deconvolveTF(img, psf)

mostrarImagen("Convolve",data,False)
deconvN = normalizar(deconv)
mostrarImagen("Deconvolucion",deconvN,True)

#cv2.imwrite("Deconvolutions/Deconvolve_310114_1hzC3.bmp",outdeconv)

tf=time()
tt=tf-to
print("El tiempo total fue: ",tt/60, "minutos")
print("Ha terminado la ejecucion")
