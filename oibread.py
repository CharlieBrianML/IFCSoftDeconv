from oiffile import *

def getMetadata(filename)
	with OifFile(filename) as oib:
		metadata = oib.mainfile
		metadata.update(dict(axes=oib.axes,shape=oib.shape,dtype=oib.dtype))
	return metadata