import numpy as np
data = np.array([[1,2,3],[4,5,6],[7,8,9]])
data2 = np.array([1,2,3,4,5,6,7,8,9])
print(data2)

def girar(data):
	turn=False
	aux = np.empty((3))
	for i in range(3):
		if(turn==True):
			for k in range(3):
				aux[k]=data[i][k]
			for l in range(3):
				data[i][l]=aux[-(l+1)]
		turn=not(turn)
		
"""def fase(numFase,data2):
	dataAux = np.empty(9)
	for j in range(numFase):
		aux=data2[0]
		for i in range(9):
			if(i==8):
				dataAux[i]=aux
			else:
				dataAux[i]=data2[i+1]
		data2=dataAux
	return data2"""
	
def faseInv(numFase,data2):
	dataAux = np.empty(9)
	for j in range(numFase):
		print("Iniciamos: ",data2)
		aux=data2[5]
		print(aux)
		for i in range(9):
			print("Dentro de for: ",data2)
			aux2=data2[i]
			print("aux2: ",aux2)
			if(i==0):
				dataAux[0]=data2[8]
				print("dataAux:----------- ",dataAux[0])
			else:
				print("data2[",i,"]= ",data2[i])
				dataAux[i]=data2[i-1]
				print("dataAux:----------- ",dataAux[i])
		print("dataAux: ",dataAux)
		data2=dataAux
		print("Num data: ",data2)
	return data2
	
#data2=fase(7,data2)
data2Inv=faseInv(2,data2)
#print(data2)
print(data2Inv)