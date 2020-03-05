import numpy as np
data = np.array([[1,2,3],[4,5,6],[7,8,9]])
data2 = np.array([1,2,3,4,5,6,7,8,9])
data3 = np.empty(9)
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
		
def fase(numFase,data2):
    for j in range(numFase):
        for i in range(9):
            if(i==8):
                data3[i]=data2[0]
            else:
                data3[i]=data2[i+1]
        data2=data3
        print(data3)
    return data2
        
#girar(data)
data2=fase(2,data2)
print(data2)