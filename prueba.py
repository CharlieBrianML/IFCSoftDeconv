import numpy as np
data = np.array([[1,2,3],[4,5,6],[7,8,9]])
data2 = np.array([1,2,3,4,5,6,7,8,9])
data3 = np.empty(9)
print(data2)
		
def fase(numFase,data2):
    for j in range(numFase):
        for i in range(9):
            if(i==8):
                data3[i]=data2[0]
            else:
                data3[i]=data2[i+1]
        data2=data3
        print(data2)
    return data2
        
#girar(data)
data2=fase(2,data2)
print(data2)