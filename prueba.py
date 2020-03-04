import numpy as np
data = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(data)
def girar(data):
    turn=True
    aux = np.empty((1, 3))
    for i in range(3):
        for j in range(3):
            if (j==2):
                if(turn==True):
                    for k in range(3):
                        aux[k]=data[i][k]
                    for l in range(3):
                        data[i][l]=aux[l]
                not(turn)
                print(turn)
girar(data)
print(data)