import numpy as np
import SMO

file_name = "testSet.txt"
train_data, labels = SMO.getData(file_name)
data= SMO.start(train_data, labels, 1, 100)
data.w = SMO.calW(data)

print('parameter-w:  ',data.w)
print('parameter-b:  ',data.b)
# print('parameter-alpha:  ',data.alpha)

for i in range(data.m):
    if data.alpha[i]>0:
        data.sup_vec_index.append(i)
        print(data.x[i])

SMO.plotData(data)