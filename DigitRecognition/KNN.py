import numpy as np
from os import listdir


def img2vector(file_name):
    f = open(file_name)
    L = []
    for i in range(32):
        line = f.readline()
        L.extend([int(x) for x in line if x != '\n'])
    f.close()
    return np.array(L)

def classify(test_vec, training_matrix, labels, k):
    #扩充测试数据
    test_data = np.tile(test_vec,(training_matrix.shape[0], 1))
    #axis=1每一行计算和
    distance = (((test_data - training_matrix)**2).sum(axis=1))**0.5 
    dis_index = distance.argsort()
    dic = dict()
    maximum = -1
    for i in range(k):
        class_num = labels[dis_index[i]]
        dic[class_num] = dic.get(class_num,0) + 1
        if dic[class_num] > maximum:
            maximum = dic[class_num]
            ans = class_num
    return ans

def file_process(file):
    file_name_str = file
    file_name = file_name_str.split('.')[0]
    class_num = int(file_name.split('_')[0])
    return file_name_str, class_num
    
def training_data(file_path):
    training_data_list = listdir(file_path)
    training_data_len = len(training_data_list)
    training_matrix = np.zeros((training_data_len,1024))
    labels = []
    for i in range(training_data_len):
        file_name_str, class_num = file_process(training_data_list[i])
        training_matrix[i, :] = img2vector('trainingDigits\\' + file_name_str)
        labels.append(class_num)
    print('training data done!')
    return training_matrix, labels
    

