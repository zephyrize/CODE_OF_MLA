import numpy as np
from os import listdir
from skimage import io
from math import floor,ceil

N = 32

# 读取矩阵文件转为向量
def img2vector(file_name):
    f = open(file_name)
    L = []
    for i in range(32):
        line = f.readline()
        L.extend([int(x) for x in line if x != '\n'])
    f.close()
    return np.array(L)

# KNN核心算法
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
    
# “训练”数据
def training_data(file_path):
    training_data_list = listdir(file_path)
    training_data_len = len(training_data_list)
    training_matrix = np.zeros((training_data_len,1024))
    labels = []
    for i in range(training_data_len):
        file_name_str, class_num = file_process(training_data_list[i])
        training_matrix[i, :] = image_process('trainingDigits\\' + file_name_str)
        labels.append(class_num)
    print('training data done!')
    return training_matrix, labels
    

# “图像的简单处理。包括切割和缩放”
def image_process(file_path):
    # print(file_path)
    f = open(file_path)
    L = []
    for i in range(32):
        line = list(f.readline())
        line = [int(x) for x in line if x != '\n']
        L.append(line)
    f.close()
    img = np.array(L)
    img = cut_picture(img)
    img = strech_picture(img)
    return img.reshape(N*N)

# 图片的切割(思想是将图像矩阵的 所有值为0的行列删掉)
def cut_picture(img):
    # print("function cut_picture:")
    size = []
    length = len(img)
    # print(length)
    width = len(img[0,:])
    # print(width)
    for i in range(length):
        if np.any(img[i]==1) == True:
            a = i
            break
    
    for i in range(length):
        if np.any(img[length-i-1]==1) == True:
            b = length-i-1
            break
    for i in range(width):
        if np.any(img[:,i]==1) == True:
            c = i
            break
    for i in range(width):
        if np.any(img[:,width-i-1]==1) == True:
            d = width-i-1
            break
    # print(a,b,c,d)
    return img[a:b+1,c:d+1]
    
# 最近邻插值算法实现图像的缩放
def strech_picture(img):
    # print("function strech_picture:")
    
    scale1 = len(img)/N
    scale2 = len(img[0])/N
    newImg = np.zeros((N, N),dtype=int)
    for i in range(N):
        for j in range(N):
            newImg[i,j] = img[int(np.floor(i*scale1)), int(np.floor(j*scale2))]
    return newImg


# 双线性插值算法实现图像的缩放
def strech_picture1(img):
    src_h, src_w = img.shape
    dst_img = np.zeros((N,N), dtype=int)
    scale_x, scale_y = float(src_w) / N, float(src_h) / N
    for dst_y in range(N):
        for dst_x in range(N):
            src_x = (dst_x + 0.5) * scale_x - 0.5
            src_y = (dst_y + 0.5) * scale_y - 0.5
            src_x0 = int(floor(src_x))
            src_x1 = min(src_x0 + 1, src_w - 1)
            src_y0 = int(floor(src_y))
            src_y1 = min(src_y0 + 1, src_h - 1)
            if src_x0 != src_x1 and src_y1 != src_y0:
                # calculate the interpolation
                temp0 = ((src_x1 - src_x) * img[src_y0, src_x0] + (src_x - src_x0) * img[src_y0, src_x1]) / (src_x1 - src_x0)
                temp1 = ((src_x1 - src_x) * img[src_y1, src_x0] + (src_x - src_x0) * img[src_y1, src_x1]) / (src_x1 - src_x0)
                dst_img[dst_y, dst_x] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1) / (src_y1 - src_y0)
    return dst_img