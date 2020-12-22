import numpy as np
from math import fabs
from matplotlib import pyplot as plt


class data_struct:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.m, self.n = x.shape
        self.alpha = np.zeros((self.m))
        self.tolor = 0.01
        self.b = 0
        self.C = 0
        self.w = np.zeros(self.n)
        self.Ek = np.zeros(self.m)
        self.sup_vec_index = []

# read data from txt file
def getData(file_name):
    data, labels = [], []
    f = open(file_name)
    for line in f.readlines():
        line = line.strip().split('\t')
        data.append([line[0], line[1]])
        labels.append(int(float(line[2])))
    return np.array(data, dtype=float), np.array(labels)


# 公式 7.108 求解剪辑后的alpha值
def clipData(alphaj_new_unc, L, H):
    if alphaj_new_unc > H:
        return H
    elif alphaj_new_unc < L:
        return L
    else:
        return alphaj_new_unc

def KKT(data, i):
    yi = data.y[i]
    alphai = data.alpha[i]
    # 公式 7.104
    g_xi = sum(data.alpha * data.y * sum(np.dot(data.x, data.x[i]))) + data.b
    # 公式 7.111
    if fabs(alphai - 0) <= data.tolor and yi*g_xi >= 1:
        return True
    # 公式 7.112
    elif alphai > -data.tolor and fabs(alphai-data.C) < data.tolor and fabs(yi*g_xi-1) < data.tolor:
        return True
    # 公式 7.113
    elif fabs(alphai-data.C) < data.tolor and yi*g_xi <= 1:
        return True
    else:
        return False


def calEk(i, data):
    return sum(data.alpha * data.y * np.dot(data.x, data.x[i])) + data.b - data.y[i]

# 随机选取alphaj的方式

def selectRandJ(i, m):
    while True:
        j = int(np.random.uniform(0,m))
        if j != i:
            return j

def selectAlphaJ(i, Ei, data):
    index_j = -1
    if Ei >= 0:
        index_j = np.argmin(data.Ek)
    if Ei < 0:
        index_j = np.argmax(data.Ek)
    if index_j == -1:
        index_j = selectRandJ(i, data.m)
    return index_j, calEk(index_j, data)

'''
def selectAlphaJ(i, Ei, data):
    index_j = -1
    maxDelta = -1
    no_zero_E = np.nonzero(data.Ek)[0]
    for k in no_zero_E:
        Ek = calEk(k, data)
        if fabs(Ei - Ek) > maxDelta:
            maxDelta = fabs(Ei - Ek)
            index_j = k
            Ej = Ek
    if index_j == -1:
        index_j = selectRandJ(i, data.m)
        Ej = calEk(index_j, data)
    return index_j, Ej
'''
def update(k, data):
    Ek = calEk(k, data)
    data.Ek[k] = Ek

def updatePara(i, data):

    # 按照书中所写，alpha_i已定，Ei可以计算，选择最大的|Ei - Ej|使得alpha_j最大
    # 特殊情况下，如果以上方法并不能使得目标函数足够下降，那么采用启发式方法继续选择alpha_j
    # 在选择 alpha_j前，需要计算Ei
    # 根据公式(7.105) 计算 Ei
    Ei = calEk(i, data)
    j, Ej= selectAlphaJ(i, Ei, data)
    # Ej = calEk(j, data)

    xi, xj = data.x[i], data.x[j]
    yi, yj = data.y[i], data.y[j]
    b = data.b
    C = data.C
    alphai_old = data.alpha[i]
    alphaj_old = data.alpha[j]

    # 根据P144页求 L，H的公式
    if yi != yj:
        L = max(0, alphaj_old - alphai_old)
        H = min(C, C + alphaj_old - alphai_old)
    else:
        L = max(0, alphaj_old + alphai_old - C)
        H = min(C, alphaj_old + alphai_old)
    if L == H:
        print("~~~~~~~~~~~L========H~~~~~~~~~~~")
        return 0
        
    # 公式7.107 计算 eta(这里未引入核函数，所以只计算内积)
    eta = sum(data.x[i] * data.x[i]) + sum(data.x[j] * data.x[j]) - 2.0 *  sum(data.x[i] * data.x[j])
    if eta <= 0:
        print("eta <= 0")
        return 0
    # 公式 7.106 计算未剪辑的解
    alphaj_new_unc = alphaj_old + yj*(Ei-Ej) / eta
    
    alphaj_new = clipData(alphaj_new_unc, L, H)

    if fabs(alphaj_new - alphaj_old) <= 0.00001:
        return 0
    

    alphai_new = alphai_old + yi * yj * (alphaj_old - alphaj_new)
    

    bi_new = -Ei - yi * (sum(data.x[i] * data.x[i])) * (alphai_new - alphai_old) - yj * (sum(data.x[j] * data.x[i])) * (alphaj_new - alphaj_old) + b
    bj_new = -Ej - yi * (sum(data.x[i] * data.x[j])) * (alphai_new - alphai_old) - yj * (sum(data.x[j] * data.x[j])) * (alphaj_new - alphaj_old) + b
    
    if alphai_new > 0 and alphai_new < C:
        b_new = bi_new
    elif alphaj_new > 0 and alphaj_new < C:
        b_new = bj_new
    else:
        b_new = (bi_new + bj_new) / 2.0
    
    data.b = b_new
    
    # 得到b_new之后，根据公式7.117，更新Ek
    data.alpha[i] = alphai_new
    data.alpha[j] = alphaj_new
    update(i, data)
    update(j, data)
    return 1

def start(x, y, C, max_iter = 100):
    data = data_struct(x, y)
    data.C = C
    # 迭代计步器
    iter = 0
    # 参数是否优化标记， 如果选了m对alpha值 但仍没有得到优化，此时就退出循环
    para_changed = 1
    while iter < max_iter and para_changed != 0:
        para_changed = 0
        iter = iter+1
        for i in range(data.m):
            # 根据P147-第1个变量的选择，外层循环在训练样本中选取违反KKT条件最严重的样本点
            if KKT(data, i) is False:
                flag = updatePara(i, data)
                if flag == 1:
                    para_changed = para_changed + 1
            if para_changed != 0:
                print('第 :%d 轮迭代，选择的i值为：%d，参数修改了:%d次' % (iter, i, para_changed))
    return data

# 根据公式 7.50
def calW(data):
    return np.dot((data.alpha * data.y), data.x)

def plotData(data):
    m, n = data.x.shape
    x0, y0 = [], []
    x1, y1 = [], []
    for i in range(m):
        if int(data.y[i]) == 1:
            x0.append(data.x[i,0])
            y0.append(data.x[i,1])
        else:
            x1.append(data.x[i,0])
            y1.append(data.x[i,1])
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(x0, y0, c='red')
    ax.scatter(x1, y1, c='green')

    # 画出分类直线。由 公式7.55超平面方程(x1, x2) * w + b = 0 ( w = (w1,w2) ),可得 x2 = (-x1w1 - b) / w2
    w = data.w
    # x = np.arange(min(min(x0),min(x1)), max(max(x0),max(x1)), 0.1)
    x = np.arange(0, 10, 0.1)
    y = (-x*w[0] - data.b) / w[1]
    ax.plot(x, y)
    plt.show()
