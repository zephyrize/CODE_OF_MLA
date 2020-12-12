import numpy as np
import matplotlib.pyplot as plt

def get_data(file_name):
    data, labels = [], []
    f = open(file_name)
    for line in f.readlines():
        line = line.strip().split()
        data.append([1.0, line[0], line[1]])
        labels.append(int(line[2]))

    return np.array(data,dtype=float), np.array(labels)

def get_data_horse(file_name):
    data, labels = [], []
    f = open(file_name)
    for line in f.readlines():
        line = line.strip().split('\t')
        temp = []
        for i in range(len(line)-1):
            temp.append(line[i])
        data.append(temp)
        labels.append(line[-1])
    return np.array(data, dtype=float), np.array(labels,dtype=float)

def sigmoid(x):
    return 1.0/(1 + np.exp(-x))

def classify(inX, w):
    ans = sigmoid(sum(inX*w))
    if ans > 0.5:
        return 1
    else:
        return 0

def grand_ascent(data, labels):
    m, n = data.shape
    w = np.ones(n)
    eta = 0.001
    iter_nums = 1000
    for i in range(iter_nums):
        y_pre = sigmoid(np.dot(data, w))
        err = labels - y_pre
        w = w + eta * np.dot(err, data)
    return w

def random_grand_ascent(data, labels):
    m, n = data.shape
    w = np.ones(n)
    iter_nums = 1000
    dataInd = np.arange(m)
    for j in range(iter_nums):
        for i in range(m):
            eta = 4/(1.0+j+i) + 0.01
            randInd = int(np.random.uniform(0, m))
            y_pre = sigmoid(sum(data[randInd]*w))
            err = labels[randInd] - y_pre
            w = w + eta * err * data[randInd]
            # dataInd = np.delete(dataInd, randInd)
    return w

def plotGraph(data, labels, w):
    m, n = data.shape
    x0, y0 = [], []
    x1, y1 = [], []
    for i in range(m):
        if int(labels[i]) == 0:
            x0.append(data[i,1])
            y0.append(data[i,2])
        else:
            x1.append(data[i,1])
            y1.append(data[i,2])

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(x0, y0, c='red')
    ax.scatter(x1, y1, c='green')
    x = np.arange(-6, 6, 0.1)
    # 设 sigmoid 函数为0，因为0是两个分类的分界处。那么有 0 = w0x0 + w1x1 + w2x2。其中x0=1， 解得x2 = (-w0 - w1x1)/w2
    y = (-w[0] - w[1]*x) / w[2]
    ax.plot(x, y)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
