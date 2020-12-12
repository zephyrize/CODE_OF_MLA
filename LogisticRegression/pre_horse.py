import LR
import numpy as np


def test():
    data, labels = LR.get_data_horse('horseColicTraining.txt')
    # print(data.shape)
    # print(labels.shape)
    w = LR.random_grand_ascent(data, labels)
    f_test = open('horseColicTest.txt')
    ok = 0
    num = 0
    for line in f_test.readlines():
        num = num+1
        line = line.strip().split('\t')
        inX = []
        for i in range(len(line)-1):
            inX.append(float(line[i]))
        inX = np.array(inX)
        label = int(line[-1])
        res = LR.classify(inX, w)
        if res == label:
            ok = ok+1
    acc_rate = 1.0*ok/num
    print('acurate rate is: ', acc_rate)
    return acc_rate
def muti_test():
    num_test = 10;
    err = 0.0
    for i in range(num_test):
        err += test()
    print(err/10)

muti_test()
