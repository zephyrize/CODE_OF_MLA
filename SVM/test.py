import numpy as np
import SMO

file_name = "testSet.txt"
train_data, labels = SMO.get_data(file_name)
data= SMO.start(train_data, labels, 0.6, 100)
data.w = SMO.cal_w(data)
SMO.plot_data(data)