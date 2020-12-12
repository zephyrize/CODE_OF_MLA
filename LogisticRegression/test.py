import LR

data, labels = LR.get_data('testSet.txt')
# print(data.shape)
# print(labels.shape)

w = LR.random_grand_ascent(data, labels)

print(w)
LR.plotGraph(data, labels, w)