import KNN
from os import listdir

def test():

    file_path_train = 'trainingDigits'
    training_matrix, training_data_labels = KNN.training_data(file_path_train)

    file_path_test = 'testDigits'
    err_cnt = 0
    test_data_list = listdir(file_path_test)
    test_data_len = len(test_data_list)
    for i in range(test_data_len):
        file_name_str, class_num = KNN.file_process(test_data_list[i])
        test_vector = KNN.img2vector('testDigits\\' + file_name_str)
        res = KNN.classify(test_vector, training_matrix, training_data_labels, 3)
        
        #if i%10 == 0:
        #    print('The predicted answer is: %d, the real answer is %d' % (res, class_num))
        
        if res != class_num:
            err_cnt +=1
            print('错误文件: ' + file_name_str)
            print('The predicted answer is: %d, the real answer is %d\n' % (res, class_num))
    print('错误的个数:', err_cnt)
    print('正确率: %f' % ((float(test_data_len-err_cnt))/(float(test_data_len))))



test()