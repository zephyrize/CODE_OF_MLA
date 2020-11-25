from matplotlib import pyplot as pyplot
from PIL import Image
import numpy as np
import KNN

file_path_train = 'trainingDigits'
training_matrix, training_data_labels = KNN.training_data(file_path_train)
for n in range(10):
    pic = Image.open('C:\\Users\\Akatsuki\\Pictures\\Screenshots\\%d.jpg' % n) # get the picture
    pic = pic.resize((32,32)) # reset the size of pic
    pic_arr = np.array(pic) # translate the pic to array
    pic_arr = pic_arr[:,:,0] # get the two dim array
    pic_arr = np.where(pic_arr==255,1,0) # modify the data in the array: 255->1, else->0
    # np.savetxt('/home/akatsuki/图片/test3.txt',pic_arr,fmt='%d') # save the 32*32 matrix
    temp = np.zeros(1024)
    for i in range(32):
        temp[32*i:32*(i+1)] = pic_arr[i,:]

    
    res = KNN.classify(temp, training_matrix, training_data_labels, 3)

    print('The %d predicted result is: %d' % (n,res))