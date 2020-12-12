from PIL import Image
import numpy as np
import KNN
N = 32



training_matrix = np.load('train_data.npy')
training_data_labels = np.load('labels.npy')

for n in range(10):
    img = Image.open('C:\\Users\\Akatsuki\\Pictures\\Screenshots\\%d.jpg' % n) # get the picture
    img = np.array(img) # translate the pic to array
    img = img[:,:,0] # get the two dim array
    img = np.where(img==255,0,1) # modify the data in the array: 255->1, else->0
    # np.savetxt('/home/akatsuki/图片/test3.txt',img,fmt='%d') # save the 32*32 matrix
    img = KNN.cut_picture(img)
    img = KNN.strech_picture(img).reshape(N*N)
    res = KNN.classify(img, training_matrix, training_data_labels, 3)
    print('The %d predicted result is: %d' % (n,res))
