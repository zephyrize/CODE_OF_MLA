import KNN
import numpy as np
file_path_train = 'trainingDigits'
training_matrix, training_data_labels = KNN.training_data(file_path_train)
np.save('train_data.npy',training_matrix)
np.save('labels.npy',training_data_labels)