
# coding: utf-8

# In[31]:


from data_processor import create_training_data
#filename = create_training_data(0, 8)
import pickle
import random
import numpy as np
import data_processor


def load_training_and_testing_data():
    with open('feature_set_class_0_to8.pickle', 'rb') as handle:
        features = pickle.load(handle)

    random.shuffle(features)
    features = np.array(features)

    testing_size = int(0.1*len(features))
    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])
    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])
    
    return train_x, train_y, test_x, test_y

