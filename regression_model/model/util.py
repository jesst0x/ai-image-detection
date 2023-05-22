import numpy as np
from PIL import Image
import os
        
# Appending training log and evaluation result into txt file        
def logging(dir, messages):
    with open(dir, 'a') as f:
        f.write(messages + '\n')

def initialize_with_zeros(dim):
    w  = np.zeros((dim, 1))
    b = 0
    return w, b

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def unpack_image_pixel(filename):
    image = Image.open(filename)
    data = np.asarray(image)
    return data

def load_data(dir):
    filenames = [os.path.join(dir, f) for f in os.listdir(dir)]
    array = np.array([unpack_image_pixel(f) for f in filenames])
    return array

def shuffle(X, Y):
    np.random.seed(40)
    Z = np.concatenate((X, Y)).T
    np.random.shuffle(Z)
    Z = Z.T
    
    return Z[:-1, :], Z[-1, :].reshape(1, Y.shape[1])
    
def transform_images(data_dir, is_real=False):
    X = load_data(data_dir)
    Y = np.zeros((1, X.shape[0]))
    if not is_real:
        Y += 1
    X_flatten = X.reshape((X.shape[0], -1)).T / 255
    return X_flatten, Y
           
# Load dataset and do necessary processing including unpacking images and shuffling of synthetic and real images
def load_dataset():
    synthetic_train_x = load_data('../../data/64x64/stylegan/train')
    synthetic_dev_x = load_data('../../data/64x64/stylegan/dev')
    synthetic_test_x = load_data('../../data/64x64/stylegan/test')
    
    real_train_x = load_data('../../data/64x64/real/train')
    real_dev_x = load_data('../../data/64x64/real/dev')
    real_test_x = load_data('../../data/64x64/real/test')
    
    # real_train_x = load_data('../../data/64x64/real_kaggle/train')
    # real_dev_x = load_data('../../data/64x64/real_kaggle_dev/dev')
    # real_test_x = load_data('../../data/64x64/real_kaggle/test')
    
    synthetic_train_y = np.zeros((1, synthetic_train_x.shape[0])) + 1
    synthetic_dev_y = np.zeros((1, synthetic_dev_x.shape[0])) + 1
    synthetic_test_y = np.zeros((1, synthetic_test_x.shape[0])) + 1
    
    real_train_y = np.zeros((1, real_train_x.shape[0]))
    real_dev_y = np.zeros((1, real_dev_x.shape[0]))
    real_test_y = np.zeros((1, real_test_x.shape[0]))  
    
    train_y = np.concatenate((synthetic_train_y, real_train_y), axis=1)
    dev_y = np.concatenate((synthetic_dev_y, real_dev_y), axis=1)
    test_y = np.concatenate((synthetic_test_y, real_test_y), axis=1)
    
    # Combine both real and synthetic images
    train_x = np.concatenate((synthetic_train_x, real_train_x))
    dev_x = np.concatenate((synthetic_dev_x, real_dev_x))
    test_x = np.concatenate((synthetic_test_x, real_test_x))
    
    # Reshaping x from (m, width, height, channel) to (width x height x channel, m) and normalize the value
    train_x_flatten = train_x.reshape((train_x.shape[0], -1)).T / 255
    dev_x_flatten = dev_x.reshape((dev_x.shape[0], -1)).T / 255
    test_x_flatten = test_x.reshape((test_x.shape[0], -1)).T / 255
    
    # Shuffle training examples
    train_x_flatten, train_y = shuffle(train_x_flatten, train_y)
    
    return train_x_flatten, train_y, dev_x_flatten, dev_y, test_x_flatten, test_y
