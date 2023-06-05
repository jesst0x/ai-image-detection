import numpy as np
from PIL import Image
import os
import shutil
import matplotlib.pyplot as plt
        
# Appending training log and evaluation result into txt file        
def logging(dir, messages):
    with open(dir, 'a') as f:
        f.write(messages + '\n')

def plot_history(history_dict, logging_dir):
    plt.plot(history_dict['accuracy'], label='train_accuracy')
    plt.plot(history_dict['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(logging_dir, 'accuracy_graph.png'))
    plt.show()
    
    plt.plot(history_dict['loss'], label='train_loss')
    plt.plot(history_dict['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(logging_dir, 'loss_graph.png'))
    plt.show()
    
def convert_pixel_to_image(image_array, size=64):
    image_array = np.uint8(image_array * 255)
    img = Image.fromarray(image_array)
    return img
    
def convert_image_to_array(filename):
    image = Image.open(filename)
    data = np.asarray(image)
    return data

# Load and transform image data to numpy arrays and labels
def load_data(dir, is_synthetic=True):
    filenames = [os.path.join(dir, f) for f in os.listdir(dir)]
    filenames.sort()
    X = np.array([convert_image_to_array(f) for f in filenames])
    Y = np.zeros((X.shape[0], 1))
    if is_synthetic:
        Y += 1
    X = X / 255
    print(X.shape, Y.shape, Y[-1])
    return X, Y
    
# Load dataset and do necessary processing including unpacking images and shuffling of synthetic and real images
def load_training_dataset(synthetic_img_dir, real_img_dir):
    synthetic_train_X, synthetic_train_Y = load_data(os.path.join(synthetic_img_dir, 'train'))
    synthetic_dev_X, synthetic_dev_Y = load_data(os.path.join(synthetic_img_dir, 'dev'))
    
    
    real_train_X, real_train_Y = load_data(os.path.join(real_img_dir, 'train'), False)
    real_dev_X, real_dev_Y = load_data(os.path.join(real_img_dir, 'dev'), False)
    
    
    train_Y = np.concatenate((synthetic_train_Y, real_train_Y))
    dev_Y = np.concatenate((synthetic_dev_Y, real_dev_Y))
    
    # Combine both real and synthetic images
    train_X = np.concatenate((synthetic_train_X, real_train_X))
    dev_X = np.concatenate((synthetic_dev_X, real_dev_X))
    return train_X, train_Y, dev_X, dev_Y