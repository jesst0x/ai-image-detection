import numpy as np
from util import sigmoid, transform_images, relu
import numpy as np
import argparse
import json

def predict(X, parameters, threshold=0.5):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    ## Classify image as ai generated if value a is more than threshold
    Y_prediction = (A2 > threshold)
    
    return Y_prediction

def evaluate(Y, Y_prediction):
    n = Y.shape[1]
    synthetic_img = 0
    real_img = 0
    
    true_positives = 0 #predicted synthetic image correctly
    true_negatives = 0 #predicted real image correctly
    
    for i in range(n):
        if Y[0][i] == 1:
            synthetic_img += 1
            if Y_prediction[0][i] == 1:
                true_positives += 1
        else:
            real_img += 1
            if Y_prediction[0][i] == 0:
                true_negatives += 1
    accuracy = {'overall_accuracy': (true_positives + true_negatives) / n * 100}  
    if synthetic_img:
        accuracy['synthetic_img_accuracy'] = true_positives / synthetic_img * 100
    if real_img:
        accuracy['real_img_accuracy'] = true_negatives / real_img * 100
    
    return accuracy
    
    
parser = argparse.ArgumentParser()
parser.add_argument('--params_file', default='../experiment/alpha_0.001_epoch_500/result.json', help='File consists of learned weight and bias')
parser.add_argument('--dataset_dir', default ='../../data/64x64/real_kaggle_dev/dev', help='Directory of images to predict and evaluate accuracy')

if __name__ == '__main__':
    args = parser.parse_args()
    data_dir = args.dataset_dir
    params_file = args.params_file
    
    params = None
    with open(params_file, 'r') as f:
        params = json.load(f)
        print(params)
        
    weight = np.asarray(params['weight'])
    bias = np.asarray(params['bias'])
    
    X, Y = transform_images(data_dir, True)