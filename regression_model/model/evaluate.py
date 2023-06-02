import numpy as np
from util import sigmoid, transform_images, logging, save_images
import numpy as np
import argparse
import json
import os
from tqdm import tqdm

# Predict based on provided parameters
def predict(X, w, b, threshold=0.5):
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)
    ## Classify image as ai generated if value a is more than threshold
    Y_prediction = (A > threshold)
    
    return Y_prediction

# Evaluate accuracy. Be careful of setting limit count of images to save as part of error analysis
def evaluate(X, Y, Y_prediction, is_saved=False, logging_dir='', image_limit = 0):
    n = Y.shape[1]
    synthetic_img = 0
    real_img = 0
    
    true_positives = 0 #predicted synthetic image correctly
    true_negatives = 0 #predicted real image correctly
    false_positives = 0
    false_negatives = 0
    
    ### Save images for error analysis purpose
    directories = [''] * 4
    if save_images and os.path.exists(logging_dir):
        for i, f in enumerate(['true_positives', 'true_negatives', 'false_positives', 'false_negatives']):
            filename = os.path.join(logging_dir, f)
            directories[i] = filename
            os.mkdir(filename)
    
    for i in tqdm(range(n)):
        if Y[0][i] == 1:
            synthetic_img += 1
            if Y_prediction[0][i] == 1:
                true_positives += 1
                # Save image for analysis
                save_images(X, i, true_positives, image_limit, is_saved, directories[0])
            else:
                false_negatives += 1
                save_images(X, i, false_negatives, image_limit, is_saved, directories[3])
        else:
            real_img += 1
            if Y_prediction[0][i] == 0:
                true_negatives += 1
                save_images(X, i, true_negatives, image_limit, is_saved, directories[1])
            else:
                false_positives += 1
                save_images(X, i, false_positives, image_limit, is_saved, directories[2])
                
    accuracy = {'overall_accuracy': (true_positives + true_negatives) / n * 100}  
    if synthetic_img:
        accuracy['synthetic_img_accuracy'] = true_positives / synthetic_img * 100
    if real_img:
        accuracy['real_img_accuracy'] = true_negatives / real_img * 100
    return accuracy
    
    
parser = argparse.ArgumentParser()
parser.add_argument('--params_file', default='../experiment/alpha_0.001_epoch_500/result.json', help='File consists of learned weight and bias')
parser.add_argument('--dataset_dir', default ='../../data/64x64/stylegan3_test/test', help='Directory of images to predict and evaluate accuracy')
parser.add_argument('--logging_dir', default='../result/test1', help='Directory to save evaluation result')
parser.add_argument('--save_images', default='n', help='Save analyzed images folder - save if y')
parser.add_argument('--is_synthetic', default='y', help='Is provided images synthetic? y if synthetic')
parser.add_argument('--saved_image_count', default='500', help='Limit of images to save for error analysis')

# Evaluate a test set
if __name__ == '__main__':
    args = parser.parse_args()
    data_dir = args.dataset_dir
    logging_dir = args.logging_dir
    params_file = args.params_file
    is_saved = True if args.save_images == 'y' else False
    is_synthetic = True if args.is_synthetic == 'y' else False
    image_limit = int(args.saved_image_count)
    
    if not os.path.exists(logging_dir):   
        os.mkdir(logging_dir)
    else:
        raise Exception('Logging directory already exists!')
    
    params = None
    with open(params_file, 'r') as f:
        params = json.load(f)
        # print(params)
        
    weight = np.asarray(params['weight'])
    bias = np.asarray(params['bias'])
    
    X, Y = transform_images(data_dir, is_synthetic)
    Y_prediction = predict(X, weight, bias)
    accuracy = evaluate(X, Y, Y_prediction, is_saved, logging_dir, image_limit)
    
    print(f'Accuracy: {accuracy}')
    with open(os.path.join(logging_dir, 'result.json'), 'w') as f:
        json.dump(accuracy, f, indent=4)
