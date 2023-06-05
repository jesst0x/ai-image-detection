import numpy as np
import util
import numpy as np
import argparse
import json
import os
from tqdm import tqdm
import tensorflow as tf
    
    
parser = argparse.ArgumentParser()
parser.add_argument('--model_file', default='../experiment/epoch_10_64x64/trained_model', help='File consists of trained model')
parser.add_argument('--dataset_dir', default ='../../data/64x64/progan_test/test', help='Directory of images to predict and evaluate accuracy')
parser.add_argument('--logging_dir', default='../result/progan_test_64x64', help='Directory to save evaluation result')
parser.add_argument('--is_synthetic', default='y', help='Is provided images synthetic? y if synthetic')


# Evaluate a test set from a trained model
if __name__ == '__main__':
    args = parser.parse_args()
    data_dir = args.dataset_dir
    logging_dir = args.logging_dir
    model_file = args.model_file
    # is_saved = True if args.save_images == 'y' else False
    is_synthetic = True if args.is_synthetic == 'y' else False
    # image_limit = int(args.saved_image_count)
    
    if not os.path.exists(logging_dir):   
        os.mkdir(logging_dir)
    else:
        raise Exception('Logging directory already exists!')
    
    test_X, test_Y = util.load_data(data_dir, is_synthetic) 

    # Load the train model
    reconstructed_model = tf.keras.models.load_model(model_file)
    test_loss, test_acc = reconstructed_model.evaluate(test_X,  test_Y, verbose=2)
    
    print(f'Accuracy: {test_acc}')
    with open(os.path.join(logging_dir, 'result.txt'), 'w') as f:
        f.write(f'Accuracy: {test_acc}')
