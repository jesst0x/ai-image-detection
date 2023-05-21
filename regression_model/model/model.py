import util
import argparse
import train
import os
from evaluate import predict, evaluate
import json

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', default='0.0001', help='Hyperparameter learning rate')
parser.add_argument('--epoch', default ='10', help='Directory to save resize dataset')

def model(train_x, train_y, dev_x, dev_y,learning_rate, num_epoch, logging_dir):
    
    w, b = util.initialize_with_zeros(train_x.shape[0])
    w, b = train.optimize(train_x, train_y, w, b, learning_rate, num_epoch, logging_dir)
    
    Y_prediction_train = predict(train_x, w, b)
    Y_prediction_dev = predict(dev_x, w, b)
    
    accuracy_train = evaluate(train_y, Y_prediction_train)
    accuracy_test = evaluate(dev_y, Y_prediction_dev)
    
    print(f'Training set accuracy: {accuracy_train}')
    print(f'Dev set accuracy: {accuracy_test}')
    
    summary = {'accuracy_train': accuracy_train, 'accuracy_test': accuracy_test, 'weight': w.tolist(), 'bias': b.tolist()}
    
    with open(os.path.join(logging_dir, 'result.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    return summary    

if __name__ == '__main__':
    args = parser.parse_args()
    #Hyperparameter
    learning_rate = float(args.learning_rate)
    num_epoch = int(args.epoch)
    
    # Make directory to save our experimental parameters
    params_string = f'alpha_{learning_rate}_epoch_{num_epoch}'
    logging_dir = '../experiment/' + params_string
    if not os.path.exists(logging_dir):   
        os.mkdir(logging_dir)
    else:
        print('Logging directory already exists, overwritting result')
    
    ##Saving hyperparameters
    with open(os.path.join(logging_dir, 'params.json'), 'w') as f:
        json.dump({'learning_rate': learning_rate, 'num_epoch': num_epoch}, f, indent=4)
        
    #Load dataset with features extracted
    train_x, train_y, dev_x, dev_y, test_x, test_y = util.load_dataset()
    
    # Running model and evaluate
    model(train_x, train_y, dev_x, dev_y, learning_rate, num_epoch, logging_dir)
    