import numpy as np
from util import sigmoid, logging, relu
import math
import os
from tqdm import tqdm


def gradient(x, y, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    z1 = np.dot(W1, x) + b1
    a1 = relu(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = sigmoid(z2)
    
    dz2 = a2 - y
    dW2 = np.dot(dz2, a1.T)
    db2 = np.sum(dz2, axis=1, keepdims=True)
    da1 = np.dot(W2.T, dz2)
    dz1 = np.array(da1, copy=True)
    dz1[z1 <= 0] = 0
    dW1 = np.dot(dz1, x.T)
    db1 = np.sum(dz1, axis=1, keepdims=True)
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    
    return grads
 
def compute_loss(X, Y, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
     
    n = X.shape[1]
     # Logistic loss
    loss =-(np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2)))/n
    return loss

def update_parameters(parameters, grads, step_size):
    
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
    
    parameters['W1'] -= (step_size * dW1)
    parameters['b1'] -= (step_size * db1)
    parameters['W2'] -= (step_size * dW2)
    parameters['b2'] -= (step_size * db2)
            

def optimize(X, Y, parameters, learning_rate, num_epoch, logging_dir):
    n = X.shape[1]
    
    log_text_file = os.path.join(logging_dir, 'log.txt')
    
    for t in range(1, num_epoch + 1):
        step_size = learning_rate / math.sqrt(t)
        for i in tqdm(range(n)):
            x, y = X[:, i: i+1], Y[:, i:i+1]
            grads = gradient(x, y, parameters)
            update_parameters(parameters, grads, step_size)
            
        loss = compute_loss(X, Y, parameters)
        if t % 10 == 0 or t == num_epoch:
            message = f'At Epoch {t}, loss={loss}'
            logging(log_text_file, message)
            print(message)
       
    return parameters
            




