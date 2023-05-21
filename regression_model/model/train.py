import numpy as np
from util import sigmoid, logging
import math
import os
from tqdm import tqdm


def gradient(x, y, w, b):
    n = x.shape[1]
    # Forward propogation
    z = np.dot(w.T, x) + b
    a = sigmoid(z)
    
    dw = np.dot(x, (a - y).T) / n
    db = np.sum(a - y) / n
    
    return dw, db
 
def compute_loss(X, Y, w, b):
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)
     
    n = X.shape[1]
     # Logistic loss
    loss =-(np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)))/n
    return loss
        

def optimize(X, Y, w, b, learning_rate, num_epoch, logging_dir):
    n = X.shape[1]
    
    log_text_file = os.path.join(logging_dir, 'log.txt')
    
    for t in range(1, num_epoch + 1):
        step_size = learning_rate / math.sqrt(t)
        for i in tqdm(range(n)):
            x, y = X[:, i: i+1], Y[:, i:i+1]
            dw, db = gradient(x, y, w, b)
            w = w - (step_size * dw)
            b = b - (step_size * db)
            
        loss = compute_loss(X, Y, w, b)
        if t % 10 == 0 or t == num_epoch:
            message = f'At Epoch {t}, loss={loss}'
            logging(log_text_file, message)
            print(message)
       
    return w, b
            




