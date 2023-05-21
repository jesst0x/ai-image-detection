import numpy as np
from util import sigmoid

def predict(X, w, b, threshold=0.5):
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)
    ## Classify image as ai generated if value a is more than threshold
    Y_prediction = (A > threshold)
    
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
    
    print(true_positives, synthetic_img,true_positives / synthetic_img * 100 )
    print(true_negatives, real_img, true_negatives / real_img * 100)            
    accuracy = {'overall_accuracy': (true_positives + true_negatives) / n * 100, 'synthetic_img_accuracy': true_positives / synthetic_img * 100, 'real_img_accuracy': true_negatives / real_img * 100}
    
    return accuracy
    