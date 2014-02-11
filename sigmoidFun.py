#__author__ = 'wacax'
#define sigmoid function, lazy numpy doesn't have one
import numpy as np

def sigmoid(X):
    g = 1.0 / (1.0 + np.exp(-X))
    return(g)