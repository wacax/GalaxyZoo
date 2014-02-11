#__author__ = 'wacax'
from sigmoidFun import sigmoid

#define sigmoid gradient
def sigmoidGradient(z):
    g = sigmoid(z) * (1-sigmoid(z))
    return(g)

