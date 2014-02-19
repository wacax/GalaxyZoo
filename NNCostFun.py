#__author__ = 'wacax'
#define Cost Function
import numpy as np
from sigmoidFun import sigmoid

def nnCostFunction(Thetas, input_layer_size, hidden1_layer_size, hidden2_layer_size, num_labels, X, y, NNlambda):

    Theta1 = np.reshape(Thetas[0: input_layer_size * hidden1_layer_size], (input_layer_size, hidden1_layer_size))
    Theta2 = np.reshape(Thetas[input_layer_size * hidden1_layer_size : hidden1_layer_size * input_layer_size + (1 + hidden1_layer_size) * hidden2_layer_size],
                        (1 + hidden1_layer_size, hidden2_layer_size))
    Theta3 = np.reshape(Thetas[len(Thetas) - (1 + hidden2_layer_size) * num_labels : len(Thetas)],
                        (1 + hidden2_layer_size, num_labels))

    m = X.shape[0]
    #Feedforward pass
    hiddenOne = sigmoid(np.dot(X, Theta1))
    vectorOfOnes =  np.tile(1.0, (hiddenOne.shape[0], 1))
    hiddenOne = np.hstack((vectorOfOnes, hiddenOne))
    hiddenTwo = sigmoid(np.dot(hiddenOne, Theta2))
    vectorOfOnes =  np.tile(1.0, (hiddenTwo.shape[0], 1))
    hiddenTwo = np.hstack((vectorOfOnes, hiddenTwo))
    out = sigmoid(np.dot(hiddenTwo, Theta3))

    #Regularization Term
    reg = (NNlambda/(2*m))*(np.sum(np.square(np.sum(Theta1[1:Theta1.shape[0], :], 0))) + np.sum(np.square(np.sum(Theta2[1:Theta2.shape[0], :], 0))) + np.sum(np.square(np.sum(Theta3[1:Theta3.shape[0], :], 0))))
    #Cost Function
    J = (1.0/m) * np.sum(np.sum(-y * np.log(out)-(1.0-y) * np.log(1.0-out), 0)) + reg
    return(J)
