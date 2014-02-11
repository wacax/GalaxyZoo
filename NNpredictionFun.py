#__author__ = 'wacax'
import numpy as np
from sigmoidFun import sigmoid

def predictionFromNNs(theta, input_layer_size, hidden1_layer_size, hidden2_layer_size, num_labels, X):
    Theta1 = np.reshape(theta[0: input_layer_size * hidden1_layer_size], (input_layer_size, hidden1_layer_size))
    Theta2 = np.reshape(theta[input_layer_size * hidden1_layer_size : hidden1_layer_size * input_layer_size + (1 + hidden1_layer_size) * hidden2_layer_size],
                        (1 + hidden1_layer_size, hidden2_layer_size))
    Theta3 = np.reshape(theta[len(theta) - (1 + hidden2_layer_size) * num_labels : len(theta)],
                        (1 + hidden2_layer_size, num_labels))

    m = X.shape[0]
    #Feedforward pass
    hiddenOne = sigmoid(np.dot(X, Theta1))
    vectorOfOnes =  np.tile(1.0, (hiddenOne.shape[0], 1))
    hiddenOne = np.hstack((vectorOfOnes, hiddenOne))
    hiddenTwo = sigmoid(np.dot(hiddenOne, Theta2))
    vectorOfOnes =  np.tile(1.0, (hiddenTwo.shape[0], 1))
    hiddenTwo = np.hstack((vectorOfOnes, hiddenTwo))
    predictionFromNets = sigmoid(np.dot(hiddenTwo, Theta3))
    return(predictionFromNets)
