#__author__ = 'wacax'
import numpy as np
from NNCostFun import nnCostFunction

def computeNumericalGradient(theta, input_layer_size, hidden1_layer_size, hidden2_layer_size, num_labels, X, y, NNlambda):

    numgrad =  np.tile(0.0, len(theta))
    perturb =  np.tile(0.0, len(theta))

    e = 1e-4
    for p in range(len(theta)):
        # Set perturbation vector
        perturb[p] = e
        loss1 = nnCostFunction(theta - perturb, input_layer_size, hidden1_layer_size, hidden2_layer_size, num_labels, X, y, NNlambda)
        loss2 = nnCostFunction(theta + perturb, input_layer_size, hidden1_layer_size, hidden2_layer_size, num_labels, X, y, NNlambda)
        # Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2.0 * e)
        perturb[p] = 0.0

    return(numgrad)
