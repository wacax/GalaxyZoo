#__author__ = 'wacax'

from numpy import reshape, tile, sin, concatenate, mod, transpose, ceil, max, array, hstack, linalg
from scipy import optimize
from computeNumericalGradient import computeNumericalGradient
from NNCostFun import nnCostFunction
from NNGradientFun import nnGradFunction



def debugInitializeWeights(fan_out, fan_in):
    W  =  tile(0.0, (1 + fan_in, fan_out))
    W = reshape(sin(range(0 ,W.shape[1] * W.shape[0])), W.shape) / 10.0
    return(W)


def checkNNGradients(NNlambda = 0.0):
    input_layer_size = 3
    hidden1_layer_size = 5
    hidden2_layer_size = 4
    num_labels = 3
    m = 5

    #We generate some 'random' test data
    Theta1 = debugInitializeWeights(input_layer_size, hidden1_layer_size)
    Theta2 = debugInitializeWeights(hidden1_layer_size, hidden2_layer_size)
    Theta3 = debugInitializeWeights(hidden2_layer_size, num_labels)

    # Reusing debugInitializeWeights to generate X
    X  = debugInitializeWeights(m, input_layer_size - 1)
    y  = 1.0 + transpose(mod(range(0, m), num_labels))

    # Unroll parameters
    nn_params = concatenate((Theta1.flatten(), Theta2.flatten(), Theta3.flatten()))

    miniBatchSize = 1000.0
    theta = nn_params
    counter = 0
    numberOfIterations = range(int(ceil(X.shape[0] / miniBatchSize)))
    for i in numberOfIterations:
        values2Train = range(counter, counter + int(miniBatchSize))
        counter = max(values2Train) + 1

        while X.shape[0] <= max(values2Train):
            values2Train.remove(values2Train[-1])

        arguments = (input_layer_size, hidden1_layer_size, hidden2_layer_size, num_labels, X[values2Train, :], y[values2Train, :], NNlambda)
        theta = optimize.fmin_l_bfgs_b(nnCostFunction, x0 = theta, fprime =  nnGradFunction, args = arguments, maxiter = 20, disp = True, iprint = 0 )
        #theta = optimize.fmin_cg(nnCostFunction, x0 = nnThetas, fprime = nnGradFunction, args = arguments, maxiter = 3, disp = True, retall= True )
        theta = array(theta[0])

    cost = nnCostFunction(theta, input_layer_size, hidden1_layer_size, hidden2_layer_size, num_labels, X, y, NNlambda)
    grad = nnGradFunction(theta, input_layer_size, hidden1_layer_size, hidden2_layer_size, num_labels, X, y, NNlambda)

    numgrad = computeNumericalGradient(theta, input_layer_size, hidden1_layer_size, hidden2_layer_size, num_labels, X, y, NNlambda)

    # Visually examine the two gradient computations.  The two columns you get should be very similar.
    print(hstack((numgrad, grad)))
    print('The above two columns you get should be very similar')

    # Evaluate the norm of the difference between two solutions.
    # If you have a correct implementation, and assuming you used EPSILON = 0.0001
    # in computeNumericalGradient.m, then diff below should be less than 1e-9
    diff = linalg.norm(numgrad-grad)/linalg.norm(numgrad+grad)

    print('If your backpropagation implementation is correct, then the relative difference will be small (less than 1e-9) relative Difference')
    print(diff)


