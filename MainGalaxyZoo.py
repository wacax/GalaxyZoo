#Galaxy Zoo Kaggle competition
#ver 1.0
#__author__ = 'wacax'

#Libraries
#import libraries
import os
import cv2
import csv
import numpy as np
import pandas as pd
from webbrowser import open as imdisplay
from scipy.sparse import lil_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import BernoulliRBM
from sklearn import preprocessing
from scipy import optimize

#Init

wd = '/home/wacax/Documents/Wacax/Kaggle Data Analysis/Galaxy Zoo/'
dataTrainDir = '/home/wacax/Documents/Wacax/Kaggle Data Analysis/Galaxy Zoo/images_training_rev1/'
dataTestDir = '/home/wacax/Documents/Wacax/Kaggle Data Analysis/Galaxy Zoo/images_test_rev1/'

#Get names of training image files
path, dirs, trainImageNames = os.walk(dataTrainDir).next()
#Get names of test image files
path, dirs, testImageNames = os.walk(dataTestDir).next()

for file in trainImageNames:
  if not file.endswith('.jpg'):
     trainImageNames.remove(file)

for file in testImageNames:
  if not file.endswith('.jpg'):
     testImageNames.remove(file)

#m = len(trainImageNames)
m = 200 #pet train dataset
#mTest = len(testImageNames)
mTest = 200 #pet test dataset
testImageNames = sorted(testImageNames)
trainImageNames = sorted(trainImageNames)

#Display test image
randImg = np.random.randint(0,len(trainImageNames))
randImg = trainImageNames[randImg]
imageTest = '%s%s' %(dataTrainDir, randImg)
imdisplay(imageTest)

#Define Labels
#read file
galaxyType = pd.read_csv('training_solutions_rev1.csv')
y = np.array(galaxyType)
y = y[:, 1:y.shape[1]]

desiredDimensions = [30, 30]

#define loading and pre-processing function grayscale
def preprocessImg(name, dim1, dim2, dataDir):
    imageName = '{0:s}{1:s}'.format(dataDir, name)
    npImage = cv2.imread(imageName)
    npImage = cv2.cvtColor(npImage, cv2.COLOR_BGR2GRAY)
    avg = np.mean(npImage.reshape(1, npImage.shape[0] * npImage.shape [1]))
    avg = np.tile(avg, (npImage.shape[0], npImage.shape [1]))
    npImage = npImage - avg
    npImage = cv2.resize(npImage, (dim1, dim2))
    return(npImage.reshape(1, dim1 * dim2))

#define loading and pre-processing function in color
def preprocessImg(name, dim1, dim2, dataDir):
    imageName = '{0:s}{1:s}'.format(dataDir, name)
    npImage = cv2.imread(imageName)
    vectorof255s =  np.tile(255., (npImage.shape[0], npImage.shape [1], 3))
    npImage = np.divide(npImage, vectorof255s)
    npImage = cv2.resize(npImage, (dim1, dim2))
    return(npImage.reshape(1, dim1 * dim2 * 3))

#define sigmoid function, lazy numpy doesn't have one
def sigmoid(X):
    g = 1.0 / (1.0 + np.exp(-X))
    return(g)

#define sigmoid gradient
def sigmoidGradient(z):
    g = sigmoid(z) * (1-sigmoid(z))
    return(g)


indexesImTrain = np.random.permutation(m)
indexesImTest = np.random.permutation(mTest)
testIndexes = range(m, m + mTest)

#Init the empty matrix
bigMatrix = np.empty(shape=(m + mTest, desiredDimensions[0] * desiredDimensions[1] * 3))

someOtherNumbers = range(m)
for i in someOtherNumbers:
    bigMatrix[i, :] = preprocessImg(trainImageNames[i], desiredDimensions[0], desiredDimensions[1], dataTrainDir)

someNumbers = range(mTest)
for ii in someNumbers:
    bigMatrix[testIndexes[ii], :] = preprocessImg(testImageNames[i], desiredDimensions[0], desiredDimensions[1], dataTestDir)

#compute the mean for each patch and subtracting it
avg = np.mean(bigMatrix, 0)
bigMatrix = bigMatrix - np.tile(avg, (bigMatrix.shape[0], 1))

#Compute Sigma
sigma = np.dot(bigMatrix.T, bigMatrix) / bigMatrix.shape[1]

#SVD decomposition
U,S,V = np.linalg.svd(sigma) # SVD decomposition of sigma

def anonFunOne(vector):
    sumS = np.sum(vector)
    for ii in range(len(vector)):
            variance = np.sum(vector[0:ii]) / sumS
            if variance > 0.99:
                return(ii)
                break

k = anonFunOne(S)
epsilon = 0.01
#xRot = np.dot(bigMatrix, U)
#covarianceMatrix = np.cov(np.dot(bigMatrix, U), rowvar=False)            # corr matrix of the rotated version of the data.
#cv2.imshow("covarianceMatrix", covarianceMatrix)
#cv2.waitKey(0)
bigMatrix = np.dot(bigMatrix, U[:, 1:k])    # reduced dimension representation of the data, where k is the number of eigenvectors to keep

#D = np.diag(1./np.sqrt(np.diag(S) + epsilon))
# whitening matrix
#W = np.dot(np.dot(U,D),U.T)
# multiply by the whitening matrix
#bigMatrix = np.dot(bigMatrix,W)

#PCA whitening
#bigMatrix = np.dot(np.dot(bigMatrix, U), np.diag(1./np.sqrt(np.diag(S) + epsilon)))

#xZCAWhite = U * np.diag(1./np.sqrt(np.diag(S) + epsilon)) * U.transpose() * bigMatrix

#Build the sparse matrix with the preprocessed image data for both train and test data
#bigMatrix = lil_matrix((m + mTest), desiredDimensions[0] * desiredDimensions[1])

#someOtherNumbers = range(m)
#for i in someOtherNumbers:
#    bigMatrix[i, :] = preprocessImg(trainImageNames[i], desiredDimensions[0], desiredDimensions[1], dataTrainDir)

#someNumbers = range(mTest)
#for ii in someNumbers:
#    bigMatrix[testIndexes[ii], :] = preprocessImg(testImageNames[i], desiredDimensions[0], desiredDimensions[1], dataTestDir)

#Transform to csr matrix and standarization
#bigMatrix = bigMatrix.tocsr()
#scaler = StandardScaler(with_mean=False)
#scaler.fit(bigMatrix)
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)  # apply same transformation to test data

#bigMatrix = preprocessing.scale(bigMatrix, with_mean=False)

#pre-train networks using Restricted Boltzmann Machine
#first layer
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
bigMatrix = min_max_scaler.fit_transform(bigMatrix)

vectorOfOnes =  np.tile(1.0, (bigMatrix.shape[0], 1))
bigMatrix = np.hstack((vectorOfOnes, bigMatrix))

RBM1 = BernoulliRBM(verbose = True)
RBM1.learning_rate = 0.05
RBM1.n_iter = 40
RBM1.n_components = 100
RBM1.fit(bigMatrix)

#Divide train Matrix and Test Matrix (for which I don't have labels)
trainMatrixReduced = bigMatrix[someOtherNumbers, :]
testMatrixReduced = bigMatrix[testIndexes, :]

#Divide training dataset for cross validation purposes
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    trainMatrixReduced, y[0:m, :], test_size=0.4, random_state=0)

ThetaHiddenOne = RBM1.components_.T

hiddenOne = sigmoid(np.dot(X_train, ThetaHiddenOne))

vectorOfOnes =  np.tile(1.0, (hiddenOne.shape[0], 1))
hiddenOne = np.hstack((vectorOfOnes, hiddenOne))

#second layer
RBM2 = BernoulliRBM(verbose = True)
RBM2.learning_rate = 0.05
RBM2.n_iter = 40
RBM2.n_components = 50
RBM2.fit(hiddenOne)

ThetaHiddenTwo = RBM2.components_.T
hiddenTwo = sigmoid(np.dot(hiddenOne, ThetaHiddenTwo))

vectorOfOnes =  np.tile(1.0, (hiddenTwo.shape[0], 1))
hiddenTwo = np.hstack((vectorOfOnes, hiddenTwo))

#third layer
RBM3 = BernoulliRBM(verbose = True)
RBM3.learning_rate = 0.05
RBM3.n_iter = 40
RBM3.n_components = 37
RBM3.fit(hiddenTwo)

ThetaHiddenThree = RBM3.components_.T

nnThetas = np.concatenate((ThetaHiddenOne.flatten(), ThetaHiddenTwo.flatten(), ThetaHiddenThree.flatten()))

input_layer_size = RBM1.components_.shape[1]
hidden1_layer_size = RBM1.n_components
hidden2_layer_size = RBM2.n_components
num_labels = RBM3.n_components
NNlambda = 1

#define Cost Function
def nnCostFunction(nnThetas, input_layer_size, hidden1_layer_size, hidden2_layer_size, num_labels, X, y, NNlambda):

    Theta1 = np.reshape(nnThetas[0: input_layer_size * hidden1_layer_size], (input_layer_size, hidden1_layer_size))
    Theta2 = np.reshape(nnThetas[input_layer_size * hidden1_layer_size : hidden1_layer_size * input_layer_size + (1 + hidden1_layer_size) * hidden2_layer_size],
                        (1 + hidden1_layer_size, hidden2_layer_size))
    Theta3 = np.reshape(nnThetas[len(nnThetas) - (1 + hidden2_layer_size) * num_labels : len(nnThetas)],
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

#define Gradient
def nnGradFunction(nnThetas, input_layer_size, hidden1_layer_size, hidden2_layer_size, num_labels, X, y, NNlambda):

    Theta1 = np.reshape(nnThetas[0: input_layer_size * hidden1_layer_size], (input_layer_size, hidden1_layer_size))
    Theta2 = np.reshape(nnThetas[input_layer_size * hidden1_layer_size : hidden1_layer_size * input_layer_size + (1 + hidden1_layer_size) * hidden2_layer_size],
                        (1 + hidden1_layer_size, hidden2_layer_size))
    Theta3 = np.reshape(nnThetas[len(nnThetas) - (1 + hidden2_layer_size) * num_labels : len(nnThetas)],
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

    delta4 = out - y
    delta4_Theta3 = np.dot(delta4, Theta3.T)
    delta3 = delta4_Theta3[:, 1:delta4_Theta3.shape[1]] * sigmoidGradient(np.dot(hiddenOne,Theta2))
    delta3_Theta2 = np.dot(delta3, Theta2.T)
    delta2 = delta3_Theta2[:, 1:delta3_Theta2.shape[1]] * sigmoidGradient(np.dot(X,Theta1))

    #Regularization term of the gradient
    reg_grad1 = (NNlambda/m) * Theta1[1:Theta1.shape[0], :]
    reg_grad2 = (NNlambda/m) * Theta2[1:Theta2.shape[0], :]
    reg_grad3 = (NNlambda/m) * Theta3[1:Theta3.shape[0], :]

    Theta1_grad = (1.0/m) * (np.dot(delta2.T, X))
    Theta2_grad = (1.0/m) * (np.dot(delta3.T, hiddenOne))
    Theta3_grad = (1.0/m) * (np.dot(delta4.T, hiddenTwo))


    Theta1_grad = (np.column_stack((Theta1_grad[:, 1], Theta1_grad[:, 1:Theta1_grad.shape[1]] + reg_grad1.T))).T
    Theta2_grad = (np.column_stack((Theta2_grad[:, 1], Theta2_grad[:, 1:Theta2_grad.shape[1]] + reg_grad2.T))).T
    Theta3_grad = (np.column_stack((Theta3_grad[:, 1], Theta3_grad[:, 1:Theta3_grad.shape[1]] + reg_grad3.T))).T

    grad = np.concatenate((Theta1_grad.flatten(), Theta2_grad.flatten(), Theta3_grad.flatten()))
    return(grad)

#Short Cut functions
def costRun(theta):
    return nnCostFunction(nnThetas, input_layer_size, hidden1_layer_size, hidden2_layer_size, num_labels, X_train, y_train, NNlambda)

def gradientRun(theta):
    return nnGradFunction(nnThetas, input_layer_size, hidden1_layer_size, hidden2_layer_size, num_labels, X_train, y_train, NNlambda)

#Optimization
#thetaOptimized = optimize.fmin_bfgs(costRun, nnThetas, gradientRun, disp=True, maxiter=400, full_output = True, retall=True)
thetaOptimized = optimize.fmin_bfgs(costRun, nnThetas, disp=True, maxiter=400, full_output = True, retall=True)


#theta = optimize.fmin_bfgs(nnCostFunction, x0 = initial_values, fprime =  nnGradFunction, args=myargs, disp = True, retall= True)
#theta = optimize.fmin_bfgs(nnCostFunction, x0 = initial_values, disp = True, retall= True)

#random grid search of hiperparameters
#create a classifier
clf = svm.SVC(verbose = True)

# specify parameters and distributions to sample from
params2Test = {'C': [1, 3, 10, 30, 100, 300], 'gamma': [0.001], 'kernel': ['rbf']}

#run randomized search
grid_search = GridSearchCV(clf, param_grid = params2Test)

start = time()
grid_search.fit(trainMatrixReduced, y[0:24999])
print("GridSearchCV took %.2f seconds for %d candidate parameter settings." % (time() - start, len(grid_search.grid_scores_)))
type(grid_search)
grid_search.grid_scores_

#Machine Learning part
#Support vector machine model
clf.fit(X_train, y_train)

#prediction
predictionFromDataset = clf.predict(X_test)

correctValues = sum(predictionFromDataset == y_test)
percentage = float(correctValues)/len(y_test)

print(percentage)

#prediction probability
predictionFromDataset2 = clf.predict_proba(X_test)
predictionFromDataset2 = predictionFromDataset2[:, 1]
fpr, tpr, thresholds = metrics.roc_curve(y_test, predictionFromDataset2)
predictionProbability = metrics.auc(fpr, tpr)

#Predict images from the test set
#Train the model with full data set
clf = svm.SVC(C = 10, gamma = 0.001, kernel= 'rbf',verbose = True)
clf.fit(trainMatrixReduced, y[0:24999]) #fix this

#Prediction
#predictionFromTest = clf.predict_proba(testMatrixReduced)
predictionFromTest = clf.predict(testMatrixReduced)
#label = predictionFromTest[:, 1]
idVector = range(1, mTest + 1)

#predictionsToCsv = np.column_stack((idVector, label))
predictionsToCsv = np.column_stack(idVector, predictionFromTest)

import csv

ofile = open('predictionVII.csv', "wb")
fileToBeWritten = csv.writer(ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)

for row in predictionsToCsv:
    fileToBeWritten.writerow(row)

ofile.close()

