#Galaxy Zoo Kaggle competition
#ver 2.0
#__author__ = 'wacax'

#Libraries
#import libraries
import os
import cv2
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from webbrowser import open as imdisplay
from sklearn.decomposition import RandomizedPCA
from sklearn import cross_validation
from sklearn.neural_network import BernoulliRBM
from sklearn import preprocessing
from scipy import optimize
from sklearn.metrics import mean_squared_error
from math import ceil
from sigmoidFun import sigmoid
from NNCostFun import nnCostFunction
from NNGradientFun import nnGradFunction
from NNpredictionFun import predictionFromNNs
from scipy import signal
from checkNNGradients import checkNNGradients
from scipy import ndimage
from skimage.util.shape import view_as_windows
from scipy.cluster.vq import kmeans2
from skimage.util.montage import montage2d

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
m =  5000 #pet train dataset
#mTest = len(testImageNames)
mTest = 5000 #pet test dataset
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
def preprocessImgGray(name, dim1, dim2, dataDir):
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

###############################################
dimensionsKernel = 8, 8
NumberOfFilters = 49

patch_shape = dimensionsKernel
n_filters = NumberOfFilters
vectorof255s =  np.tile(255., (424, 424, 3))
NumberOfPatches = 10
NumberofImagesSampled = 10

idx11 = np.random.random_integers(0, len(trainImageNames), NumberofImagesSampled)

def extractPatches(name, patch_shape, dataDir, vector, numberOfPatchesPerImage):
    imageName = '{0:s}{1:s}'.format(dataDir, name)
    npImage = cv2.imread(imageName)
    npImage = np.divide(npImage, vector)
    patchesSampled = np.empty(shape=(NumberOfPatches, patch_shape[0] * patch_shape[1], 3))
    for i in range(npImage.shape[2]):
        patches = view_as_windows(npImage[:,:,i], patch_shape)
        patches = patches.reshape(-1, patch_shape[0] * patch_shape[1])[::8]
        patchesSampled[:,:,i] = patches[np.random.random_integers(0, patches.shape[0], numberOfPatchesPerImage), :]
    return(patchesSampled)

#Init an empty matrix
PatchesMatrix = np.empty(shape=(NumberofImagesSampled * NumberOfPatches,
                                patch_shape[0] * patch_shape[1], 3))

someOtherNumbers = range(NumberofImagesSampled)
counter = 0
for i in someOtherNumbers:
    PatchesMatrix[counter:counter + NumberOfPatches, :, :] = extractPatches(trainImageNames[idx11[i]], patch_shape, dataTrainDir, vectorof255s, NumberOfPatches)
    counter += NumberOfPatches

###kmeans method Gabors Extractor
montages3Channels =  np.empty(shape=(n_filters, 8, 8, PatchesMatrix.shape[2]))
for i in range(PatchesMatrix.shape[2]):
    fb, _ = kmeans2(PatchesMatrix[:,:,i], n_filters, minit='points')
    fb = fb.reshape((-1,) + patch_shape)
#    fb_montage = montage2d(fb, rescale_intensity=True)
    montages3Channels[:,:, :, i] = fb

###RBM method
montages3Channels =  np.empty(shape=(1+ patch_shape[0] * patch_shape[1], n_filters, PatchesMatrix.shape[2]))
RBMPatches = BernoulliRBM(n_components=n_filters, learning_rate=0.1, verbose =True)
vectorOfOnes= np.tile(1.0, (PatchesMatrix.shape[0], 1))
for i in range(PatchesMatrix.shape[2]):
    RBMPatches.fit(np.hstack((vectorOfOnes, PatchesMatrix[:,:,i])))
    RBMComponents = RBMPatches.components_.T
    montages3Channels[:,:, i] = RBMComponents

b =  montages3Channels[0, :, :]
montages3Channels = montages3Channels[1:montages3Channels.shape[0],:,:]

####################################################
#define loading and pre-processing function in color
def preprocessImg(name, dim1, dim2, dataDir, vector, kernelList, b):
    imageName = '{0:s}{1:s}'.format(dataDir, name)
    npImage = cv2.imread(imageName)
    npImage = np.divide(npImage, vector)
    for i in range(npImage.shape[2]):
        convulutedOneImageMatrix = np.empty(shape=(npImage.shape[0] - patch_shape[0] + 1, npImage.shape[1] - patch_shape[1] + 1, n_filters, npImage.shape[2]))
        for ii in range(n_filters):
            kernel = np.reshape(kernelList[:, ii, i], (patch_shape[0], patch_shape[1]))
            convolutedImage = signal.fftconvolve(npImage[:,:,i], np.flipud(np.fliplr(kernel)), mode='valid')
#            convolutedImage = ndimage.convolve(npImage[:,:,i], np.flipud(np.fliplr(kernel)), mode='constant', cval=0.0)
            convulutedOneImageMatrix[:,:,ii, i] = sigmoid(convolutedImage + b[ii, i])

#424 − 8 + 1

indexesImTrain = np.random.permutation(m)
indexesImTest = np.random.permutation(mTest)
testIndexes = range(m, m + mTest)

#Important DO NOT forget
vectorof255s =  np.tile(255., (424, 424, 3))


#Init the empty matrix
bigMatrix = np.empty(shape=(m + mTest, desiredDimensions[0] * desiredDimensions[1] * 3))

someOtherNumbers = range(m)
for i in someOtherNumbers:
    bigMatrix[i, :] = preprocessImg(trainImageNames[i], desiredDimensions[0], desiredDimensions[1], dataTrainDir)

someNumbers = range(mTest)
for ii in someNumbers:
    bigMatrix[testIndexes[ii], :] = preprocessImg(testImageNames[i], desiredDimensions[0], desiredDimensions[1], dataTestDir)

#display raw images
randImgIds = np.random.randint(0, bigMatrix.shape[0], 9)
for i in range(1, len(randImgIds)+1):
    imageFile = np.reshape(bigMatrix[randImgIds[i-1], :], (desiredDimensions[0], desiredDimensions[1], 3))
    location = int('{0:d}{1:d}{2:d}'.format(3, 3, i))
    plt.subplot(location), plt.imshow(imageFile), plt.title(randImgIds[i-1])

plt.show()

#compute the mean for each patch and subtracting it
bigMatrix = preprocessing.scale(bigMatrix)

#calculate number of components to retain 99% of variance
#Extract around 10000 samples and calculate the 99% of variance on a small dataset
#randIndexes = np.random.randint(0, bigMatrix.shape[0], 10000)

#Compute Sigma
#sigma = np.dot(bigMatrix[randIndexes, :].T, bigMatrix[randIndexes, :]) / bigMatrix[randIndexes, :].shape[1]
#SVD decomposition
#U,S,V = np.linalg.svd(sigma) # SVD decomposition of sigma
#def anonFunOne(vector):
#    sumS = np.sum(vector)
#    for ii in range(len(vector)):
#            variance = np.sum(vector[0:ii]) / sumS
#            if variance > 0.99:
#                return(ii)
#                break
#k = anonFunOne(S) + 100

#pca = RandomizedPCA(n_components = k, whiten = True)
#bigMatrix = pca.fit_transform(bigMatrix) # reduced dimension representation of the data, where k is the number of eigenvectors to keep

#display images with 99% of variance

#for i in range(1, len(randImgIds)+1):
#    imageFile = np.reshape(bigMatrix[randImgIds[i-1], 0:bigMatrix.shape[1] - 1], (10, 13, 3))
#    location = int('{0:d}{1:d}{2:d}'.format(3, 3, i))
#    plt.subplot(location), plt.imshow(imageFile), plt.title(randImgIds[i-1])

#plt.show()

#pre-train networks using Restricted Boltzmann Machine
#first layer
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
bigMatrix = min_max_scaler.fit_transform(bigMatrix)

vectorOfOnes =  np.tile(1.0, (bigMatrix.shape[0], 1))
bigMatrix = np.hstack((vectorOfOnes, bigMatrix))

#Divide train Matrix and Test Matrix (for which I don't have labels)
trainMatrixReduced = bigMatrix[someOtherNumbers, :]
testMatrixReduced = bigMatrix[testIndexes, :]

RBM1 = BernoulliRBM(verbose = True)
RBM1.learning_rate = 0.005
RBM1.n_iter = 40
RBM1.n_components = 700
RBM1.fit(bigMatrix)

ThetaHiddenOne = RBM1.components_.T

bigMatrix = sigmoid(np.dot(bigMatrix, ThetaHiddenOne))

vectorOfOnes =  np.tile(1.0, (bigMatrix.shape[0], 1))
bigMatrix = np.hstack((vectorOfOnes, bigMatrix))

#second layer
RBM2 = BernoulliRBM(verbose = True)
RBM2.learning_rate = 0.03
RBM2.n_iter = 40
RBM2.n_components = 500
RBM2.fit(bigMatrix)

ThetaHiddenTwo = RBM2.components_.T
bigMatrix = sigmoid(np.dot(bigMatrix, ThetaHiddenTwo))

vectorOfOnes =  np.tile(1.0, (bigMatrix.shape[0], 1))
bigMatrix = np.hstack((vectorOfOnes, bigMatrix))

#third layer
RBM3 = BernoulliRBM(verbose = True)
RBM3.learning_rate = 0.01
RBM3.n_iter = 40
RBM3.n_components = 37
RBM3.fit(bigMatrix)

ThetaHiddenThree = RBM3.components_.T

#Unroll Parameters
nnThetas = np.concatenate((ThetaHiddenOne.flatten(), ThetaHiddenTwo.flatten(), ThetaHiddenThree.flatten()))

input_layer_size = RBM1.components_.shape[1]
hidden1_layer_size = RBM1.n_components
hidden2_layer_size = RBM2.n_components
num_labels = RBM3.n_components
NNlambda = 1


#display random weights of the first layer
#randWeightsIds = np.random.randint(0, RBM1.n_components, 9)


#for i in range(1, len(randImgIds)+1):
#    imageFile = np.reshape(ThetaHiddenOne[randWeightsIds[i-1], :], (desiredDimensions[0], desiredDimensions[1], 3))
#    location = int('{0:d}{1:d}{2:d}'.format(3, 3, i))
#    plt.subplot(location), plt.imshow(imageFile), plt.title(randImgIds[i-1])

#plt.show()

#Divide training dataset for cross validation purposes
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    trainMatrixReduced, y[0:trainMatrixReduced.shape[0], :], test_size=0.4, random_state=0)

#compute Numerical Gradient
#Check gradients by running checkNNGradients
#checkNNGradients(NNlambda)

#mini-batch learning with either L-BFGS or Conjugate gradient
#Optimization
miniBatchSize = 1000.0
theta = nnThetas
counter = 0
numberOfIterations = range(int(ceil(X_train.shape[0] / miniBatchSize)))
for i in numberOfIterations:
    values2Train = range(counter, counter + int(miniBatchSize))
    counter = np.max(values2Train) + 1

    while X_train.shape[0] <= np.max(values2Train):
        values2Train.remove(values2Train[-1])

    arguments = (input_layer_size, hidden1_layer_size, hidden2_layer_size, num_labels, X_train[values2Train, :], y_train[values2Train, :], NNlambda)
    theta = optimize.fmin_l_bfgs_b(nnCostFunction, x0 = theta, fprime =  nnGradFunction, args = arguments, maxiter = 20, disp = True, iprint = 0 )
    #theta = optimize.fmin_cg(nnCostFunction, x0 = nnThetas, fprime = nnGradFunction, args = arguments, maxiter = 3, disp = True, retall= True )
    theta = np.array(theta[0])


#First prediction
predictionTest = predictionFromNNs(theta, input_layer_size, hidden1_layer_size, hidden2_layer_size, num_labels, X_test)
#RMSE score
RMSE = np.sqrt(mean_squared_error(y_test, predictionTest))
print(RMSE)

predictionMatrix = predictionFromNNs(theta, input_layer_size, hidden1_layer_size, hidden2_layer_size, num_labels, testMatrixReduced)

galaxyID = []
for file in testImageNames:
    galaxyID.append(file.replace('.jpg', ''))

predictionMatrix = np.column_stack((galaxyID, predictionMatrix))

nameColumns = list(galaxyType.columns)
predictionMatrix = np.row_stack((nameColumns, predictionMatrix))

ofile = open('predictionI.csv', "wb")
fileToBeWritten = csv.writer(ofile, delimiter=',')

for row in predictionMatrix:
    fileToBeWritten.writerow(row)

ofile.close()

