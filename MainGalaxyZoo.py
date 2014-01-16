#Galaxy Zoo Kaggle competition
#ver 1.0
#__author__ = 'wacax'

#Libraries
#import libraries
import os
import cv2
import numpy as np
from webbrowser import open as imdisplay
from scipy.sparse import lil_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn import cross_validation
from sklearn import svm
from sklearn import metrics

#Init

wd = '/home/wacax/Documents/Wacax/Kaggle Data Analysis/Galaxy Zoo/'
dataTrainDir = '/home/wacax/Documents/Wacax/Kaggle Data Analysis/Galaxy Zoo/images_training/'
dataTestDir = '/home/wacax/Documents/Wacax/Kaggle Data Analysis/Galaxy Zoo/images_test/'

os.chdir(wd)

#Display test image
imageTest = '%s%s' %(dataTrainDir, '999993.jpg')
imdisplay(imageTest)

#Using cv2
imTest = cv2.imread(imageTest)
print type(imTest)
imTestGray = cv2.cvtColor(imTest, cv2.COLOR_BGR2GRAY)
print type(imTestGray)
cv2.imshow('dst_rt', imTest)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('dst_rt', imTestGray)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Define Labels
#read file
galaxyType = pd.read_csv('solutions_training.csv')
desiredDimensions = [50, 50]

#define loading and pre-processing function grayscale
def preprocessImg(number, dim1, dim2, dataDir):
    imageName = '{0:s}{1:d}{2:s}'.format(dataDir, number, '.jpg')
    npImage = cv2.imread(imageName)
    npImage = cv2.cvtColor(npImage, cv2.COLOR_BGR2GRAY)
    avg = np.mean(npImage.reshape(1, npImage.shape[0] * npImage.shape [1]))
    avg = np.tile(avg, (npImage.shape[0], npImage.shape [1]))
    npImage = npImage - avg
    npImage = cv2.resize(npImage, (dim1, dim2))
    return(npImage.reshape(1, dim1 * dim2))

#m = 5000 #pet Train dataset
m = 12500 #full Train dataset
mTest = 12500 #number of images in the test set

