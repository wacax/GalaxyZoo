#Galaxy Zoo Kaggle competition
#ver 1.0
#__author__ = 'wacax'

#Libraries
import os
import cv
import cv2
import numpy as np
import pandas as pd
from webbrowser import open as imdisplay
from sklearn.feature_extraction import image

#Init

wdPath = '/home/wacax/Documents/Wacax/Kaggle Data Analysis/Galaxy Zoo'
trainingPath = '/home/wacax/Documents/Wacax/Kaggle Data Analysis/Galaxy Zoo/images_training/'
testPath = '/home/wacax/Documents/Wacax/Kaggle Data Analysis/Galaxy Zoo/images_test/'

os.chdir(wdPath)

#Display test image
imageTest = '%s%s' %(trainingPath, '999993.jpg')
imdisplay(imageTest)

#Using cv2
imTest = cv2.imread(imageTest)
print type(imTest)
imTestGray = cv2.cvtColor(imTest, cv2.COLOR_BGR2GRAY)
print type(imTestGray)
cv2.imshow('dst_rt', imTest)
cv2.waitKey(0)
cv2.imshow('dst_rt', imTestGray)
cv2.waitKey(0)

raw_input() #pauses the program

cv2.destroyAllWindows()

#Define Labels
#read file
galaxyType = pd.read_csv('solutions_training.csv')

#Implementing PCA/Whitening
avg = np.mean(imTest, 1);     #Compute the mean pixel intensity value separately for each patch.