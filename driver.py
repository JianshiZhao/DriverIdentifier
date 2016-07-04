# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 21:24:55 2016

@author: Jianshi
"""
import os
import pandas
import numpy
from PIL import Image
from time import time
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD
import pdb
import pyqtgraph as pg

#
driver_list = pandas.read_csv('./driver_imgs_list.csv')

#------------------------ Functions to Convert Images to Arrays --------------

img_path = './imgs/train'
img_class_folder = os.listdir(img_path)

def get_train_list(img_path = './imgs/train'):
    '''
    This function is to get all the img files in a list under the img_path 
    '''
    img_class = os.listdir(img_path)
    img_list = {}
    for cls in img_class:
        cls_path = os.path.join(img_path,cls)
        img_list[cls] = [os.path.join(cls_path, img) for img in os.listdir(cls_path) if img.endswith('.jpg')]    
    
    return img_list
    

def load_imgs_to_array(img_list, scale = 4):
    '''
    Load all the training images into a numpy array according to img_list. Each
    image is flatten to a one dimensional array. The last element of each row
    is the label for that image. Image shape [640,480]. For memory issues, images
    are rescaled according to the desired scale.
    '''
    # Load and convert image to gray scale, and check the shape of the image. 
    img = numpy.array(Image.open(img_list.values()[0][0]).convert('L'))    
    img_arr = {}
    img_label = {}
    for cls in img_list:
        print "Loading class :", cls
        
        img_arr[cls] = numpy.empty((len(img_list[cls]),img.shape[0]/scale*img.shape[1]/scale),dtype='uint8')
        for i in range(len(img_list[cls])):
            img_arr[cls][i] = numpy.array(Image.open(img_list[cls][i]).resize((160,120),Image.ANTIALIAS).convert('L')).ravel()
        img_label[cls] = numpy.full((len(img_list[cls]),1),int(cls[1]),dtype = 'uint8')
        img_arr[cls] = numpy.hstack([img_arr[cls],img_label[cls]])

    # Stack all the classes together        
    flat_img_arr = numpy.vstack([img_arr[cls] for cls in img_list])   
    
    return flat_img_arr
    

def load_and_save_img_arr():
    '''
    This function load the images to array and save to disk for further training
    '''
    img_list = get_train_list()
    flat_img_arr = load_imgs_to_array(img_list)
    numpy.save('./flat_img_arr_small.npy',flat_img_arr)
    print "flat_img_arr saved!"
    return flat_img_arr


def get_test_list(img_path = './imgs/test'):
    '''
    This function is to get all the img files in a list under the test folder 
    '''
    test_imgs = os.listdir(img_path)
    test_list = [os.path.join(img_path, img) for img in test_imgs if img.endswith('.jpg')]        
    return test_list


def load_test_img(test_path):
    test_list = get_test_list(img_path = test_path)
    test_img = numpy.empty((len(test_list),160*120),dtype = 'uint8')
    for i in range(len(test_list)):
        test_img[i] = numpy.array(Image.open(test_list[i]).resize((120,160),Image.ANTIALIAS).convert('L')).ravel()
    return test_img    
        



#------------------------------ PCA and Training -----------------------------

if __name__ == '__main__':
    try:
        flat_img_arr = numpy.load('./flat_img_arr_small.npy')
    except: 
        flat_img_arr = load_and_save_img_arr()
    
    
    # Features X and Labels y are seperated    
    img_shape = [160,120]
    X = flat_img_arr[:,:-1]
    y = flat_img_arr[:,-1]
    
    num_features = X.shape[1]
    num_samples = X.shape[0]
    num_classes = len(numpy.unique(y))
    
    print "Total dataset size:"
    print "n_samples: %d" % num_samples
    print "n_features: %d" % num_features
    print "n_classes: %d" % num_classes

    # Split into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    num_comp = 50
        
    print "Extracting the top %d eigen_images from %d training images" % (num_comp, num_features)
    t0 = time()
    svd = TruncatedSVD(n_components=num_comp)
    svd.fit(X_train)
#    pca = RandomizedPCA(n_components = num_comp, whiten=True).fit(X_train)
    print "done in %0.3fs" % (time() - t0)    
    
    eigen_images = svd.components_.reshape((num_comp,img_shape[0], img_shape[1]))
    
        
    pg.image(eigen_images)

    print "Projecting the input data on the eigen_images orthonormal basis"
    t0 = time()
    X_train_pca = svd.transform(X_train)
    X_test_pca = svd.transform(X_test)
    print "done in %0.3fs" % (time() - t0)
    
    print "Fitting the classifier to the training set"
    
    t0 = time()
    
#    param_grid = {
#                 'C': [5,10,50,100,500],
#                  'gamma': [1e-8,5e-8,1e-7,5e-7],
#                  }
    param_grid = {
                 'C': [50],
                  'gamma': [5e-8],
                  }
   
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced',probability = True), param_grid)
    clf = clf.fit(X_train_pca, y_train)
    print "done in %0.3fs" % (time() - t0)
    print "Best estimator found by grid search:"
    print clf.best_estimator_
    
    print "Predict driver behavior classes: "
    t0 = time()
    y_pred = clf.predict(X_test_pca)
    y_pred2 = clf.predict(X_train_pca)
    print "done in %0.3fs" % (time() - t0)
    
    accuracy = accuracy_score(y_test,y_pred)
    print "Prediction Accuracy:", accuracy


# ------------------------------ Predict the test images --------------------------
      
#    img_test = numpy.array(Image.open('./imgs/test/img_102149.jpg').resize((120,160),Image.ANTIALIAS).convert('L')).ravel()
#    img_pca = svd.transform(img_test.reshape(1,-1))
#    pred = clf.predict(img_pca)
#    print pred





