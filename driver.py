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

#
driver_list = pandas.read_csv('./driver_imgs_list.csv')

#------------------------ Functions to Convert Images to Arrays --------------

img_path = './imgs/train'
img_class_folder = os.listdir(img_path)

def get_img_list(img_path = './imgs/train'):
    '''
    This function is to get all the img files in a list under the img_path 
    '''
    img_class = os.listdir(img_path)
    img_list = {}
    for cls in img_class:
        cls_path = os.path.join(img_path,cls)
        img_list[cls] = [os.path.join(cls_path, img) for img in os.listdir(cls_path) if img.endswith('.jpg')]    
    
    return img_list
    

def load_imgs_to_array(img_list):
    '''
    Load all the training images into a numpy array according to img_list. Each
    image is flatten to a one dimensional array. The last element of each row
    is the label for that image. Image shape [640,480]
    '''
    # Load and convert image to gray scale, and check the shape of the image. 
    img = numpy.array(Image.open(img_list.values()[0][0]).convert('L'))    
    img_arr = {}
    img_label = {}
    for cls in img_list:
        print "Loading class :", cls
        img_arr[cls] = numpy.empty((len(img_list[cls]),img.shape[0]*img.shape[1]),dtype='uint8')
        for i in range(len(img_list[cls])):
            img_arr[cls][i] = numpy.array(Image.open(img_list[cls][i]).convert('L')).ravel()
        img_label[cls] = numpy.full((len(img_list[cls]),1),int(cls[1]),dtype = 'uint8')
        img_arr[cls] = numpy.hstack([img_arr[cls],img_label[cls]])

    # Stack all the classes together        
    flat_img_arr = numpy.vstack([img_arr[cls] for cls in img_list])   
    
    return flat_img_arr
    

def load_and_save_img_arr():
    '''
    This function load the images to array and save to disk for further training
    '''
    img_list = get_img_list()
    flat_img_arr = load_imgs_to_array(img_list)
    numpy.save('./flat_img_arr.npy',flat_img_arr)
    print "flat_img_arr saved!"
    return flat_img_arr


#------------------------------ PCA and Training -----------------------------

if __name__ == '__main__':
    try:
        flat_img_arr = numpy.load('./flat_img_arr.npy')
    except: 
        flat_img_arr = load_and_save_img_arr()
    
    # Features X and Labels y are seperated    
    img_shape = [640,480]
    X = flat_img_arr[:,:-1]
    y = flat_img_arr[:,-1]
    
    num_features = X.shape[1]
    num_samples = X.shape[0]
    num_classes = len(numpy.unique(y))
    
    print "Total dataset size:"
    print "n_samples: %d" % num_samples
    print "n_features: %d" % num_features
    print "n_classes: %d" % num_classes
    
    numpy.random.seed(42)
    # Split into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    num_comp = 200
    
    print "Extracting the top %d eigen_images from %d training images" % (num_comp, num_features)
    t0 = time()
    pca = RandomizedPCA(n_components = num_comp, whiten=True).fit(X_train)
    print "done in %0.3fs" % (time() - t0)
    
    eigen_images = pca.components_.reshape((num_comp,img_shape[0], img_shape[1]))
    
    print "Projecting the input data on the eigen_images orthonormal basis"
    t0 = time()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print "done in %0.3fs" % (time() - t0)
    
    print "Fitting the classifier to the training set"
    
    t0 = time()
    
    param_grid = {
                 'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
                  }
    
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(X_train_pca, y_train)
    print "done in %0.3fs" % (time() - t0)
    print "Best estimator found by grid search:"
    print clf.best_estimator_
    
    print "Predict driver behavior classes: "
    t0 = time()
    y_pred = clf.predict(X_test_pca)
    print "done in %0.3fs" % (time() - t0)
    
    accuracy = accuracy_score(y_test,y_pred)
    print "Prediction Accuracy:", accuracy




