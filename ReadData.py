# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 09:04:00 2021

@author: Gebruiker
"""
import cv2
import os
import glob
import matplotlib.pyplot as plt
import random

def load_images_from_folder(folder):
    '''Reads and displays all images in a given folder'''
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def load_a_few_images_from_folder(folder,amount_of_images=5):
    '''Generates a random list of indexes of lenth 5, where each random
    value is an integer between 0 and the amount of images in the folder.
    Next the images with these indexes are read and displayed'''
    #create random indexes list
    files = os.listdir(folder)
    number_files = len(files)
    randomlist = []
    for i in range(0,amount_of_images):
        n = random.randint(0,number_files)
        randomlist.append(n)
    #read images with random indexes from given folder
    images = []
    data_path = os.path.join(folder,'*g') 
    files = glob.glob(data_path)
    for i in randomlist:
        img = plt.imread(files[i]) 
        images.append(img)
    return images

def display_images(images,title):
    '''displays all images in one figure given a list of images as 
    input and a title for the figure'''
    fig, ax = plt.subplots(nrows=len(images), ncols=1, figsize=(20,20))
    fig.suptitle(title)
    for i in range(len(ax)):
        ax[i].imshow(images[i])
    plt.show()
        

train_0_images = load_a_few_images_from_folder(r'C:\Users\Gebruiker\Documents\TUe\Jaar 3\8P361\train\0')
train_1_images = load_a_few_images_from_folder(r'C:\Users\Gebruiker\Documents\TUe\Jaar 3\8P361\train\1')

display_images(train_0_images, 'train_images_zero')
display_images(train_1_images, 'train_images_one')

#aanpassing