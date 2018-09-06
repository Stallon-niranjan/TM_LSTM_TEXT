from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import _pickle as cPickle
import random, re, os, collections
import Config
import csv

from gensim.models import Word2Vec

#from gensim.models import FastText
config = Config.Config()

def Read_Train_Data(config):

    with open(os.path.join(config.data_dir, 'train_dataset.csv'), 'r', encoding='utf8', errors='ignore') as ftext:
        reader = pd.read_csv(ftext)
    # Cleaning the texts
        trainingdata = []
        doc = []
        keywords = []
        for i in range(0, reader.shape[0]):
            review = re.sub('[^a-zA-Z]', ' ', reader['Description'][i])
            review = review.lower()

            review = review.split()
            #review = [' '.join(review)]
            doc.append(review)
        
        #for i in range(0, reader.shape[0]):
            reviewq = re.sub('[^a-zA-Z]', ' ', reader['Product_name'][i])
            reviewq = reviewq.lower()

            reviewq = reviewq.split()
            #review = ' '.join(review)
            keywords.append(reviewq)
            
        trainingdata = doc + keywords    
    return trainingdata


data = Read_Train_Data(config) 


# Create CBOW model
model1 = Word2Vec(data, size=5, window=5, min_count=3, workers=4)
model1.train(data, total_examples=len(data), epochs=10)
model1.wv.save_word2vec_format('Data/vec_2X')
#model1.wv.save_word2vec_format(config.data_dir+"vec_1.txt", binary=False)
# Create Skip Gram model
model3 = Word2Vec(data, min_count=3, size=5, window=5,workers=4, sg = 1)
model3.train(data, total_examples=len(data), epochs=10)
model3.wv.save_word2vec_format('Data/vec_3X')