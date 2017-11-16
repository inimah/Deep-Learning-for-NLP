# -*- coding: utf-8 -*-
#__author__ = "@tita"

from __future__ import print_function
import os
import sys
import h5py
#import _pickle as cPickle
import cPickle
import itertools

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from scipy import linalg,dot
from language_models import *
from gensim.models import Word2Vec, Doc2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback
from keras.models import load_model
from keras.models import Sequential, Model

np.random.seed([3,1415])

import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-batch_size', type=int, default=100)
ap.add_argument('-nb_epoch', type=int, default=100)
ap.add_argument('-mode', default='train')
args = vars(ap.parse_args())


BATCH_SIZE = args['batch_size']
NB_EPOCH = args['nb_epoch']
MODE = args['mode']



DATAPATH = 'data/airlines'
EMBEDDINGS = 'embeddings/w2v'
RESULTS ='evals'


# reading file in pickle format
def readPickle(pickleFilename):
	f = open(pickleFilename, 'rb')
	obj = cPickle.load(f)
	f.close()
	return obj

# saving file into pickle format
def savePickle(dataToWrite,pickleFilename):
	f = open(pickleFilename, 'wb')
	cPickle.dump(dataToWrite, f)
	f.close()

# saving file into hdf5 format
# only works for array with same length
################################################
def saveH5File(h5Filename,datasetName,dataToWrite):
		# h5Filename should be in format 'name-of-file.h5'
		# datasetName in String "" format
		with h5py.File(h5Filename, 'w') as hf:
				hf.create_dataset(datasetName,  data=dataToWrite)

# reading file in hdf5 format
# only works for array with same length
################################################
def readH5File(h5Filename,datasetName):
		# h5Filename should be in format 'name-of-file.h5'
		# datasetName in String "" format
		with h5py.File(h5Filename, 'r') as hf:
				data = hf[datasetName][:]
		return data

if __name__ == '__main__':

	class TrainingHistory(Callback):
		
		def on_train_begin(self, logs={}):
			self.losses = []
			self.acc = []
			self.i = 0
			self.save_every = 50
		def on_batch_end(self, batch, logs={}):
			self.losses.append(logs.get('loss'))
			self.acc.append(logs.get('acc'))
			self.i += 1
		
	history_train= TrainingHistory()
	
	
	vocab = readPickle(os.path.join(DATAPATH,'vocab'))

	labelled_docs  = readPickle(os.path.join(DATAPATH,'fin_labelled_strtweets'))

	tweets = []
	labels = []
	for i, data in enumerate(labelled_docs):
		text = data[0]
		label = data[2]
		tweets.append(text)
		labels.append(label)

		
	word_vector = readPickle(os.path.join(EMBEDDINGS,'w2v_embed1'))
	doc_vector = readPickle(os.path.join(EMBEDDINGS,'avg_embed1'))
	
	xy = list(zip(labels,tweets,doc_vector))
	xy_vec = np.array(xy,dtype=object)

	# shuffling data
	ind_rand = np.arange(len(xy_vec))
	np.random.shuffle(ind_rand)
	vec_data = xy_vec[ind_rand]

	# splitting data into training and testing sets
	nTrain = int(.8 * len(vec_data))
	trainDat = vec_data[:nTrain]
	testDat = vec_data[nTrain:]

	savePickle(trainDat, os.path.join(RESULTS,'w2v_avg_trainDat'))
	savePickle(testDat, os.path.join(RESULTS,'w2v_avg_testDat'))
	
	numClasses = 3
	
	xtrain = []
	ytrain = []
	for i, data in enumerate(trainDat):
		xtrain.append(data[2])
		ytrain.append(data[0])

	xtest = []
	ytest = []
	for i, data in enumerate(testDat):
		xtest.append(data[2])
		ytest.append(data[0])

	
	x_train = np.array(xtrain, dtype='float32')
	x_test = np.array(xtest, dtype='float32')
	
	# the following sklearn module will tranform nominal to numerical (0,1,2)
	numEncoder = LabelEncoder()
	numEncoder.fit(ytrain)
	y_train_num = numEncoder.transform(ytrain)
	# because our output is multiclass classification problems, 
	# we need to transform the class label into categorical encoding ([1,0,0],[0,1,0],[0,0,1])
	y_train_cat = to_categorical(y_train_num, num_classes=numClasses)

	numEncoder = LabelEncoder()
	numEncoder.fit(ytest)
	y_test_num = numEncoder.transform(ytest)
	# because our output is multiclass classification problems, 
	# we need to transform the class label into categorical encoding ([1,0,0],[0,1,0],[0,0,1])
	y_test_cat = to_categorical(y_test_num, num_classes=numClasses)

	
	############################

	# evaluate on MLP classifier
	mlp = mlpClassifier(x_train, numClasses)
	mlp.fit(x_train, y_train_cat, batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH, callbacks=[history_train])
	# verbose: 0 for no logging to stdout, 1 for progress bar logging, 2 for one log line per epoch
	score = mlp.evaluate(x_test, y_test_cat, verbose=2)

	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	savePickle(history_train.losses, os.path.join(RESULTS,'eval_mlp_w2v_avg.losses'))
	savePickle(history_train.acc,os.path.join(RESULTS,'eval_mlp_w2v_avg.acc'))
