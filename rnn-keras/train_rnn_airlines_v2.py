# -*- coding: utf-8 -*-
#__author__ = "@tita"
#__update__ = "12.11.2017"
#__maintainer__ = "@tita"
#__email__ = "i.nimah@tue.nl"

from __future__ import print_function
import os
import sys
import numpy as np
#import _pickle as cPickle
import cPickle
import h5py

from language_models import *
from keras.callbacks import Callback
from keras.models import Sequential, Model
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical



import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-batch_size', type=int, default=100)
ap.add_argument('-nb_epoch', type=int, default=10)
ap.add_argument('-mode', default='train')
args = vars(ap.parse_args())


BATCH_SIZE = args['batch_size']
NB_EPOCH = args['nb_epoch']
MODE = args['mode']

DATAPATH = 'data/airlines'
W2V_PATH = 'embeddings/w2v'
RESULTS = 'embeddings/rnn_v2/run4'


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


# return a numeric sequence of words in a sentence  into one-hot vector
# this is for last layer with time distributed (3D dimension)
# the text input is whole text of documents (without splitting into sentences)
def matrixVectorization3D(docs,timesteps,vocab_length):
	# create zero matrix
	sequences = np.zeros((len(docs), timesteps, vocab_length))
	# the encoding here is also the index of word in vocabulary list as such 
	# it is the index of one hot vector with '1' value
	for i, text in enumerate(docs):
		for j, word_index in enumerate(text):
			sequences[i, j, word_index] = 1
	return sequences

if __name__ == '__main__':


	'''
	orig_stdout = sys.stdout
	f = open(os.path.join(RESULTS,'out_train_rnn.txt'),'w')
	sys.stdout = f
	'''
	

	class TrainingHistory(Callback):
		
		def on_train_begin(self, logs={}):
			self.losses = []
			self.acc = []
			'''
			self.crossentropy = []
			self.crossentropy2 = []
			self.perplexity = []
			self.perplexity2 = []
			'''
			self.i = 0
			#self.save_every = 50
		def on_batch_end(self, batch, logs={}):
			self.losses.append(logs.get('loss'))
			self.acc.append(logs.get('acc'))
			'''
			self.crossentropy.append(logs.get('crossentropy'))
			self.crossentropy2.append(logs.get('crossentropy2'))
			self.perplexity.append(logs.get('perplexity'))
			self.perplexity2.append(logs.get('perplexity2'))
			'''
			self.i += 1
		
	history = TrainingHistory()

	# reading stored pre-processed (in pickle format)

	vocab = readPickle(os.path.join(DATAPATH,'vocab'))
	# encoded version (sequences in integer format)
	labelled_docs = readPickle(os.path.join(DATAPATH,'fin_labelled_numtweets'))
	revertVocab = dict((v,k) for (k,v) in vocab.items())
	eof_token = revertVocab['EOF']

	tokenized_docs = []
	for i, data in enumerate(labelled_docs):
		text = data[0]
		tokenized_docs.append(text)


	# data structure for language modeling based on training
	# x_train is original sequence began with 'SOF' token (index = 0 in our vocab index)
	# y_train is original sequence ended with 'EOF' token (last index-1 in our vcab index)
	x_train = []
	y_train = []
	for i, tokens in enumerate(tokenized_docs):	
		x_tokens = list(tokens)
		y_tokens = list(tokens)
		# insert 'SOF' token which is encoded as 0 
		x_tokens.insert(0,0)
		x_train.append(x_tokens)
		# append 'EOF' token 
		y_tokens.append(eof_token)
		y_train.append(y_tokens)

	# saving data into pickle ...
	savePickle(x_train, os.path.join(RESULTS,'rnn_xtrain'))
	savePickle(y_train,os.path.join(RESULTS,'rnn_ytrain'))
	

	# use word embedding from previously trained skipgram model
	w2v_embed1 = readPickle(os.path.join(W2V_PATH,'w2v_embed1'))
	

	VOCAB_LENGTH = len(vocab)
	EMBEDDING_DIM = w2v_embed1.shape[1]
	MAX_SEQUENCE_LENGTH = 35

	print('[INFO] Zero padding...')
	X = pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH, dtype='int32')
	Y = pad_sequences(y_train, maxlen=MAX_SEQUENCE_LENGTH, dtype='int32')

	savePickle(X, os.path.join(RESULTS,'lm_paddedx_input'))
	savePickle(Y, os.path.join(RESULTS,'lm_paddedy_output'))

	model = languageModelBiLSTM2(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, w2v_embed1)

	# Encoding Y sequences and fitting data into model in batch

	k_start = 1 
	i_end = 0
	mini_batches = 1000
	loss = []
	acc = []
	'''
	crossentropy = []
	crossentropy2 = []
	perplexity = []
	perplexity2 = []
	'''

	for k in range(k_start, NB_EPOCH+1):
		# training and encoding documents in mini batches:
		i_loss = []
		i_acc = []
		'''
		i_crossentropy = []
		i_crossentropy2 = []
		i_perplexity = []
		i_perplexity2 = []
		'''
		for i in range(0, len(X), mini_batches):
			if i + mini_batches >= len(X):
				i_end = len(X)
			else:
				i_end = i + mini_batches

			# encoding y output as one hot vector with dimension size of vocabulary
			# Y_encoded = matrixVectorization3D(Y[i:i_end],MAX_SEQUENCE_LENGTH,VOCAB_LENGTH)

			# instead of encoding Y output sequence into one-hot encoding of vocabulary size,
			# we can encode it based on its word embedding matrix projection

			# embedding layer
			embedding_layer = Model(inputs=model.input, outputs=model.get_layer('embedding_layer').output)
			Y_encoded = embedding_layer.predict(Y[i:i_end]) 

			print('[INFO] Training model: epoch {}th {}/{} samples'. format(k,i,len(X)))
			model.fit(X[i:i_end], Y_encoded, batch_size=BATCH_SIZE, nb_epoch=1, verbose=2, callbacks=[history])

			i_loss.append(history.losses)
			i_acc.append(history.acc)
			'''
			i_crossentropy.append(history.crossentropy)
			i_crossentropy2.append(history.crossentropy2)
			i_perplexity.append(history.perplexity)
			i_perplexity2.append(history.perplexity2)
			'''
			


		model.save_weights(os.path.join(RESULTS, 'weights_lm_bilstm.hdf5_{}.hdf5'.format(k)))
		loss.append(i_loss)
		acc.append(i_acc)
		'''
		crossentropy.append(i_crossentropy)
		crossentropy2.append(i_crossentropy2)
		perplexity.append(i_perplexity)
		perplexity2.append(i_perplexity2)
		'''

	
	#saveH5File('lm_yencoded_output.h5','Y_encoded',Y_encoded)
	

	model.save(os.path.join(RESULTS,'lm_bilstm.h5'))
	
	savePickle(loss, os.path.join(RESULTS,'lm_bilstm_loss'))
	savePickle(acc, os.path.join(RESULTS,'lm_bilstm_accuracy'))

	'''
	savePickle(crossentropy, os.path.join(RESULTS,'crossentropy'))
	savePickle(crossentropy2, os.path.join(RESULTS,'crossentropy2'))
	savePickle(perplexity, os.path.join(RESULTS,'perplexity'))
	savePickle(perplexity2, os.path.join(RESULTS,'perplexity2'))
	'''

	# encoder layer
	encoder = Model(inputs=model.input, outputs=model.get_layer('bilstm_encoder').output)
	encoder_embeddings = encoder.predict(X)
	#savePickle(doc_embed_LM1b,'doc_embed_LM1b')
	saveH5File(os.path.join(RESULTS,'encoder_embeddings.h5'),'embedding',encoder_embeddings)

	# decoder
	decoder = Model(inputs=model.input, outputs=model.get_layer('bilstm_decoder_2').output)
	decoder_embeddings = decoder.predict(X)
	#savePickle(doc_sent_embed_LM1b,'doc_sent_embed_LM1b')
	saveH5File(os.path.join(RESULTS,'decoder_embeddings.h5'),'embedding',decoder_embeddings)

	'''
	sys.stdout = orig_stdout
	f.close()
	'''
	

