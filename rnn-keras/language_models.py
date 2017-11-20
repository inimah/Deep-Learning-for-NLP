# -*- coding: utf-8 -*-
#__author__ = "@tita"
#__update__ = "01.07.2017"
#__maintainer__ = "@tita"
#__email__ = "i.nimah@tue.nl"

#from __future__ import print_function
import os
import sys
import numpy as np
import nltk
import math
from functools import partial
from gensim.models import Word2Vec, Doc2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
#import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import *
from keras.preprocessing import sequence
from keras.optimizers import Adam, RMSprop, SGD
from keras import optimizers
from keras.layers.merge import Concatenate
import keras.backend as K


# since we preserve all possible words/characters/information, we need to add regex in tokenizer of sklearn TfIdfVectorizer 
'''
pattern = r"""
 (?x)                   # set flag to allow verbose regexps
 (?:[A-Z]\.)+           # abbreviations, e.g. U.S.A.
 |\$?\d+(?:\.?,\d+)?%?       # numbers, incl. currency and percentages
 |\w+(?:[-']\w+)*       # words w/ optional internal hyphens/apostrophe
 |(?:[`'^~\":;,.?(){}/\\/+/\-|=#$%@&*\]\[><!])         # special characters with meanings
 """
 '''


################################################
# word2vec models
################################################
def wordEmbedding(documents, vocab, argsize, argiter):

	
	# description of parameters in gensim word2vec model

	#`sg` defines the training algorithm. By default (`sg=0`), CBOW is used. Otherwise (`sg=1`), skip-gram is employed.
	#`min_count` = ignore all words with total frequency lower than this
	#`max_vocab_size` = limit RAM during vocabulary building; if there are more unique words than this, then prune the infrequent ones. Every 10 million word types
	# need about 1GB of RAM. Set to `None` for no limit (default).
	#`workers` = use this many worker threads to train the model (=faster training with multicore machines).
	#`hs` = if 1, hierarchical softmax will be used for model training. If set to 0 (default), and `negative` is non-zero, negative sampling will be used.
	#`negative` = if > 0, negative sampling will be used, the int for negative specifies how many "noise words" should be drawn (usually between 5-20).
	# Default is 5. If set to 0, no negative samping is used.
	

	word2vec_models = [
	# skipgram model with hierarchical softmax and negative sampling
	Word2Vec(size=argsize, min_count=0, window=5, sg=1, hs=1, negative=5, iter=argiter),
	# cbow model with hierarchical softmax and negative sampling
	Word2Vec(size=argsize, min_count=0, window=5, sg=0, hs=1, negative=5, iter=argiter)

	]

	# model 1 = skipgram 
	model1 = word2vec_models[0]
	# model 2 = cbow
	model2 = word2vec_models[1]
 

	# building vocab for each model (creating 2, in case one model subsampling word vocabulary differently)
	model1.build_vocab(documents)
	model2.build_vocab(documents)

	# the following vocabulary index is built by word2vec so the order of word indexing is different with our original index
	# since python dictionary format also does not have a notion of order
	# as such when retrieving the resulting embedding, we have to make sure that
	# the weight vector refers to the same word index

	# example of data structure from word2vec gensim vocab

	'''
	list(w2v_1.wv.vocab.items())[:5]

	[('lined', <gensim.models.keyedvectors.Vocab at 0x7f5203b73748>),
	('couple', <gensim.models.keyedvectors.Vocab at 0x7f5203a5da20>),
	('polite', <gensim.models.keyedvectors.Vocab at 0x7f5203b737b8>),
	('prerecorded', <gensim.models.keyedvectors.Vocab at 0x7f5203a42c88>),
	('shade', <gensim.models.keyedvectors.Vocab at 0x7f5203b737f0>)]
	'''

	word2vec_vocab1 = dict([(v.index,k) for k, v in model1.wv.vocab.items()])   
	word2vec_vocab2 = dict([(v.index,k) for k, v in model2.wv.vocab.items()])

	revert_w2v_vocab1 = dict((v,k) for (k,v) in word2vec_vocab1.items())
	revert_w2v_vocab2 = dict((v,k) for (k,v) in word2vec_vocab2.items())

	embedding1 = np.zeros(shape=(len(vocab), argsize), dtype='float32')
	embedding2 = np.zeros(shape=(len(vocab), argsize), dtype='float32')

	print('Training word2vec model...')

	# number of tokens
	n_tokens = sum([len(sent) for sent in documents])
	# number of sentences/documents
	n_examples = len(documents)
	model1.train(documents, total_words=n_tokens, total_examples=n_examples, epochs=argiter)
	model2.train(documents, total_words=n_tokens, total_examples=n_examples, epochs=argiter)
	

	word2vec_weights1 = model1.wv.syn0
	word2vec_weights2 = model2.wv.syn0    

	for i, w in vocab.items():

		if w not in word2vec_vocab1.values():
			continue
		embedding1[i, :] = word2vec_weights1[revert_w2v_vocab1[w], :]

		if w not in word2vec_vocab2.values():
			continue
		embedding2[i, :] = word2vec_weights2[revert_w2v_vocab2[w], :]

	
	return model1, model2, embedding1, embedding2

################################################
# doc2vec models
################################################
def docEmbedding(documents, vocab, argsize, argiter):


	
	# doc2vec models
	doc2vec_models = [
	# PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
	Doc2Vec(dm=1, dm_concat=1, size=argsize, window=5, negative=5, hs=1, min_count=0, alpha=0.025, min_alpha=0.025),
	# PV-DBOW 
	Doc2Vec(dm=0, size=argsize, negative=5, hs=1, min_count=0, alpha=0.025, min_alpha=0.025),
	# PV-DM w/average
	Doc2Vec(dm=1, dm_mean=1, size=argsize, window=5, negative=5, hs=1, min_count=0, alpha=0.025, min_alpha=0.025),
	]

	model1 = doc2vec_models[0]
	model2 = doc2vec_models[1]
	model3 = doc2vec_models[2]

	model1.build_vocab(documents)
	model2.build_vocab(documents)
	model3.build_vocab(documents)


	doc2vec_vocab1 = dict([(v.index,k) for k, v in model1.wv.vocab.items()])   
	doc2vec_vocab2 = dict([(v.index,k) for k, v in model2.wv.vocab.items()])
	doc2vec_vocab3 = dict([(v.index,k) for k, v in model3.wv.vocab.items()])

	print('Training doc2vec model...')

	# number of tokens
	n_tokens = sum([len(sent) for sent in documents])
	# number of sentences/documents
	n_examples = len(documents)

	model1.train(documents, total_examples=n_examples, epochs=argiter)
	model2.train(documents, total_examples=n_examples, epochs=argiter)
	model3.train(documents, total_examples=n_examples, epochs=argiter)

	doc2vec_wv1 = model1.wv.syn0
	doc2vec_wv2 = model2.wv.syn0
	doc2vec_wv3 = model3.wv.syn0


	doc2vec_weights1 = np.array(model1.docvecs)
	doc2vec_weights2 = np.array(model2.docvecs)
	doc2vec_weights3 = np.array(model3.docvecs)

	return model1, model2, model3, doc2vec_weights1, doc2vec_weights2, doc2vec_weights3, doc2vec_vocab1, doc2vec_vocab2, doc2vec_vocab3, doc2vec_wv1, doc2vec_wv2, doc2vec_wv3


################################################
# generating sentence-level / document embedding by averaging word2vec
# document here is sentence - or sequence of words
################################################
def averageWE(w2v_weights, vocab, documents):

	#w2v_vocab = word2vec_model.wv.index2word
	#w2v_weights = word2vec_model.wv.syn0
	w2v_vocab = list(vocab.values())
	w2v = dict(zip(w2v_vocab, w2v_weights))
	dim = w2v_weights.shape[1]

	doc_embedding = []

	for i,text in enumerate(documents):
		embedding = np.mean([w2v[w] for w in text if w in w2v]
			or [np.zeros(dim)], axis=0)
		
		doc_embedding.append(embedding)

	return np.array(doc_embedding)

################################################
# generating sentence-level / document embedding by averaging word2vec and Tf-Idf penalty
# document here is sentence - or sequence of words
################################################

def countFrequency(word, doc):
	return doc.count(word)

def docFrequency(word, list_of_docs):
	count = 0
	for document in list_of_docs:
		if countFrequency(word, document) > 0:
			count += 1
	return 1 + count

def computeIDF(word, list_of_docs):

	# idf(term) = ( log ((1 + nd)/(1 + df(doc,term))) ) 
	# where nd : number of document in corpus; 
	# df : doc frequency (number of documents containing term)

	idf = math.log( (1 + len(list_of_docs)) / float(docFrequency(word, list_of_docs)))

	return idf

def averageIdfWE(w2v_weights, vocab, documents):

	#w2v_vocab = word2vec_model.wv.index2word
	#w2v_weights = word2vec_model.wv.syn0


	
	print('calculating Tf-Idf weights...')


	w2v_vocab = list(vocab.values())
	w2v = dict(zip(w2v_vocab, w2v_weights))
	dim = w2v_weights.shape[1]

	wordIdf = {}

	for i,txt in enumerate(w2v_vocab): 
		wordIdf[txt] = computeIDF(txt, documents)

	doc_embedding = []
	for i,text in enumerate(documents):
		embedding = np.mean([w2v[w] * wordIdf[w]
				for w in text if w in w2v] or
				[np.zeros(dim)], axis=0)
		
		doc_embedding.append(embedding)

	return np.array(doc_embedding)



# evaluate document/word vector
def mlpClassifier(embeddings, NUM_CLASSES):

	N_SAMPLES = embeddings.shape[0]
	EMBEDDING_DIM = embeddings.shape[1]
	hidden_size = 50

	latent_vector = Input(shape=(EMBEDDING_DIM,), name='latent_vector')
	dense_layer = Dense(hidden_size, activation='relu', name='dense_layer')(latent_vector)
	prediction = Dense(NUM_CLASSES, activation='softmax', name='prediction')(dense_layer)
	model = Model(latent_vector, prediction)
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())

	'''
	_________________________________________________________________
	_________________________________________________________________
	Layer (type)                 Output Shape              Param #
	=================================================================
	latent_vector (InputLayer)   (None, 50)                0
	_________________________________________________________________
	dense_layer (Dense)          (None, 50)                2550
	_________________________________________________________________
	prediction (Dense)           (None, 3)                 153
	=================================================================
	Total params: 2,703
	Trainable params: 2,703
	Non-trainable params: 0


	'''
	return model

def mlpSeqClassifier(embeddings, NUM_CLASSES):

	EMBEDDING_DIM = embeddings.shape[1]
	hidden_size = 50

	model = Sequential()
	model.add(Dense(hidden_size, input_dim=EMBEDDING_DIM, activation='relu', name='dense_layer'))
	model.add(Dropout(0.5))
	model.add(Dense(NUM_CLASSES, activation='softmax', name='prediction'))

	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())

	return model


################################################
# Neural Language Model (Sequence prediction)
# encoder - decoder architecture
# objective: predicts the next word sequence (target word) based on previous context words 
# sentence / document embedding is retrieved from the encoder part

################################################

# NOT BEING USED AT THE MOMENT
# with keras sequential model
# full encoder - decoder model 
def languageModel(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights):

	hidden_size = 50
	num_layers = 3

	model = Sequential()
	
	model.add(Embedding(VOCAB_LENGTH, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, trainable = True, mask_zero=True, weights=[embedding_weights], name='embedding_layer'))

	# Creating encoder network
	# encoding text input (sequence of words) into sentence embedding
	model.add(LSTM(hidden_size,name='lstm_enc'))
	model.add(RepeatVector(MAX_SEQUENCE_LENGTH))

	# Creating decoder network
	# objective function: predicting next words (language model)
	for i in range(num_layers):
		model.add(LSTM(hidden_size, name='lstm_dec%s'%(i+1), return_sequences=True))
	model.add(TimeDistributed(Dense(VOCAB_LENGTH), name='td_output'))
	model.add(Activation('softmax', name='last_output'))
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())

	return model

# USE THIS ONE (lm1a model)
# with keras functional API
# full encoder - decoder model 
# apply for TimeDistributed layer

def languageModelLSTM(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights):

	
	sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name='sequence_input', dtype='int32')
	embedded_layer = Embedding(VOCAB_LENGTH, EMBEDDING_DIM, weights=[embedding_weights], trainable = True, mask_zero=True, name='embedding_layer')(sequence_input)
	lstm_encoder = LSTM(EMBEDDING_DIM, name='lstm_encoder')(embedded_layer)
	encoder_repeat = RepeatVector(MAX_SEQUENCE_LENGTH,name='encoder_repeat')(lstm_encoder)
	# Creating decoder network
	# objective function: predicting next words (language model)
	lstm_decoder = LSTM(EMBEDDING_DIM, return_sequences=True,name='lstm_decoder')(encoder_repeat)
	prediction = TimeDistributed(Dense(VOCAB_LENGTH, activation='softmax'), name='td_prediction')(lstm_decoder)
	model = Model(sequence_input, prediction)

	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())

	'''
	_________________________________________________________________
	_________________________________________________________________
	Layer (type)                 Output Shape              Param #
	=================================================================
	sequence_input (InputLayer)  (None, 25)                0
	_________________________________________________________________
	embedding_layer (Embedding)  (None, 25, 50)            140700
	_________________________________________________________________
	lstm_encoder (LSTM)          (None, 50)                20200
	_________________________________________________________________
	encoder_repeat (RepeatVector (None, 25, 50)            0
	_________________________________________________________________
	lstm_decoder_2 (LSTM)        (None, 25, 50)            20200
	_________________________________________________________________
	td_prediction (TimeDistribut (None, 25, 2814)          143514
	=================================================================
	Total params: 324,614
	Trainable params: 324,614
	Non-trainable params: 0



	'''

	return model

# add perplexity metrics evaluation

def crossentropy(y_true, y_pred):
	
	return K.categorical_crossentropy(y_true, y_pred)


def crossentropy2(y_true, y_pred):

	return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,
													   logits=y_pred))



def perplexity(y_true, y_pred):

	cross_entropy = K.categorical_crossentropy(y_true, y_pred)
	perplexity = K.pow(2.0, cross_entropy)
	return perplexity

def perplexity2(y_true, y_pred):

	cross_entropy = crossentropy2(y_true, y_pred)
	perplexity = K.pow(2.0, cross_entropy)
	return perplexity




# USE THIS ONE (lm1b model)
# with Bidirectional LSTM
def languageModelBiLSTM(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights):


	sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name='sequence_input', dtype='int32')
	embedded_layer = Embedding(VOCAB_LENGTH, EMBEDDING_DIM, weights=[embedding_weights], trainable = True, mask_zero=True, name='embedding_layer')(sequence_input)
	bilstm_encoder = Bidirectional(LSTM(EMBEDDING_DIM),name='bilstm_encoder')(embedded_layer)
	encoder_repeat = RepeatVector(MAX_SEQUENCE_LENGTH,name='encoder_repeat')(bilstm_encoder)
	# Creating decoder network
	# objective function: predicting next words (language model)
	bilstm_decoder = Bidirectional(LSTM(EMBEDDING_DIM, return_sequences=True),name='bilstm_decoder')(encoder_repeat)
	prediction = TimeDistributed(Dense(VOCAB_LENGTH, activation='softmax'),name='td_prediction')(bilstm_decoder)
	model = Model(sequence_input, prediction)

	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())

	'''
	Layer (type)                 Output Shape              Param #
	=================================================================
	sequence_input (InputLayer)  (None, 35)                0
	_________________________________________________________________
	embedding_layer (Embedding)  (None, 35, 50)            474300    
	_________________________________________________________________
	bilstm_encoder (Bidirectiona (None, 50)                15200     
	_________________________________________________________________
	encoder_repeat (RepeatVector (None, 35, 50)            0         
	_________________________________________________________________
	bilstm_decoder_2 (Bidirectio (None, 35, 50)            15200     
	_________________________________________________________________
	td_prediction (TimeDistribut (None, 35, 9486)          483786    
	=================================================================
	Total params: 988,486
	Trainable params: 988,486
	Non-trainable params: 0
	_________________________________________________________________
	None


	'''

	return model

# the main different with previous model is this one use embedding dimension instead of vocabulary size
# to reduce computation resource

	
def languageModelBiLSTM2(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights):

	

	sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name='sequence_input', dtype='int32')
	embedded_layer = Embedding(VOCAB_LENGTH, EMBEDDING_DIM, weights=[embedding_weights], trainable = True, mask_zero=False, name='embedding_layer')(sequence_input)
	bilstm_encoder = Bidirectional(LSTM(EMBEDDING_DIM),name='bilstm_encoder')(embedded_layer)
	encoder_repeat = RepeatVector(MAX_SEQUENCE_LENGTH,name='encoder_repeat')(bilstm_encoder)
	# Creating decoder network
	# objective function: predicting next words (language model)
	bilstm_decoder = Bidirectional(LSTM(EMBEDDING_DIM, return_sequences=True),name='bilstm_decoder')(encoder_repeat)
	prediction = TimeDistributed(Dense(EMBEDDING_DIM, activation='softmax'),name='td_prediction')(bilstm_decoder)
	model = Model(sequence_input, prediction)

	model.compile(loss='cosine_proximity', optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())

	return model



# USE THIS ONE (lm1c model)
def languageModelGRU(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights):

	sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name='sequence_input', dtype='int32')
	embedded_layer = Embedding(VOCAB_LENGTH, EMBEDDING_DIM, weights=[embedding_weights], trainable = True, mask_zero=True, name='embedding_layer')(sequence_input)
	gru_encoder = GRU(EMBEDDING_DIM, name='gru_encoder')(embedded_layer)
	encoder_repeat = RepeatVector(MAX_SEQUENCE_LENGTH,name='encoder_repeat')(gru_encoder)
	# Creating decoder network
	# objective function: predicting next words (language model)
	gru_decoder = GRU(EMBEDDING_DIM, return_sequences=True,name='gru_decoder')(encoder_repeat)
	prediction = TimeDistributed(Dense(VOCAB_LENGTH, activation='softmax'), name='td_prediction')(gru_decoder)
	model = Model(sequence_input, prediction)

	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())

	return model

# USE THIS ONE (lm1d model)
def languageModelBiGRU(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights):



	sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name='sequence_input', dtype='int32')
	embedded_layer = Embedding(VOCAB_LENGTH, EMBEDDING_DIM, weights=[embedding_weights], trainable = True, mask_zero=True, name='embedding_layer')(sequence_input)
	bigru_encoder = Bidirectional(GRU(EMBEDDING_DIM),name='bigru_encoder')(embedded_layer)
	encoder_repeat = RepeatVector(MAX_SEQUENCE_LENGTH,name='encoder_repeat')(bigru_encoder)
	# Creating decoder network
	# objective function: predicting next words (language model)
	bigru_decoder = Bidirectional(GRU(EMBEDDING_DIM, return_sequences=True),name='bigru_decoder')(encoder_repeat)
	prediction = TimeDistributed(Dense(VOCAB_LENGTH, activation='softmax'),name='td_prediction')(bigru_decoder)
	model = Model(sequence_input, prediction)

	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())

	return model


# WON'T USE THIS ONE
# THIS IS NOT POSSIBLE 
# (OK IT'S POSSIBLE - BUT LAST LAYER WILL ONLY PREDICT WORDS INSIDE THAT SEQUENCE NOT FROM THE VOCABULARY LIST)
# Since the output should be in 3D matrix shape e.g. (9326, 25, 2814)
# except that we merge/concatenate the one hot vector 

# with keras functional API
# only encoder
# apply for Dense layer

def languageModelLSTMDense(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights):

	
	sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
	embedded_layer = Embedding(VOCAB_LENGTH, EMBEDDING_DIM, weights=[embedding_weights], trainable = True, mask_zero=True, name='embedding_layer')(sequence_input)
	encoder_layer = LSTM(EMBEDDING_DIM, name='lstm_enc')(embedded_layer)
	prediction = Dense(VOCAB_LENGTH, activation='softmax', name='dense_output')(encoder_layer)
	model = Model(sequence_input, prediction)

	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())

	'''
	_________________________________________________________________
	Layer (type)                 Output Shape              Param #
	=================================================================
	input_1 (InputLayer)         (None, 25)                0
	_________________________________________________________________
	embedding_layer (Embedding)  (None, 25, 50)            140700
	_________________________________________________________________
	lstm_enc (LSTM)              (None, 50)                20200
	_________________________________________________________________
	dense_output (Dense)         (None, 2814)              143514
	=================================================================
	Total params: 304,414
	Trainable params: 304,414
	Non-trainable params: 0
	_________________________________________________________________
	None


	'''

	return model


# WON'T USE THIS ONE
# THIS IS NOT POSSIBLE 
# (OK IT'S POSSIBLE - BUT LAST LAYER WILL ONLY PREDICT WORDS INSIDE THAT SEQUENCE NOT FROM THE VOCABULARY LIST)
# Since the output should be in 3D matrix shape e.g. (9326, 25, 2814)
# except that we merge/concatenate the one hot vector 
# with bidirectional LSTM
def languageModelBiLSTMDense(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights):

	hidden_size = 25

	sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
	embedded_layer = Embedding(VOCAB_LENGTH, EMBEDDING_DIM, weights=[embedding_weights], trainable = True, mask_zero=True, name='embedding_layer')(sequence_input)
	encoder_layer = Bidirectional(LSTM(hidden_size),name='bilstm_enc')(embedded_layer)
	prediction = Dense(VOCAB_LENGTH, activation='softmax', name='dense_output')(encoder_layer)
	model = Model(sequence_input, prediction)

	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())

	'''
	________________________________________________________________
	_________________________________________________________________
	Layer (type)                 Output Shape              Param #
	=================================================================
	input_1 (InputLayer)         (None, 25)                0
	_________________________________________________________________
	embedding_layer (Embedding)  (None, 25, 50)            140700
	_________________________________________________________________
	bilstm_enc (Bidirectional)   (None, 50)                15200
	_________________________________________________________________
	dense_output (Dense)         (None, 2814)              143514
	=================================================================
	Total params: 299,414
	Trainable params: 299,414
	Non-trainable params: 0
	_________________________________________________________________
	None

	'''


	return model

# with keras functional API

################################################
# Neural Classification Model 
# encoder - decoder architecture
# objective: predicts class label of input
# sentence / document embedding is retrieved from the encoder part

################################################

def classificationModelGRUDense(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, NUM_CLASSES, EMBEDDING_DIM, embedding_weights):

	hidden_size = 50
	num_layers = 3

	sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
	embedded_layer = Embedding(VOCAB_LENGTH, EMBEDDING_DIM, weights=[embedding_weights], trainable = True, mask_zero=True, name='embedding_layer')(sequence_input)
	gru_encoder = GRU(hidden_size, name='gru_encoder')(embedded_layer)
	prediction = Dense(NUM_CLASSES, activation='softmax', name='dense_output')(gru_encoder)
	model = Model(sequence_input, prediction)

	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())

	'''
	_________________________________________________________________
	

	'''

	return model

def classificationModelBiGRUDense(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, NUM_CLASSES, EMBEDDING_DIM, embedding_weights):

	hidden_size = 50
	num_layers = 3

	sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
	embedded_layer = Embedding(VOCAB_LENGTH, EMBEDDING_DIM, weights=[embedding_weights], trainable = True, mask_zero=True, name='embedding_layer')(sequence_input)
	bigru_encoder = Bidirectional(GRU(hidden_size),name='bigru_encoder')(embedded_layer)
	prediction = Dense(NUM_CLASSES, activation='softmax', name='dense_output')(bigru_encoder)
	model = Model(sequence_input, prediction)

	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())

	'''
	_________________________________________________________________
	

	'''

	return model

def classificationModelLSTMDense(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, NUM_CLASSES, EMBEDDING_DIM, embedding_weights):

	hidden_size = 50

	sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
	embedded_layer = Embedding(VOCAB_LENGTH, EMBEDDING_DIM, weights=[embedding_weights], trainable = True, mask_zero=True, name='embedding_layer')(sequence_input)
	lstm_encoder = LSTM(hidden_size, name='lstm_encoder')(embedded_layer)
	prediction = Dense(NUM_CLASSES, activation='softmax', name='dense_output')(lstm_encoder)
	model = Model(sequence_input, prediction)

	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())

	'''
	_________________________________________________________________
	Layer (type)                 Output Shape              Param #
	=================================================================
	input_1 (InputLayer)         (None, 25)                0
	_________________________________________________________________
	embedding_layer (Embedding)  (None, 25, 50)            140700
	_________________________________________________________________
	lstm_encoder (LSTM)          (None, 50)                20200
	_________________________________________________________________
	dense_output (Dense)         (None, 3)                 153
	=================================================================
	Total params: 161,053
	Trainable params: 161,053
	Non-trainable params: 0
	_________________________________________________________________
	'''

	return model


def classificationModelBiLSTMDense(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, NUM_CLASSES, EMBEDDING_DIM, embedding_weights):

	hidden_size = 25

	sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
	embedded_layer = Embedding(VOCAB_LENGTH, EMBEDDING_DIM, weights=[embedding_weights], trainable = True, mask_zero=True, name='embedding_layer')(sequence_input)
	bilstm_encoder = Bidirectional(LSTM(hidden_size),name='bilstm_encoder')(embedded_layer)
	prediction = Dense(NUM_CLASSES, activation='softmax', name='dense_output')(bilstm_encoder)
	model = Model(sequence_input, prediction)

	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())
	'''
	_________________________________________________________________
	Layer (type)                 Output Shape              Param #
	=================================================================
	input_1 (InputLayer)         (None, 25)                0
	_________________________________________________________________
	embedding_layer (Embedding)  (None, 25, 50)            140700
	_________________________________________________________________
	bilstm_encoder (Bidirectiona (None, 50)                15200
	_________________________________________________________________
	dense_output (Dense)         (None, 3)                 153
	=================================================================
	Total params: 156,053
	Trainable params: 156,053
	Non-trainable params: 0
	_________________________________________________________________

	'''


	return model

################################################
# Neural Translation Model 
# encoder - decoder architecture
# objective: predicts words in target language given sequence of words in source language
# sentence / document embedding is retrieved from the encoder part

################################################
def translationModel(X_vocab_len, X_max_len, y_vocab_len, y_max_len, EMBEDDING_DIM, embedding_weights):

	hidden_size = 200
	num_layers = 3

	model = Sequential()
	
	model.add(Embedding(X_vocab_len, EMBEDDING_DIM, input_length=X_max_len, mask_zero=True, weights=[embedding_weights], name='embedding_layer'))
	# Creating encoder network
	model.add(LSTM(hidden_size,name='lstm_enc_1'))
	model.add(RepeatVector(y_max_len))

	# Creating decoder network
	for i in range(num_layers):
		model.add(LSTM(hidden_size, name='lstm_%s'%(i+2), return_sequences=True))
	model.add(TimeDistributed(Dense(y_vocab_len,name='dense')))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())


	return model


################################################
# Hierarchical Neural Language Model 
# with encoder - decoder type architecture
#
# input array for this model is in 4D shape
# ( rows,       cols,     time_steps,    n_dim     )
# ( _____   ___________   _________   ____________ )
# ( n_docs, n_sentences,   n_words    dim_embedding)
################################################

# full encoder - decoder

def hierarchyLanguage1(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights):

	hidden_size = 50
	num_layers = 3

	# word-level input (sequences of words in current sentence)
	sentences_input = Input(shape=(MAX_SEQUENCE_LENGTH, ), name='sequence_input', dtype='int32')
	embedded_sentences = Embedding(VOCAB_LENGTH, EMBEDDING_DIM, weights=[embedding_weights], trainable = True, mask_zero=True, name='embedding_layer')(sentences_input)
	sent_lstm_encoder = LSTM(hidden_size, name='sent_lstm_encoder')(embedded_sentences)
	sent_model = Model(sentences_input, sent_lstm_encoder)
	# sentence-level input (sequences of sentences)
	doc_input = Input(shape=(MAX_SENTS,MAX_SEQUENCE_LENGTH), name='doc_input', dtype='int32')
	# sentence vector
	sent_vector = TimeDistributed(sent_model, name='sent_vector')(doc_input)
	# document vector
	doc_lstm_encoder = LSTM(col_hidden_size, name='doc_lstm_encoder')(sent_vector)
	# output for sentence vector

	sent_output = TimeDistributed(Dense(VOCAB_LENGTH, activation='softmax'), name='sent_output')(sent_vector)
	doc_output = Dense(VOCAB_LENGTH, activation='softmax', name='doc_output')(encoded_docs)
	
	model = Model(inputs=[doc_input, sent_vector], outputs=[doc_output, sent_output])
	

	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())
	return model


def hierarchyLanguageLSTMTD(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights):

	row_hidden_size = 100
	col_hidden_size = 100
	num_layers = 3


	sentences_input = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int32')
	# embedding layer
	embedded_sentences = Embedding(VOCAB_LENGTH, EMBEDDING_DIM, weights=[embedding_weights], trainable = True, mask_zero=True, name='embedding_layer')(sentences_input)
	# Encoder model
	# Encodes sentences
	sent_lstm = LSTM(row_hidden_size, name='lstm_enc_1')(embedded_sentences)
	sent_model = Model(sentences_input, sent_lstm)

	doc_input = Input(shape=(MAX_SENTS,MAX_SEQUENCE_LENGTH), dtype='int32')

	encoded_sentences = TimeDistributed(sent_model)(doc_input)
	
	# Encodes documents
	encoded_docs = LSTM(col_hidden_size, name='lstm_enc_2')(encoded_sentences)

	# Decoder model
	encoder_output = RepeatVector(MAX_SEQUENCE_LENGTH,name='encoder_repeat')(encoded_docs)

	# Creating decoder network
	# objective function: predicting next words (language model)
	for i in range(num_layers):
		decoder_layer = LSTM(col_hidden_size, return_sequences=True,name='lstm_dec_%s'%(i+2))(encoder_output)

	decoder = TimeDistributed(Dense(VOCAB_LENGTH, activation='softmax', name='dense_output'))(decoder_layer)

	model = Model(doc_input, decoder)

	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())

	'''
_________________________________________________________________
	Layer (type)                 Output Shape              Param #
	=================================================================
	input_2 (InputLayer)         (None, 15, 100)           0
	_________________________________________________________________
	time_distributed_1 (TimeDist (None, 15, 100)           8136700
	_________________________________________________________________
	lstm_enc_2 (LSTM)            (None, 100)               80400
	_________________________________________________________________
	encoder_repeat (RepeatVector (None, 100, 100)          0
	_________________________________________________________________
	lstm_dec_3 (LSTM)            (None, 100, 100)          80400
	_________________________________________________________________
	time_distributed_2 (TimeDist (None, 100, 80563)        8136863
	=================================================================
	Total params: 16,434,363
	Trainable params: 16,434,363
	Non-trainable params: 0
	_________________________________________________________________
	None


	'''


	return model



# only encoder (no decoder layer)
# and dense layer
def hierarchyLanguageLSTMDense(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights):

	hidden_size = 100

	# this will process sequence of words in a current sentence 
	# LSTM layer for this input --> predicts word based on preceding words --> this will represent our sentence vector
	# P(wt | w1...wt-1) or in another words P(sentence) = P(w1...wt-1)
	sentences_input = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int32')
	embedded_sentences = Embedding(VOCAB_LENGTH, EMBEDDING_DIM, weights=[embedding_weights], trainable = True, mask_zero=True, name='embedding_layer')(sentences_input)
	sent_lstm = LSTM(hidden_size, name='lstm_enc_1')(embedded_sentences)
	sent_model = Model(sentences_input, sent_lstm)

	# this will process sequence of sentences as part of a document
	# representing document vector
	# P(St | S1..St-1)
	doc_input = Input(shape=(MAX_SENTS,MAX_SEQUENCE_LENGTH), dtype='int32')
	# the following TimeDistributed layer will link LSTM learning of sentence model (as joint distribution of words in sentence) to document model (as joint distribution of sentences)
	encoded_sentences = TimeDistributed(sent_model)(doc_input)
	encoded_docs = LSTM(hidden_size, name='lstm_enc_2')(encoded_sentences)
	
	# the class label here is words in fixed vocabulary list
	prediction = Dense(VOCAB_LENGTH, activation='softmax', name='dense_output')(encoded_docs)
	model = Model(doc_input, prediction)

	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())

	'''
	_________________________________________________________________
	Layer (type)                 Output Shape              Param #
	=================================================================
	input_2 (InputLayer)         (None, 15, 100)           0
	_________________________________________________________________
	time_distributed_1 (TimeDist (None, 15, 100)           8136700
	_________________________________________________________________
	lstm_enc_2 (LSTM)            (None, 100)               80400
	_________________________________________________________________
	dense_output (Dense)         (None, 80563)             8136863
	=================================================================
	Total params: 16,353,963
	Trainable params: 16,353,963
	Non-trainable params: 0
	_________________________________________________________________
	None


	'''


	return model

################################################
# Hierarchical Neural Classification Model 
#
# input array for this model is in 4D shape (after embedding layer)
# ( cols,       rows,     time_steps,    n_dim     )
# ( _____   ___________   _________   ____________ )
# ( n_docs, n_sentences,   n_words    dim_embedding)
################################################


# !!!!NOT BEING USED!!!!
# with time distributed layer to get the projection of vector on class labels 
# include latent vector dimension in the last layer (, ndim, class size) instead of just (, class size)
def hierarchyClassifier1(MAX_SENTENCES, MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights, num_classes):

	
	row_hidden_size = 100
	col_hidden_size = 100

	if num_classes > 2:
		loss_function = 'categorical_crossentropy'
	else:
		loss_function = 'binary_crossentropy'

	sentences_input = Input(shape=(MAX_SENTENCES,MAX_SEQUENCE_LENGTH,), dtype='int32')
	embedded_sentences = Embedding(VOCAB_LENGTH, EMBEDDING_DIM, weights=[embedding_weights], trainable = True, mask_zero=True, name='embedding_layer')(sentences_input)
	lstm_sentence = TimeDistributed(LSTM(row_hidden_size,name='lstm_enc_1'))(embedded_sentences)
	
	docs_model = LSTM(col_hidden_size,name='lstm_enc_2', return_sequences=True)(lstm_sentence)

	# Prediction
	prediction = TimeDistributed(Dense(num_classes, activation='softmax', name='dense_out'))(docs_model)
	model = Model(sentences_input, prediction)
	model.compile(loss=loss_function, optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())

	'''
	Layer (type)                 Output Shape              Param #   
	=================================================================
	input_2 (InputLayer)         (None, 15, 100)           0         
	_________________________________________________________________
	time_distributed_1 (TimeDist (None, 15, 100)           8136700   
	_________________________________________________________________
	lstm_enc_2 (LSTM)            (None, 15, 100)           80400     
	_________________________________________________________________
	time_distributed_2 (TimeDist (None, 15, 80563)         8136863   
	=================================================================
	Total params: 16,353,963
	Trainable params: 16,353,963
	Non-trainable params: 0
	_________________________________________________________________
	None

	'''


	return model


# USE THIS
# with dense layer
def hierarchyClassifierLSTM(MAX_SENTENCES, MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights, num_classes):


	hidden_size = 50

	if num_classes > 2:
		loss_function = 'categorical_crossentropy'
	else:
		loss_function = 'binary_crossentropy'

	sentences_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name='sentences_input', dtype='int32')
	embedded_sentences = Embedding(VOCAB_LENGTH, EMBEDDING_DIM, weights=[embedding_weights], trainable = True, mask_zero=True, name='embedding_layer')(sentences_input)
	lstm_encoder = LSTM(hidden_size,name='lstm_encoder')(embedded_sentences)
	sentences_model = Model(sentences_input, lstm_encoder)

	docs_input = Input(shape=(MAX_SENTENCES, MAX_SEQUENCE_LENGTH), name='docs_input', dtype='int32')
	sentence_vector = TimeDistributed(sentences_model,name='sentence_vector')(docs_input)
	#dropout = Dropout(0.2)(docs_encoded)
	docs_encoder = LSTM(hidden_size,name='docs_encoder')(sentence_vector)

	# Prediction
	prediction = Dense(num_classes, activation='softmax', name='dense_out')(docs_encoder)
	model = Model(docs_input, prediction)
	model.compile(loss=loss_function, optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())

	'''
	_________________________________________________________________
	Layer (type)                 Output Shape              Param #
	=================================================================
	docs_input (InputLayer)      (None, 100, 25)           0
	_________________________________________________________________
	sentence_vector (TimeDistrib (None, 100, 50)           1989650
	_________________________________________________________________
	docs_encoder (LSTM)          (None, 50)                20200
	_________________________________________________________________
	dense_out (Dense)            (None, 3)                 153
	=================================================================
	Total params: 2,010,003
	Trainable params: 2,010,003
	Non-trainable params: 0
	_________________________________________________________________
	None


	'''


	return model


def hierarchyClassifierBiLSTM(MAX_SENTENCES, MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights, num_classes):


	hidden_size = 50

	if num_classes > 2:
		loss_function = 'categorical_crossentropy'
	else:
		loss_function = 'binary_crossentropy'

	sentences_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name='sentences_input', dtype='int32')
	embedded_sentences = Embedding(VOCAB_LENGTH, EMBEDDING_DIM, weights=[embedding_weights], trainable = True, mask_zero=True, name='embedding_layer')(sentences_input)
	bilstm_encoder = Bidirectional(LSTM(hidden_size),name='bilstm_encoder')(embedded_sentences)
	sentences_model = Model(sentences_input, bilstm_encoder)

	docs_input = Input(shape=(MAX_SENTENCES, MAX_SEQUENCE_LENGTH), name='docs_input', dtype='int32')
	sentence_vector = TimeDistributed(sentences_model,name='sentence_vector')(docs_input)
	#dropout = Dropout(0.2)(docs_encoded)
	docs_encoder = Bidirectional(LSTM(hidden_size),name='docs_encoder')(sentence_vector)

	# Prediction
	prediction = Dense(num_classes, activation='softmax', name='dense_out')(docs_encoder)
	model = Model(docs_input, prediction)
	model.compile(loss=loss_function, optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())

	'''
	
	'''


	return model

def hierarchyClassifierGRU(MAX_SENTENCES, MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights, num_classes):


	hidden_size = 50

	if num_classes > 2:
		loss_function = 'categorical_crossentropy'
	else:
		loss_function = 'binary_crossentropy'

	sentences_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name='sentences_input', dtype='int32')
	embedded_sentences = Embedding(VOCAB_LENGTH, EMBEDDING_DIM, weights=[embedding_weights], trainable = True, mask_zero=True, name='embedding_layer')(sentences_input)
	gru_encoder = GRU(hidden_size,name='gru_encoder')(embedded_sentences)
	sentences_model = Model(sentences_input, gru_encoder)

	docs_input = Input(shape=(MAX_SENTENCES, MAX_SEQUENCE_LENGTH), name='docs_input', dtype='int32')
	sentence_vector = TimeDistributed(sentences_model,name='sentence_vector')(docs_input)
	#dropout = Dropout(0.2)(docs_encoded)
	docs_encoder = GRU(hidden_size,name='docs_encoder')(sentence_vector)

	# Prediction
	prediction = Dense(num_classes, activation='softmax', name='dense_out')(docs_encoder)
	model = Model(docs_input, prediction)
	model.compile(loss=loss_function, optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())

	'''
	

	'''


	return model


def hierarchyClassifierBiGRU(MAX_SENTENCES, MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights, num_classes):


	hidden_size = 50

	if num_classes > 2:
		loss_function = 'categorical_crossentropy'
	else:
		loss_function = 'binary_crossentropy'

	sentences_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name='sentences_input', dtype='int32')
	embedded_sentences = Embedding(VOCAB_LENGTH, EMBEDDING_DIM, weights=[embedding_weights], trainable = True, mask_zero=True, name='embedding_layer')(sentences_input)
	bigru_encoder = Bidirectional(GRU(hidden_size),name='BIgru_encoder')(embedded_sentences)
	sentences_model = Model(sentences_input, bigru_encoder)

	docs_input = Input(shape=(MAX_SENTENCES, MAX_SEQUENCE_LENGTH), name='docs_input', dtype='int32')
	sentence_vector = TimeDistributed(sentences_model,name='sentence_vector')(docs_input)
	#dropout = Dropout(0.2)(docs_encoded)
	docs_encoder = Bidirectional(GRU(hidden_size),name='docs_encoder')(sentence_vector)

	# Prediction
	prediction = Dense(num_classes, activation='softmax', name='dense_out')(docs_encoder)
	model = Model(docs_input, prediction)
	model.compile(loss=loss_function, optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())

	'''
	
	'''


	return model


def hierarchyAtt(MAX_SENTENCES, MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights, num_classes):


	hidden_size = 50

	if num_classes > 2:
		loss_function = 'categorical_crossentropy'
	else:
		loss_function = 'binary_crossentropy'

	sentences_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name='sentences_input', dtype='int32')
	embedded_sentences = Embedding(VOCAB_LENGTH, EMBEDDING_DIM, weights=[embedding_weights], trainable = True, mask_zero=True, name='embedding_layer')(sentences_input)
	bilstm_encoder = Bidirectional(LSTM(hidden_size),name='bilstm_encoder')(embedded_sentences)
	sentences_model = Model(sentences_input, bilstm_encoder)

	docs_input = Input(shape=(MAX_SENTENCES, MAX_SEQUENCE_LENGTH), name='docs_input', dtype='int32')
	sentence_vector = TimeDistributed(sentences_model,name='sentence_vector')(docs_input)
	#dropout = Dropout(0.2)(docs_encoded)
	docs_encoder = Bidirectional(LSTM(hidden_size),name='docs_encoder')(sentence_vector)

	# Prediction
	prediction = Dense(num_classes, activation='softmax', name='dense_out')(docs_encoder)
	model = Model(docs_input, prediction)
	model.compile(loss=loss_function, optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())

	'''
	
	'''


	return model


def seqTDEncDec(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights):

	row_hidden_size = 200
	col_hidden_size = 200
	num_layers = 3

	model = Sequential()	
	# input captured here is sequence of sentences in shape (rows, time_steps, n_dim)
	model.add(Embedding(VOCAB_LENGTH, EMBEDDING_DIM, input_shape=(MAX_SEQUENCE_LENGTH,), mask_zero=True, trainable = True, weights=[embedding_weights], name='embedding_layer'))
	# encoding rows (sentences)
	model.add(TimeDistributed(LSTM(row_hidden_size,name='lstm_enc_1')))

	model.add(RepeatVector(y_max_len))

	# Creating decoder network
	for i in range(num_layers):
		model.add(LSTM(hidden_size, name='lstm_%s'%(i+2), return_sequences=True))
	model.add(TimeDistributed(Dense(y_vocab_len,name='dense')))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())

	# encoding cols (documents)
	model.add(LSTM(col_hidden_size,name='lstm_enc_2'))
	model.add(Dense(num_classes, activation='softmax', name='dense_out'))
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

	print(model.summary())

	return model

def seqTDClassifier(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights, num_classes):

	row_hidden_size = 200
	col_hidden_size = 200

	if num_classes > 2:
		loss_function = 'categorical_crossentropy'
	else:
		loss_function = 'binary_crossentropy'

	model = Sequential()	
	# input captured here is sequence of sentences in shape (rows, time_steps, n_dim)
	model.add(Embedding(VOCAB_LENGTH, EMBEDDING_DIM, input_shape=(MAX_SEQUENCE_LENGTH,), mask_zero=True, trainable = True, weights=[embedding_weights], name='embedding_layer'))
	# encoding rows (sentences)
	model.add(TimeDistributed(LSTM(row_hidden_size,name='lstm_enc_1')))

	# encoding cols (documents)	model.add(TimeDistributed(LSTM(row_hidden_size,name='lstm_enc_1')))
	
	model.add(LSTM(col_hidden_size,name='lstm_enc_2'))
	model.add(Dense(num_classes, activation='softmax', name='dense_out'))
	model.compile(loss=loss_function, optimizer='rmsprop', metrics=['accuracy'])

	print(model.summary())


	return model


################################################
# Hierarchical LSTM encoder - decoder type 2
# with keras functional API vs. sequential model





def seqClassifier(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights, num_classes):

	n_docs = 1
	row_hidden_size = 200
	col_hidden_size = 200

	if num_classes > 2:
		loss_function = 'categorical_crossentropy'
	else:
		loss_function = 'binary_crossentropy'


	sentences_model = Sequential()
	# input captured here is sequence of sentences in shape (rows, time_steps, n_dim)
	sentences_model.add(Embedding(VOCAB_LENGTH, EMBEDDING_DIM, input_shape=(MAX_SEQUENCE_LENGTH,), mask_zero=True, trainable = True, weights=[embedding_weights], name='embedding_layer'))
	# Creating encoder for capturing sentence embedding
	sentences_model.add(LSTM(row_hidden_size,name='lstm_enc_1'))

	# Creating encoder for capturing document embedding
	docs_model = Sequential()
	docs_model.add(TimeDistributed(Input(shape=(n_docs, MAX_SEQUENCE_LENGTH), dtype='int32', name='td_input_docs')))
	

	model = Sequential()
	model.add(Merge([sentences_model, docs_model], mode='concat'))
	model.add(LSTM(col_hidden_size,name='lstm_enc_2'))

	# Prediction
	model.add(Dense(num_classes, activation='softmax', name='dense_out'))
	model.compile(loss=loss_function, optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())

	return model

# with time-distributed layer
# return all time steps in previous layer
def simpleSeqClassifier2(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights):

	hidden_size = 200

	model = Sequential()

	model.add(Embedding(VOCAB_LENGTH, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, trainable = True, mask_zero=True, weights=[embedding_weights], name='embedding_layer'))
	model.add(Dropout(0.25))

	model.add(LSTM(hidden_size, name='lstm_enc'))
	model.add(RepeatVector(MAX_SEQUENCE_LENGTH))
	#model.add(LSTM(hidden_size, name='lstm_dec',return_sequences=True))

	model.add(LSTM(hidden_size, name='lstm_dec',return_sequences=True))
	model.add(TimeDistributed(Dense(1)))

	#model.add(TimeDistributed(Dense(1)))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

	print(model.summary())

	return model


# old model without pre-trained embedding matrix
def seqEncDec_old(X_vocab_len, X_max_len, y_vocab_len, y_max_len, EMBEDDING_DIM):

	hidden_size = 200
	num_layers = 3

	model = Sequential()
	
	model.add(Embedding(X_vocab_len, EMBEDDING_DIM, input_length=X_max_len, mask_zero=True, name='embedding_layer'))
	# Creating encoder network
	model.add(LSTM(hidden_size,name='lstm_enc_1'))
	model.add(RepeatVector(y_max_len))

	# Creating decoder network
	for i in range(num_layers):
		model.add(LSTM(hidden_size, name='lstm_%s'%(i+2), return_sequences=True))
	model.add(TimeDistributed(Dense(y_vocab_len,name='dense')))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())


	return model

################################################
# learning non-aligned bi-lingual input and output
# by merging 2 sequential models

################################################
def seqParallelEnc(X_vocab_len, X_max_len, y_vocab_len, y_max_len, EMBEDDING_DIM, X_embedding_weights, y_embedding_weights):

	hidden_size = 200
	nb_classes = 2

	encoder_a = Sequential()
	encoder_a.add(Embedding(X_vocab_len, EMBEDDING_DIM, input_length=X_max_len, mask_zero=True, weights=[X_embedding_weights], name='X_embedding_layer'))
	encoder_a.add(LSTM(hidden_size,name='lstm_a'))

	encoder_b = Sequential()
	encoder_b.add(Embedding(y_vocab_len, EMBEDDING_DIM, input_length=y_max_len, mask_zero=True, weights=[y_embedding_weights], name='y_embedding_layer'))
	encoder_b.add(LSTM(hidden_size,name='lstm_b'))

	decoder = Sequential()
	decoder.add(Merge([encoder_a, encoder_b], mode='concat'))
	decoder.add(Dense(hidden_size, activation='relu'))
	decoder.add(Dense(nb_classes, activation='softmax'))

	decoder.compile(loss='categorical_crossentropy', optimizer='rmsprop')

	print(decoder.summary())


	return decoder



	
