from __future__ import print_function
import numpy as np
import pandas as pd
import os
import sys
import re
import math
from itertools import groupby
from string import punctuation
import _pickle as cPickle
import nltk
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA


DATAPATH = 'data/airlines'

# saving file into pickle format
def savePickle(dataToWrite,pickleFilename):
	f = open(pickleFilename, 'wb')
	cPickle.dump(dataToWrite, f)
	f.close()

# reading file in pickle format
def readPickle(pickleFilename):
	f = open(pickleFilename, 'rb')
	obj = cPickle.load(f)
	f.close()
	return obj



# term frequency
def countFrequency(term, tokenized_document):
	return tokenized_document.count(term)

# document frequency
def docFrequency(term, tokenized_documents):
	count = 0
	for doc in tokenized_documents:
		if countFrequency(term, doc) > 0:
			count += 1
	return 1 + count


def createTfIdf(tokenized_documents, vocab, words_IDF):

	print('creating TFIDF matrix...')

	# in this CSR matrix, each row is : (doc-id, word-index, value);
	# value is either Term Frequency (TF), TF-IDF 

	# doc-id 
	
	all_rows = []
	all_cols = []
	all_datatf = []
	all_datatfidf = []

	for i, doc in enumerate(tokenized_documents):
		
		tf = []

		for word_index,term in vocab.items():
			tf.append(countFrequency(term, doc))
		

		tf_arr = np.array(tf)
		# col : word-index --> index of word when TF != 0
		col  = np.where( tf_arr>0 )[0]
		n_row = len(col)
		# row : doc-id --> current document-id, repeat for length of word-index when TF != 0
		row = np.array([i]*n_row)
		# data : value --> TF
		datatf = tf_arr[tf_arr.nonzero()]

		# get IDF values in list
		idf_val = []

		for j in range(len(col)):
			word_index = col[j]
			idf_val.append(words_IDF[word_index])

		# storing TF-IDF values
		data_tfidf = [a*b for a,b in zip(list(datatf),idf_val)]

		# store all list
		all_rows.extend(list(row))
		all_cols.extend(list(col))
		all_datatf.extend(list(datatf))
		all_datatfidf.extend(data_tfidf)
	
	
	arr_rows = np.array(all_rows)
	arr_cols = np.array(all_cols)
	arr_datatf = np.array(all_datatf)
	arr_datatfidf = np.array(all_datatfidf)

	# create sparse matrix for BOW
	bow_smatrix = csr_matrix((arr_datatf,(arr_rows,arr_cols)))
	tfidf_smatrix = csr_matrix((arr_datatfidf,(arr_rows,arr_cols)))


	return bow_smatrix, tfidf_smatrix


def getIDF(vocab, tokenized_documents):

	words_IDF = {}

	for index,word in vocab.items():

		# idf(term) = ( log ((1 + nd)/(1 + df(doc,term))) ) 
		# where nd : number of document in corpus; 
		# df : doc frequency (number of documents containing term)
		idf = math.log( (1 + len(tokenized_documents)) / float(docFrequency(word, tokenized_documents)))
		words_IDF[index] = idf

	return words_IDF 

def getTFIDF(tfidf_smatrix):

	# sparse matrix structure
	# (0, 26) 1
	# (0, 100) 1
	# r,c = smatrix.nonzero()
	# tfidf_data = smatrix.data
	# to get row dense matrix
	# row = smatrix.getrow(row_id).toarray()[0].ravel()
	# top_ten_indices = row.argsort()[-10:]
	# top_ten_values = row[row.argsort()[-10:]]

	doc_id, word_index = tfidf_smatrix.nonzero()
	tfidf = tfidf_smatrix.data
	arr_tfidf = list(zip(doc_id,word_index,tfidf))
	df_tfidf = pd.DataFrame(arr_tfidf, columns=['doc_id', 'word_id', 'tfidf'])
	#sorted_tfidf = df_tfidf.sort_values(['tfidf'], ascending=False)
	#sorted_tfidf = sorted_tfidf.reset_index(drop=True)


	return df_tfidf

def getTFMatrix(bow_matrix):

	doc_id, word_index = bow_matrix.nonzero()
	tf = bow_matrix.data
	arr_tf = list(zip(doc_id,word_index,tf))
	df_tf = pd.DataFrame(arr_tf, columns=['doc_id', 'word_id', 'tf'])
	
	return df_tf


def reduceDimensionPCA(n_dim, doc_vector):

	pca = PCA(n_components=n_dim)
	pca.fit(doc_vector)
	reduced_doc_vectors = pca.transform(doc_vector)

	return reduced_doc_vectors

if __name__ == '__main__':

	vocab = readPickle(os.path.join(DATAPATH,'vocab'))
	labelled_docs  = readPickle(os.path.join(DATAPATH,'fin_labelled_strtweets'))

	tokenized_tweets = []
	for i, data in enumerate(labelled_docs):
		text = data[0]
		tokenized_tweets.append(text)


	#############################################################
	# Generating term document matrix for word-based LDA model

	# get IDF weight
	words_IDF = getIDF(vocab, tokenized_tweets)
	savePickle(words_IDF, os.path.join(DATAPATH,'words_IDF'))

	# create term-document bow and tfidf matrix
	bow_smatrix, tfidf_smatrix = createTfIdf(tokenized_tweets, vocab, words_IDF)

	df_tfidf = getTFIDF(tfidf_smatrix)
	df_bow_smatrix = getTFMatrix(bow_smatrix)

	# sorting based on document
	unique_doc = df_tfidf.doc_id.unique()
	s_unique_doc = np.sort(unique_doc)
	unique_words = df_tfidf.word_id.unique()
	s_unique_words = np.sort(unique_words)

	# original term document matrix without reduction
	arr_td = np.zeros(shape=(len(s_unique_doc), len(s_unique_words)), dtype='float32')
		
	
	for i in range(len(arr_td)):
		docid = s_unique_doc[i]
		doc_wordlist = list(df_tfidf[df_tfidf['doc_id']==docid]['word_id'])
		for j in range(len(s_unique_words)):
			wid = s_unique_words[j]
			# get tfidf value based on doc_id and word_index

			if wid in doc_wordlist:
				tfidf_val = df_tfidf.loc[((df_tfidf['doc_id']==docid) & (df_tfidf['word_id']==wid))]['tfidf'].iloc[0]
			else:
				tfidf_val = 0
			arr_td[i][j] = tfidf_val
	
	savePickle(arr_td, os.path.join(DATAPATH,'arr_td'))

	# reduce dimension to 50-dim using PCA
	pca_50dim = reduceDimensionPCA(50,arr_td)

	savePickle(pca_50dim, os.path.join(DATAPATH,'pca_50dim'))

