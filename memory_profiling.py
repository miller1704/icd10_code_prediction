from flask import Flask, request, render_template, jsonify
from pandas import read_csv
import numpy as np
from numba import jit

#setup internal libraries (default for python)
import pickle
import json

#just for memory profiling
import psutil
import memory_profiler
import sys

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def similarity_func(X_train_csm_column
                    ,X_train_index_pointer
                    ,X_train_sparse_row_num
                    ,X_test): # Function is compiled to machine code when called the first time
    
    #initialize storage
    X_train_shared_tokens_compressed = []
    X_train_shared_tokens = []
    X_train_total_tokens = []
    
    #create 1d array of length equal to compressed matrix rows containing 1 if that row (> 0 x_train token) has a match in test
    for X_train_col in X_train_csm_column:   # Numba likes loops
        X_train_shared_tokens_compressed.append(X_test[X_train_col] > 0)
        
    #create 1d array of length equal to original matrix rows containing sum of total and intersecting tokens
    for row_sparse in range(X_train_sparse_row_num):
        total = 0
        intersections = 0
        #for each pointer, iterate through the compressed matrix to add up total & total intersections per row
        for row_compressed in range(X_train_index_pointer[row_sparse],X_train_index_pointer[row_sparse+1]):
            intersections += X_train_shared_tokens_compressed[row_compressed]
            total += 1
        #then insert within arrays that have length of sparse matrix rows
        X_train_total_tokens.append(total)
        X_train_shared_tokens.append(intersections)
        #then return similarity
    return (np.array(X_train_shared_tokens) / np.array(X_train_total_tokens), np.array(X_train_total_tokens))

#define prediction function
def diag_predict(text):
    try:
        with open('tfidf_model', 'rb') as out1:
            tfidf_model = pickle.load(out1) 
        #transform test data into bag of words (as csm)
        X_test = tfidf_model.transform(np.array([text])).toarray()
        del tfidf_model
        tr,tc = X_test.shape
        X_test = X_test.reshape(tc,)
        #create a numpy friendly 3 array compressed sparse matrix
        with open('X_train', 'rb') as out2:
            X_train = pickle.load(out2)
        X_train_sparse_row_num = len(X_train.indptr)-1 #index pointer has length 1 longer than rows in sparse matrix

        #run similarity func
        data = read_csv(
            filepath_or_buffer = './static/icd10_all_codes.csv'
            ,sep=','
            ,engine='python')
        data['similarity'], data['total_tokens'] = similarity_func(np.array(X_train.indices)
                                                                   ,np.array(X_train.indptr)
                                                                   ,X_train_sparse_row_num
                                                                   ,X_test)
        del X_train
        data['similarity_adjusted'] = data['similarity']*(np.log(data['total_tokens']+10) / np.log(2))
        #sort data twice in output (prioritizing memory over processing)
        return data.sort_values(by='similarity_adjusted',ascending=False)['code'].iloc[0],data.sort_values(by='similarity_adjusted',ascending=False)['description'].iloc[0]
    except Exception as err:
        return err, err

@profile
def prediction():
    text = 'Patient has Sepsis '
    if text:
        icd10_code, icd10_description = diag_predict(text)
    else:
        icd10_code = ""
        icd10_description = ""
    return jsonify(
        {"icd10_code":str(icd10_code),"icd10_description": str(icd10_description)}
        )

if __name__ == '__main__':
    prediction()