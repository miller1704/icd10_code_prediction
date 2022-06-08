from flask import Flask, request, render_template, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pandas import read_csv
import numpy as np    
from numba import jit

#setup internal libraries (default for python)
import pickle

#setup app
app = Flask(__name__)

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["500 per day", "300 per hour"]
)

def load_data():
    if ('X_train_csm_column' not in globals().keys()) & ('X_train_index_pointer' not in globals().keys()):
        with open('X_train', 'rb') as out2:
            X_train  = pickle.load(out2)
            globals()['X_train_csm_column'] = np.array(X_train.indices)
            globals()['X_train_index_pointer'] = np.array(X_train.indptr)
            del X_train
    if 'tfidf_model' not in globals().keys():
        with open('tfidf_model', 'rb') as out1:
            globals()['tfidf_model'] = pickle.load(out1) 
        #transform test data into bag of words (as csm)
    if 'data' not in globals().keys():
        with open('data', 'rb') as out2:
            globals()['data']  = pickle.load(out2)
    from sklearn.feature_extraction.text import TfidfVectorizer


#utility: similarity measurement; paralellized
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
def diag_predict_y_in_x(text):
    #load global variables as needed
    load_data()
    X_test = tfidf_model.transform(np.array([text])).toarray()
    tr,tc = X_test.shape
    X_test = X_test.reshape(tc,)
    #create a numpy friendly 3 array compressed sparse matrix
    X_train_sparse_row_num = len(data['code']) #index pointer has length 1 longer than rows in sparse matrix
    #load data and run similarity
    data['similarity'], data['total_tokens'] = similarity_func(X_train_csm_column
                                                               ,X_train_index_pointer
                                                               ,X_train_sparse_row_num
                                                               ,X_test)
    data['similarity_adjusted'] = data['similarity']*(np.log(data['total_tokens']+10) / np.log(2))
    #sort data twice in output (prioritizing memory over processing)
    return data.sort_values(by='similarity_adjusted',ascending=False)['code'].iloc[0],data.sort_values(by='similarity_adjusted',ascending=False)['description'].iloc[0]

@app.route("/")
def index():
    return render_template('/index.html')

@app.route('/prediction')
@limiter.limit("1/4 second", override_defaults=False)
def prediction():
    text = request.args.get('text', '', type=str)
    if text:
        icd10_code, icd10_description = diag_predict_y_in_x(text)
    else:
        icd10_code = ""
        icd10_description = ""
    return jsonify({"icd10_code":str(icd10_code),"icd10_description": str(icd10_description)})


        
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


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
