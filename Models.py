import time
import numpy as np
import pandas as pd
from metrics import * 
from scipy.sparse import csr_matrix, diags, csc_matrix, find
from scipy.linalg import lu_factor, lu_solve, cholesky
from scipy.linalg.lapack import spotri, dpotri

import os
from bottleneck import argpartition
from copy import deepcopy


def sparsify(B, threshold, max_in_col=None):
    """
    Sets all elements of B which are smaller than the threshold (in absolute value) to zero. Additionally,
    it is ensured that there are at most max_in_col many nonzero entries per column.
    
    Limiting the number of elements per column in the model matrix means limiting the number of "neighbours" an item can have, if the predictions are made as:
    pred = X_test @ B
    """

    
    if not max_in_col:
        max_in_col = B.shape[0]
    
    if threshold == 0.:
        indexes = argpartition(-np.abs(B), max_in_col-1, axis=0)[:max_in_col, :]
        values = np.take_along_axis(B, indexes, axis=0)
        A = np.zeros(B.shape, dtype=np.float32)
        A[indexes, np.arange(B.shape[0])[None, :]] = values
        A = csc_matrix(A)
        print("Resulting sparsity of A: {}".format( A.nnz*1.0 / A.shape[0] / A.shape[0]) )
        
    else: 
        idx = np.where( np.abs(B) > threshold)
        A = csc_matrix( (B[idx], idx), shape=B.shape, dtype=np.float32)

        count_in_col = A.getnnz(axis=0)
        i_list = np.where(count_in_col > max_in_col)[0]
        print("Number of items with more than {} entries in column: {}".format(max_in_col, len(i_list)) )
        for i in i_list:
            j = A[:,i].nonzero()[0]
            k = argpartition(-np.abs(np.asarray(A[j,i].todense()).flatten()), max_in_col)[max_in_col:]
            A[  j[k], i ] = 0.0
        A.eliminate_zeros()
        print("Resulting sparsity of A: {}".format( A.nnz*1.0 / A.shape[0] / A.shape[0]) )

    return A 

    
def sparsity_pattern_cov(XtX, n_users, alpha, threshold, max_in_col):
    # if alpha=1, then XtX becomes the correlation matrix below
    XtX_diag = np.diag(XtX)
    idx_diag = np.diag_indices(XtX.shape[0])

    mu = XtX_diag / n_users
    variance_times_usercount = XtX_diag - mu * mu * n_users

    # we dont want to keep the diagonal
    XtX[idx_diag] = 0.

    # standardizing the data-matrix XtX (if alpha=1, then XtX becomes the correlation matrix)
    XtX -= mu[:,None] * (mu * n_users)
    rescaling = np.power(variance_times_usercount, alpha / 2.0) 
    scaling = 1.0  / rescaling
    XtX = scaling[:,None] * XtX * scaling

    # apply threshold
    idx = np.where( np.abs(XtX) > threshold)
    A = csc_matrix( (XtX[idx], idx), shape=XtX.shape, dtype=np.float32)
    
    # enforce at most max_in_col many nnz per column
    count_in_col = A.getnnz(axis=0)
    i_list = np.where(count_in_col > max_in_col)[0]
    print("Number of items with more than {} entries in column: {}".format(max_in_col, len(i_list)) )
    for i in i_list:
        j = A[:,i].nonzero()[0]
        k = argpartition(-np.abs(np.asarray(A[j,i].todense()).flatten()), max_in_col)[max_in_col:]
        A[  j[k], i ] = 0.0
    A.eliminate_zeros()
    print("Resulting sparsity of A: {}".format( A.nnz*1.0 / A.shape[0] / A.shape[0]) )

    return A 


class BaseRecommender:
    def evaluate_metrics(self, test_data_pr: pd.DataFrame, test_data_eval: pd.DataFrame,
                         uid_str: str, sid_str: str, metrics: list):
        # if nusers_pr < nusers_eval raise error
        metric_callers = []
        max_topk = 0
        for metric in metrics:
            try:
                metric_callers.append(eval(metric))
                max_topk = max(max_topk, int(metric.split("k=")[-1].strip(")")))
            except:
                raise NotImplementedError('metrics={} not implemented.'.format(metric))
        
        print('create lookup')
        self.lookup = pd.DataFrame({'range_index': test_data_pr[uid_str].astype('category').cat.codes.unique(),
                                    'initial_index': test_data_pr[uid_str].unique()})
        self.lookup.set_index('initial_index', drop=True, inplace=True)
        
        print('create true item df')
        true_item_df = test_data_eval.groupby(uid_str, sort=False)[sid_str].apply(list)
        true_items_chunk = true_item_df.values     
        indexes = self.lookup.loc[true_item_df.index.values].range_index.values
        
        print('predicting')
        top_maxk_items = self.predict(test_data_pr, uid_str, sid_str, max_topk)
        print('predicting done, reindexing')
        top_maxk_items = top_maxk_items[indexes, :]
        self.top_maxk_items = top_maxk_items
        print('calculating metrics')
        results = [[fn(top_maxk_items, true_items) for fn in metric_callers] \
                    for top_maxk_items, true_items in zip(top_maxk_items, true_items_chunk)]
        
        average_result = np.average(np.array(results), axis=0).tolist()
        return_dict = dict(zip(metrics, average_result))
        return return_dict     


class NumpyRecommender(BaseRecommender):
    def _parse_data(self, data: pd.DataFrame, uid_str: str, sid_str: str, dtype=np.float32):
        nUsers = len(data[uid_str].unique())
        users = data[uid_str].astype('category').cat.codes.values
        items = data[sid_str].values
        vals = np.ones(len(data))
        X = csr_matrix((vals, (users, items)), shape=(nUsers, self.nItems), dtype=dtype)
        return X
    
    
class TorchRecommender(BaseRecommender):
    def _parse_data_csr(self, data: pd.DataFrame, uid_str: str, sid_str: str):
        """
        Unused, since multiplying csr matricies in a loop can cause memory leak.
        """
        nUsers = len(data[uid_str].unique())
        
        # calculate crow_indices
        uids_csr = [0]
        #data = data.sort_values(by=uid_str)
        counts = data[uid_str].value_counts(sort=False).to_list()
        for val in counts:
            uids_csr.append(uids_csr[-1] + val)
        sids = data[sid_str].to_list()
        
        # construct sparse csr tensor
        X = torch.sparse_csr_tensor(
                crow_indices=uids_csr, col_indices=sids,
                values=torch.ones(len(data)),
                size=(nUsers, self.nItems),
                #device=torch.device('cpu')
        )
        return X
    
    
    def _parse_data(self, data: pd.DataFrame, uid_str: str, sid_str: str):
        nUsers = len(data[uid_str].unique())
        users = data[uid_str].astype('category').cat.codes.values
        items = data[sid_str].values
        X = torch.sparse_coo_tensor(np.vstack((users, items)), 
                                    values=torch.ones(len(data)),
                                    size=(nUsers, self.nItems),
                                    device=torch.device('cpu'))
        return X

    
class MostPopularRecommender(NumpyRecommender): 
    def __init__(self, nItems):
        self.nItems = nItems
    
    
    def fit(self, train_data: pd.DataFrame, uid_str: str, sid_str: str):
        self.freq_vec = self._parse_data(train_data, uid_str, sid_str).sum(axis=0).A1
        
        
    def predict(self, data: pd.DataFrame, uid_str: str, sid_str: str, k: int, return_scores=False):
        self.k = k
        self.X_test = self._parse_data(data, uid_str, sid_str)
        
        self.nUsers = len(data[uid_str].unique())
        X_rowidx, X_colidx, _ = find(self.X_test)
        
        # freq_vec contains the item frequencies, we just stack this vector nUsers many times
        self.pred = np.tile(self.freq_vec, (self.nUsers, 1))
        
        # we dont want to predict any popular items that the user has prev. interacted with
        # -> set their score to -inf
        self.pred[X_rowidx, X_colidx] = -np.inf
        
        # top k items most popular items for each user
        self.topk = argpartition(-self.pred, self.k - 1, axis=1)[:, :self.k]
        
        # we need to sort manually
        n_test = self.X_test.shape[0]
        sorted_idx = (-self.pred)[np.arange(n_test)[:, None], self.topk].argsort()
        self.topk = self.topk[np.arange(n_test)[:, None], sorted_idx]
        return self.topk
    
    
        
        
class ItemKNN(NumpyRecommender):
    def __init__(self, similarity_measure, 
                 num_neighbors, 
                 renormalize_similarity,
                 alpha,
                 renormalization_interval,
                 enable_average_bias, 
                 min_similarity_threshold,
                 trunc_entries,
                 trunc_val,
                 l1_normalization):
        self.similarity_measure = similarity_measure
        self.num_neighbors = num_neighbors
        self.alpha = alpha
        self.renormalize_similarity = renormalize_similarity
        self.renormalization_interval = renormalization_interval
        self.enable_average_bias = enable_average_bias
        self.min_similarity_threshold = min_similarity_threshold
        self.trunc_entries = trunc_entries
        self.trunc_val = trunc_val
        self.l1_normalization = l1_normalization
        
        
    def pearson_sparse_data(self, X, eps):
        """ This could be numerically unstable (cancellation) """
        pearson = np.dot(X.T, X).toarray()
        n = X.shape[0]
        mu = X.mean(axis=0).A1
        pearson -= n * np.outer(mu, mu)

        var = np.diag(pearson)
        var_scaled = np.power(var, self.alpha / 2.0) # np.sqrt(var)
        inv_var_scaled = 1 / np.maximum(var_scaled, eps)

        pearson = pearson * inv_var_scaled
        pearson = pearson.T * inv_var_scaled
        return pearson
    
    
    def cosine_sim(self, X, eps):
        sim_mat = np.dot(X.T, X).toarray()
        
        norm_cols = np.diag(sim_mat)
        norm_cols_scaled = np.power(norm_cols, self.alpha / 2.0) # np.sqrt(norm_cols)
        
        inv_norm_cols_scaled = 1 / np.maximum(norm_cols_scaled, eps)
        sim_mat = sim_mat * inv_norm_cols_scaled
        sim_mat = sim_mat.T * inv_norm_cols_scaled
        return sim_mat
   
        
    def fit(self, train_data: pd.DataFrame, uid_str: str, sid_str: str, nItems: int, eps: float = 1e-10):    
        if not set([uid_str, sid_str]) <= set(train_data.columns):
            raise ValueError(f'{uid_str} or {sid_str} not present in test_data columns')
            
        # one might wonder why we cant just do np.unique or np.max + 1 to get nItems..
        # depending on whether weak or strong generalisation is used, 
        # some items might only be in the test-set, but we need B to have to correct size of nItems x nItems
        self.nItems = nItems
        self.X_train = self._parse_data(train_data, uid_str, sid_str)
        
        # calculate the similarity matrix for the pairwise similarities of the columns of X
        if self.similarity_measure == 'pearson':
            # here we use that the pearson correlation of two columns of X (x_i and x_j) is defined as
            # the cosine similarity of x_i - \bar{x_i} and x_j - \bar{x_j}
            sim_mat = self.pearson_sparse_data(self.X_train, eps=eps)
            sim_mat[np.diag_indices(self.nItems)] = 0.
            
        elif self.similarity_measure == 'cosine':
            sim_mat = self.cosine_sim(self.X_train, eps=eps) 
            sim_mat[np.diag_indices(self.nItems)] = 0. # set diagnal to 0
            # TODO: is it really a good idea to to set the diag to -1? 
            # since the columns of X are non-negative, the similarity will lie in [0,1], not [-1,1]
            # (below entries below a threshold are set to zero anyways)
            
        else:
            raise NotImplementedError("similarity_measure=%s is not supported." % self.similarity_measure)
        
        
        if self.renormalize_similarity: 
            if self.renormalization_interval == '[0,1]':
                # map to interval [0, 1]
                min_val, max_val = sim_mat.min(), sim_mat.max()
                sim_mat = (sim_mat - min_val) / (max_val - min_val) 
                sim_mat[np.diag_indices(self.nItems)] = 0.
            elif self.renormalization_interval == '[-1,1]':
                min_val, max_val = sim_mat.min(), sim_mat.max()
                sim_mat = (2 * (sim_mat - min_val) / (max_val - min_val)) - 1
                sim_mat[np.diag_indices(self.nItems)] = 0.
            else:
                raise ValueError(f'{self.renormalization_interval} is not a supported interval')
            
        if self.trunc_entries:
            sim_mat[sim_mat < self.min_similarity_threshold] = self.trunc_val  
        
        indices = argpartition(-np.abs(sim_mat), self.num_neighbors - 1, axis=1)[:, :self.num_neighbors]
        values = sim_mat[np.arange(self.nItems)[:, None], indices]
        sim_mat = np.zeros(sim_mat.shape, dtype=np.float32)
        sim_mat[np.arange(self.nItems)[:, None], indices] = values
        
        # normalize the rows to have l1-Norm of 1
        if self.l1_normalization:
            norrms_l1 = np.linalg.norm(sim_mat, ord=1, axis=1)
            norms_l1 = np.maximum(norrms_l1, eps * np.ones(len(norrms_l1))) #safeguard for division by 0
            sim_mat = sim_mat / norrms_l1[:, None]
        # use [:, None] to ensure broadcasting along the correct axis
        
        self.sim_mat = csr_matrix(sim_mat)
        
    
    def predict(self, data: pd.DataFrame, uid_str: str, sid_str: str, k: int):
        if not set([uid_str, sid_str]) <= set(data.columns):
            raise ValueError(f'{uid_str} or {sid_str} not present in data columns')
        self.k = k    
        self.X_test = self._parse_data(data, uid_str, sid_str)
        X_rowidx, X_colidx, _ = find(self.X_test)
    
        if self.enable_average_bias:
            # subtracting the mean will result in the matricies not being sparse anymore
            X_dense = self.X_test.toarray()
            item_mean = self.X_train.mean(axis=0).A1
            pred = np.dot(X_dense - item_mean, self.sim_mat.toarray().T) + item_mean
            
            # we dont want to predict items that the user already interacted with 
            pred[X_rowidx, X_colidx] = -np.inf
            del X_dense
            
        else:
            pred = np.dot(self.X_test, self.sim_mat.T).toarray()
            
            # we dont want to predict items that the user already interacted with 
            pred[X_rowidx, X_colidx] = -np.inf 
        
        # top k items most popular items for each user
        self.topk = argpartition(-pred, self.k - 1, axis=1)[:, :self.k]
        
        # we need to sort manually
        n_test = self.X_test.shape[0]
        sorted_idx = (-pred)[np.arange(n_test)[:, None], self.topk].argsort()
        self.topk = self.topk[np.arange(n_test)[:, None], sorted_idx]
        return self.topk
                
    
        

        
        
        
class DLAE(NumpyRecommender):
    """
    Denoising Linear Autoencoder. For details, see: https://proceedings.neurips.cc/paper/2020/file/e33d974aae13e4d877477d51d8bafdc4-Paper.pdf
    
    Default case is not computing the inverse explicitly to obtain the model matrix, but rather making predictions via solving a Linear System using Cholesky.
    """
    def __init__(self, lmbda, p, method='chol'):
        self.lmbda = lmbda
        self.p = p
        self.method = method
        

    def _upper_triangular_to_symmetric(self, upper_triang):
        upper_triang += np.triu(upper_triang, k=1).T
        
        
    def _faster_spd_inverse(self, matrix, dtype=np.float32):
        if matrix.dtype != dtype:
            raise ValueError('Supplied dtype doesnt match matrix dtype')
            
        cholesky_decomp = cholesky(matrix)
        if dtype == np.float32:
            inv, info = spotri(cholesky_decomp)
        elif dtype == np.float64:
            inv, info = dpotri(cholesky_decomp) 
            
        else:
            raise ValueError(f'Unsupported dtype: {dtype}')
            
        if info != 0:
            raise ValueError('spotri failed on input')
        self._upper_triangular_to_symmetric(inv)
        return inv

        
    def fit(self, train_data: pd.DataFrame, uid_str: str, sid_str: str, nItems: int):    
        if not set([uid_str, sid_str]) <= set(train_data.columns):
            raise ValueError(f'{uid_str} or {sid_str} not present in test_data columns')

        self.nItems = nItems
        self.X_train = self._parse_data(train_data, uid_str, sid_str)
        
        # calculate the closed-form solution to the optimization problem
        G = np.dot(self.X_train.T, self.X_train).toarray()
        G_diag = np.diag(G) # vector with diagonal entries of G
        
        if self.method == 'lu':
            lambda_mat_inv_diag = 1 / (self.p / (1 - self.p) * G_diag + self.lmbda)
            A = np.multiply(G, lambda_mat_inv_diag[None,:]) + np.eye(self.nItems)
            self.lu, self.piv = lu_factor(A)
            
        
        elif self.method == 'chol':
            lambda_mat_diag = self.p / (1 - self.p) * G_diag + self.lmbda
            L = cholesky(G + np.diag(lambda_mat_diag), lower=True)
            diag_L = np.diag(L)
            self.lu = L / diag_L + diag_L[:, None] * L.T / lambda_mat_diag[None, :] #np.tril(L / np.diag(L), k=-1)
            self.lu[np.diag_indices(self.nItems)] -= 1
            
            del L
            self.piv = np.arange(self.nItems)
        
            
        elif self.method == 'inv':
            # compute the inverse directly (i.e the model matrix)
            lambda_mat_diag = self.p / (1 - self.p) * G_diag + self.lmbda
            self.B = np.eye(self.nItems) - np.multiply( _faster_spd_inverse(G + np.diag(lambda_mat_diag)), lambda_mat_diag[None,:])
            
        else:
            raise NotImplementedError(f"{self.method} is not a supported method.")
            
    
    def predict(self, data: pd.DataFrame, uid_str: str, sid_str: str, 
                k: int, return_scores: bool = False):
        if not set([uid_str, sid_str]) <= set(data.columns):
            raise ValueError(f'{uid_str} or {sid_str} not present in data columns')
                     
        self.k = k
        self.X_test = self._parse_data(data, uid_str, sid_str)
        X_rowidx, X_colidx, _ = find(self.X_test)
        
        if self.method in ['lu', 'chol']:
            X_test_dense = self.X_test.toarray().T
            self.pred = (X_test_dense - lu_solve((self.lu, self.piv), X_test_dense)).T      
            
        elif self.method == 'inv':
            X_test_dense = self.X_test.toarray()
            self.pred = X_test_dense @ self.B
        
        else:
            raise NotImplementedError(f"{self.method} is not a supported method.")
            
        del X_test_dense
        # we dont want to predict any popular items that the user has prev. interacted with
        # -> set their score to -inf
        self.pred[X_rowidx, X_colidx] = -np.inf
        
        # top k items most popular items for each user
        self.topk = argpartition(-self.pred, self.k - 1, axis=1)[:, :self.k]
        
        # we need to sort manually
        n_test = self.X_test.shape[0]
        sorted_idx = (-self.pred)[np.arange(n_test)[:, None], self.topk].argsort()
        self.topk = self.topk[np.arange(n_test)[:, None], sorted_idx]
        return self.topk
    
    
    
class EDLAE(NumpyRecommender):
    """
    Emphasized Denoising Linear Autoencoder. For details, see: https://proceedings.neurips.cc/paper/2020/file/e33d974aae13e4d877477d51d8bafdc4-Paper.pdf
    
    Inverse is computed via LAPACK routine, which is faster than using np.linalg.inv, since this doesnt exploit the definiteness of the Matrix.
    """
    
    def __init__(self, lmbda, p):
        self.lmbda = lmbda
        self.p = p
        
        
    def _upper_triangular_to_symmetric(self, upper_triang):
        upper_triang += np.triu(upper_triang, k=1).T


    def _faster_spd_inverse(self, matrix, dtype=np.float32):
        if matrix.dtype != dtype:
            raise ValueError('Supplied dtype doesnt match matrix dtype')
            
        cholesky_decomp = cholesky(matrix)
        if dtype == np.float32:
            inv, info = spotri(cholesky_decomp)
        elif dtype == np.float64:
            inv, info = dpotri(cholesky_decomp) 
            
        else:
            raise ValueError(f'Unsupported dtype: {dtype}')
            
        if info != 0:
            raise ValueError('spotri failed on input')
        self._upper_triangular_to_symmetric(inv)
        return inv

        
    def fit(self, train_data: pd.DataFrame, uid_str: str, sid_str: str, nItems: int, store_G_inv=False):    
        if not set([uid_str, sid_str]) <= set(train_data.columns):
            raise ValueError(f'{uid_str} or {sid_str} not present in test_data columns')

        self.nItems = nItems
        self.X_train = self._parse_data(train_data, uid_str, sid_str)
        
        # calculate the closed-form solution to the optimization problem
        G = np.dot(self.X_train.T, self.X_train).toarray()
        G_diag = np.diag(G) # vector with diagonal entries of G
        np.fill_diagonal(G, G_diag + self.p / (1 - self.p) * G_diag + self.lmbda)
        
        if store_G_inv:
            self.G_inv = G
            
        G = self._faster_spd_inverse(G) 
        print('finished inverse computation')
        self.B = G / (-np.diag(G))
        np.fill_diagonal(self.B, val=0.)
        
            
    
    def predict(self, data: pd.DataFrame, uid_str: str, sid_str: str, 
                k: int, return_scores: bool = False):
        if not set([uid_str, sid_str]) <= set(data.columns):
            raise ValueError(f'{uid_str} or {sid_str} not present in data columns')
               
        self.k = k
        self.X_test = self._parse_data(data, uid_str, sid_str)
        X_rowidx, X_colidx, _ = find(self.X_test)
        
        X_test_dense = self.X_test.toarray()
        pred = np.dot(X_test_dense, self.B) 
        del X_test_dense 
            
        # we dont want to predict items that the user already interacted with 
        pred[X_rowidx, X_colidx] = -np.inf
        
        # top k items most popular items for each user
        self.topk = argpartition(-pred, self.k - 1, axis=1)[:, :self.k]
        
        # we need to sort manually
        n_test = self.X_test.shape[0]
        sorted_idx = (-pred)[np.arange(n_test)[:, None], self.topk].argsort()
        self.topk = self.topk[np.arange(n_test)[:, None], sorted_idx]
        return self.topk
    


class MRFApprox(NumpyRecommender):
    """
    No mean removal and different regularization.
    
    Based on the code from: https://github.com/hasteck/MRF_NeurIPS_2019
    """
    def __init__(self, lmbda):
        self.lmbda = lmbda
    
    def _calculate_sparsity_pattern(self, XtX, alpha, threshold, max_in_col):

        #  if alpha=1, then XtX becomes the correlation matrix below
        usercount = self.X_train.shape[0]
        
        XtX_diag = np.diag(XtX)
        
        mu = XtX_diag / usercount
        variance_times_usercount = XtX_diag - mu * mu * usercount
        
        # standardizing the data-matrix XtX (if alpha=1, then XtX becomes the correlation matrix)
        XtX -= mu[:,None] * (mu * usercount)
        rescaling = np.power(variance_times_usercount, alpha / 2.0) 
        scaling = 1.0  / rescaling
        XtX = scaling[:,None].astype(np.float32) * XtX * scaling.astype(np.float32)


        # apply threshold
        idx = np.where( np.abs(XtX) > threshold)
        A = csc_matrix( (XtX[idx], idx), shape=XtX.shape, dtype=np.float32)
        # enforce maxInColumn, see section 3.1 in paper
        count_in_col = A.getnnz(axis=0)
        i_list = np.where(count_in_col > max_in_col)[0]
        print("Number of items with more than {} entries in column: {}".format(max_in_col, len(i_list)) )
        for i in i_list:
            j = A[:,i].nonzero()[0] # those are the nnz entries in column i
            k = argpartition(-np.abs(np.asarray(A[j,i].todense()).flatten()), max_in_col)[max_in_col:]
            A[j[k], i] = 0.0
        A.eliminate_zeros()
        print("Resulting sparsity of A: {}".format( A.nnz * 1.0 / A.shape[0] / A.shape[0]) )

        return A, XtX, scaling, rescaling
    
    
    def get_A(self, train_data: pd.DataFrame, uid_str: str, sid_str: str, nItems: int, 
            alpha, threshold, max_in_col):    
        if not set([uid_str, sid_str]) <= set(train_data.columns):
            raise ValueError(f'{uid_str} or {sid_str} not present in test_data columns')
            
        # one might wonder why we cant just do np.unique or np.max + 1 to get nItems..
        # depending on whether weak or strong generalisation is used, 
        # some items might only be in the test-set, but we need B to have to correct size of nItems x nItems
        self.nItems = nItems
        self.X_train = self._parse_data(train_data, uid_str, sid_str)
        self.alpha = alpha
        self.threshold = threshold
        self.max_in_col = max_in_col
        
        XtX = np.dot(self.X_train.T, self.X_train).toarray()
        
        # try the other sparsity pattern
        A, XtX, scaling, rescaling = self._calculate_sparsity_pattern(XtX=deepcopy(XtX), alpha=self.alpha, threshold=self.threshold, max_in_col=self.max_in_col)
        print(XtX.dtype)
        return A
    
    def _upper_triangular_to_symmetric(self, upper_triang):
        upper_triang += np.triu(upper_triang, k=1).T

        
    def _faster_spd_inverse(self, matrix, dtype=np.float32):
        if matrix.dtype != dtype:
            raise ValueError('Supplied dtype doesnt match matrix dtype')
            
        cholesky_decomp = cholesky(matrix)
        if dtype == np.float32:
            inv, info = spotri(cholesky_decomp)
        elif dtype == np.float64:
            inv, info = dpotri(cholesky_decomp) 
            
        else:
            raise ValueError(f'Unsupported dtype: {dtype}')
            
        if info != 0:
            raise ValueError('spotri failed on input')
        self._upper_triangular_to_symmetric(inv)
        return inv


        
    def fit(self, train_data: pd.DataFrame, uid_str: str, sid_str: str, nItems: int, 
            alpha, threshold, max_in_col, r, dtype=np.float32, store_dense = False):    
        if not set([uid_str, sid_str]) <= set(train_data.columns):
            raise ValueError(f'{uid_str} or {sid_str} not present in test_data columns')
            
        # one might wonder why we cant just do np.unique or np.max + 1 to get nItems..
        # depending on whether weak or strong generalisation is used, 
        # some items might only be in the test-set, but we need B to have to correct size of nItems x nItems
        self.nItems = nItems
        self.store_dense = store_dense
        self.X_train = self._parse_data(train_data, uid_str, sid_str, dtype=dtype)
        self.alpha = alpha
        self.threshold = threshold
        self.r = r
        self.max_in_col = max_in_col
        
        XtX = np.dot(self.X_train.T, self.X_train).toarray()

        A, XtX, scaling, rescaling = self._calculate_sparsity_pattern(XtX=deepcopy(XtX), alpha=self.alpha, threshold=self.threshold, max_in_col=self.max_in_col)
        
        idx_diag = np.diag_indices(XtX.shape[0])
        XtX[idx_diag] += self.lmbda 
        
        # list L in the paper, sorted by item-counts per column, ties broken by item-popularities as reflected by np.diag(XtX)
        A_col_count = A.getnnz(axis=0)
        L = np.argsort(A_col_count + np.diag(XtX) / 2.0/ np.max(np.diag(XtX)))[::-1]  

        print("iterating through steps 1,2, and 4 in section 3.2 of the paper ...")
        todo_indicators = np.ones(A_col_count.shape[0])
        block_list = []   # list of blocks. Each block is a list of item-indices, to be processed in step 3 of the paper
        for i in L:
            if todo_indicators[i] == 1:
                n_i, _, vals=find(A[:,i])  # step 1 in paper: set n contains item i and its neighbors N
                sorted_ind = np.argsort(np.abs(vals))[::-1]
                n_i = n_i[sorted_ind]
                block_list.append(n_i) # list of items in the block, to be processed in step 3 below
                # remove possibly several items from list L, as determined by parameter r  
                d_count = max(1, int(np.ceil(len(n_i) * self.r)))
                d = n_i[:d_count] # set D, see step 2 in the paper
                todo_indicators[d] = 0  # step 4 in the paper        

        print("now step 3 in section 3.2 of the paper: iterating ...")
        # now the (possibly heavy) computations of step 3:
        # given that steps 1,2,4 are already done, the following for-loop could be implemented in parallel.   

        B_rowidx, B_colidx, B_val = [], [], []
        
        for n_i in block_list:
            #calculate dense solution for the items in set n
            if not len(n_i) > 0:
                continue
                
            B_block = self._faster_spd_inverse( XtX[np.ix_(n_i,n_i)] , dtype=dtype)
            #B_block = np.linalg.inv( XtX[np.ix_(n_i,n_i)] )
            
            B_block /= -np.diag(B_block)
                        
            # determine set D based on parameter r
            d_count = max(1,int(np.ceil(len(n_i)*self.r)))
            d = n_i[:d_count] # set D in paper
            
            # store the solution regarding the items in D
            block_idx = np.meshgrid(d,n_i)
            B_rowidx.extend(block_idx[1].flatten().tolist())
            B_colidx.extend(block_idx[0].flatten().tolist())
            B_val.extend(B_block[:, :d_count].flatten().tolist())
 

        del XtX
        
        square_shape = (self.nItems, self.nItems)
        print("final step: obtaining the sparse matrix B by averaging the solutions regarding the various sets D ...")
        B_sum = csc_matrix((B_val, (B_rowidx, B_colidx)), shape=square_shape, dtype=np.float32) 
        B_count = csc_matrix((np.ones(len(B_rowidx), dtype=np.float32), (B_rowidx, B_colidx)), shape=square_shape, dtype=np.float32) 
        B_count_values = find(B_count)[2] 
        B_sum_rowidx, B_sum_colidx, B_sum_values = find(B_sum)
        
        
        if len(B_sum_values) == len(B_count_values):
            B = csc_matrix((B_sum_values / B_count_values, (B_sum_rowidx, B_sum_colidx)), shape=square_shape, dtype=np.float32)
            
        else:
            # this can happen if an entry sufficiently close to zero is created in B_sum
            # in single precision, entries of B_block can also be zero
            B_count_rowidx, B_count_colidx, B_count_values = find(B_count) 
            B_sum_values = B_sum[B_count_rowidx, B_count_colidx].A1 # .A1 is the same as flattening
            B = csc_matrix((B_sum_values / B_count_values, (B_count_rowidx, B_count_colidx)), shape=square_shape, dtype=np.float32)
        
        B[idx_diag] = 0.0
        
        print("forcing the sparsity pattern of A onto B ...")
        B = csr_matrix( ( np.asarray(B[A.nonzero()]).flatten(),  A.nonzero() ), shape=B.shape, dtype=np.float32)
        B = diags(scaling, dtype=np.float32).dot(B).dot(diags(rescaling, dtype=np.float32))
        
        if store_dense:
            print('storing B matrix as dense')
            self.B = B.toarray()
        else:
            self.B = B
        
    
    def predict(self, data: pd.DataFrame, uid_str: str, sid_str: str, 
                k: int, return_scores: bool = False):
        if not set([uid_str, sid_str]) <= set(data.columns):
            raise ValueError(f'{uid_str} or {sid_str} not present in data columns')
            
        if self.store_dense:
            B = csr_matrix(self.B)            
            
        self.X_test = self._parse_data(data, uid_str, sid_str)
        
        if self.store_dense:
            pred = np.dot(self.X_test, B) #self.pred
            del B
            
        else:
            pred = np.dot(self.X_test, self.B)
            
        # we dont want to predict items that the user already interacted with 
        pred = pred + (-np.inf * self.X_test)        
        
        self.topk = argpartition_row(pred, k)
        
        # we need to sort manually
        n_test = self.X_test.shape[0]
        sorted_idx = (-pred)[np.arange(n_test)[:, None], self.topk].toarray().argsort()
        self.topk = self.topk[np.arange(n_test)[:, None], sorted_idx]
        return self.topk
        
        
class ADMM_slim(NumpyRecommender):
    """
    Code adapted from: http://www.cs.columbia.edu/~jebara/papers/wsdm20_ADMM.pdf
    More details on the theory can also be found in this paper.
    """
    def __init__(self, lambda1, lambda2, rho=10000, t=10, tau=2, force_nonneg=False, eps_rel=1e-3, eps_abs=1e-3):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.rho = rho
        self.t = t
        self.tau = tau
        self.force_nonneg = force_nonneg
        self.eps_rel = eps_rel
        self.eps_abs = eps_abs
        
        
    def _upper_triangular_to_symmetric(self, upper_triang):
        upper_triang += np.triu(upper_triang, k=1).T

        
    def _faster_spd_inverse(self, matrix):
        cholesky_decomp = cholesky(matrix)
        inv, info = spotri(cholesky_decomp)
        if info != 0:
            raise ValueError('spotri failed on input')
        self._upper_triangular_to_symmetric(inv)
        return inv
    
    
    def _softthreshold_nonneg(self, B, Gamma):
        if self.lambda1 == 0:
            if self.force_nonneg:
                return np.abs(B)
            else:
                return B
        else:
            x = B + Gamma / self.rho
            threshold = self.lambda1 / self.rho
            if self.force_nonneg:
                return np.where(threshold < x, x - threshold, 0)
            else:
                return np.where(threshold < x, x - threshold,
                                np.where(x < - threshold, x + threshold, 0))
            
    def _check_convergence(self, B, C, C_old, Gamma):
        B_norm = np.linalg.norm(B)
        C_norm = np.linalg.norm(C)
        Gamma_norm = np.linalg.norm(Gamma)

        eps_primal = self.eps_abs * self.nItems + self.eps_rel * np.max([B_norm, C_norm])
        eps_dual = self.eps_abs * self.nItems + self.eps_rel * Gamma_norm

        R_primal_norm = np.linalg.norm(B - C)
        R_dual_norm = np.linalg.norm(C  - C_old) * self.rho

        converged = R_primal_norm < eps_primal and R_dual_norm < eps_dual
        return converged, R_primal_norm, R_dual_norm, eps_primal, eps_dual
    
    
    def fit(self, train_data: pd.DataFrame, uid_str: str, sid_str: str, nItems: int, n_iter, check_convergence=False, adjust_rho=False):
        self.n_iter = n_iter
        self.adjust_rho = adjust_rho
        self.check_convergence = check_convergence
        self.nItems = nItems
        self.X_train = self._parse_data(train_data, uid_str, sid_str)
        
        XtX = np.dot(self.X_train.T, self.X_train).toarray()
        diag_indices = np.diag_indices(self.nItems)
        XtX[diag_indices] += self.lambda2 + self.rho
        P = self._faster_spd_inverse(XtX)
        XtX[diag_indices] -= self.lambda2 + self.rho
        B_aux = - P * (self.lambda2 + self.rho) # same as -P * (np.ones(self.nItems) * * (self.lambda2 + self.rho))
        B_aux[diag_indices] += 1.

        diag_P = np.diag(P)

        # initialize
        Gamma = np.zeros(XtX.shape, dtype=np.float32)
        C = np.zeros(XtX.shape, dtype=np.float32)

        # iterate until convergence
        for i in range(self.n_iter):
            print(f"Starting iteration {i}")
            if self.check_convergence:
                C_old = C.copy()
            B_tilde = B_aux + P.dot(self.rho * C - Gamma)
            gamma = np.diag(B_tilde) / diag_P
            B = B_tilde - P * gamma
            C = self._softthreshold_nonneg(B + Gamma/self.rho, self.lambda1/self.rho)

            print(f'Current Sparsity:{np.sum(C != 0) / (C.shape[0] * C.shape[1]) * 100:.3f}%')
            Gamma += self.rho * (B - C)
            
            if self.check_convergence:
                self.converged, R_primal_norm, R_dual_norm, eps_primal, eps_dual = self._check_convergence(B, C, C_old, Gamma)
                print(f'Primal Residual Norm: {R_primal_norm}, eps Primal: {eps_primal}\n' +
                      f'Dual Residual Norm: {R_dual_norm}, eps Dual: {eps_dual}')

                if adjust_rho:
                    if R_primal_norm > self.t * R_dual_norm:
                        self.rho = self.rho * self.tau
                        print(f'Increased rho by factor {self.tau}')

                    elif R_dual_norm > self.t * R_primal_norm:
                        self.rho = self.rho / self.tau
                        print(f'Decreased rho by factor {self.tau}')

                print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')

                if self.converged:
                    break
                            
        self.C = C
        
    
    
    def predict(self, data: pd.DataFrame, uid_str: str, sid_str: str, k: int):
        if not set([uid_str, sid_str]) <= set(data.columns):
            raise ValueError(f'{uid_str} or {sid_str} not present in data columns')
            
        self.k = k
        self.X_test = self._parse_data(data, uid_str, sid_str)
        X_rowidx, X_colidx, _ = find(self.X_test)
        X_test_dense = self.X_test.toarray()
        pred = np.dot(X_test_dense, self.C) 
        del X_test_dense 
            
        # we dont want to predict items that the user already interacted with 
        pred[X_rowidx, X_colidx] = -np.inf
        
        # top k items most popular items for each user
        self.topk = argpartition(-pred, self.k - 1, axis=1)[:, :self.k]
        
        # we need to sort manually
        n_test = self.X_test.shape[0]
        sorted_idx = (-pred)[np.arange(n_test)[:, None], self.topk].argsort()
        self.topk = self.topk[np.arange(n_test)[:, None], sorted_idx]
        return self.topk
    
    
    
class ADMM_EDLAE(NumpyRecommender):
    """
    Compared to the normal SLIM model, this just uses the slightly different regularization term of the EDLAE model. (better results).
    """
    def __init__(self, lambda1, lambda2, p, rho=10000, t=10, tau=2, force_nonneg=False, eps_rel=1e-3, eps_abs=1e-3):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.p = p
        self.rho = rho
        self.t = t
        self.tau = tau
        self.force_nonneg = force_nonneg
        self.eps_rel = eps_rel
        self.eps_abs = eps_abs
        
        
    def _upper_triangular_to_symmetric(self, upper_triang):
        upper_triang += np.triu(upper_triang, k=1).T

        
    def _faster_spd_inverse(self, matrix):
        cholesky_decomp = cholesky(matrix)
        inv, info = spotri(cholesky_decomp)
        if info != 0:
            raise ValueError('spotri failed on input')
        self._upper_triangular_to_symmetric(inv)
        return inv
    
    
    def _softthreshold_nonneg(self, B, Gamma):
        if self.lambda1 == 0:
            if self.force_nonneg:
                return np.abs(B)
            else:
                return B
        else:
            x = B + Gamma / self.rho
            threshold = self.lambda1 / self.rho
            if self.force_nonneg:
                return np.where(threshold < x, x - threshold, 0)
            else:
                return np.where(threshold < x, x - threshold,
                                np.where(x < - threshold, x + threshold, 0))
            
    def _check_convergence(self, B, C, C_old, Gamma):
        B_norm = np.linalg.norm(B)
        C_norm = np.linalg.norm(C)
        Gamma_norm = np.linalg.norm(Gamma)

        eps_primal = self.eps_abs * self.nItems + self.eps_rel * np.max([B_norm, C_norm])
        eps_dual = self.eps_abs * self.nItems + self.eps_rel * Gamma_norm

        R_primal_norm = np.linalg.norm(B - C)
        R_dual_norm = np.linalg.norm(C  - C_old) * self.rho

        converged = R_primal_norm < eps_primal and R_dual_norm < eps_dual
        return converged, R_primal_norm, R_dual_norm, eps_primal, eps_dual
    
    
    def fit(self, train_data: pd.DataFrame, uid_str: str, sid_str: str, nItems: int, n_iter, check_convergence=False, adjust_rho=False, return_info=False):
        start = time.time()
        self.n_iter = n_iter
        self.adjust_rho = adjust_rho
        self.check_convergence = check_convergence
        self.nItems = nItems
        self.X_train = self._parse_data(train_data, uid_str, sid_str)
        
        XtX = np.dot(self.X_train.T, self.X_train).toarray()
        XtX_diag = deepcopy(np.diag(XtX))
        diag_indices = np.diag_indices(self.nItems)
        diag_entries = self.p / (1 - self.p) * XtX_diag + self.lambda2 + self.rho
        XtX[diag_indices] += diag_entries
        P = self._faster_spd_inverse(XtX)
        #XtX[diag_indices] -= diag_entries
        del XtX
        B_aux = - P * (diag_entries) # same as -P * (np.ones(self.nItems) * (self.lambda2 + self.rho))
        B_aux[diag_indices] += 1.

        diag_P = np.diag(P)

        # initialize
        Gamma = np.zeros(P.shape, dtype=np.float32)
        C = np.zeros(P.shape, dtype=np.float32)

        # iterate until convergence
        for i in range(self.n_iter):
            print(f"Starting iteration {i}")
            if self.check_convergence:
                C_old = C.copy()
            B_tilde = B_aux + P.dot(self.rho * C - Gamma)
            gamma = np.diag(B_tilde) / diag_P
            B = B_tilde - P * gamma
            
            if i == 0:
                self.closed_form = B
                
            C = self._softthreshold_nonneg(B + Gamma/self.rho, self.lambda1/self.rho)

            print(f'Current Sparsity:{np.sum(C != 0) / (C.shape[0] * C.shape[1]) * 100:.3f}%')
            Gamma += self.rho * (B - C)
            
            if self.check_convergence:
                self.converged, R_primal_norm, R_dual_norm, eps_primal, eps_dual = self._check_convergence(B, C, C_old, Gamma)
                print(f'Primal Residual Norm: {R_primal_norm}, eps Primal: {eps_primal}\n' +
                      f'Dual Residual Norm: {R_dual_norm}, eps Dual: {eps_dual}')

                if adjust_rho:
                    if R_primal_norm > self.t * R_dual_norm:
                        self.rho = self.rho * self.tau
                        print(f'Increased rho by factor {self.tau}')

                    elif R_dual_norm > self.t * R_primal_norm:
                        self.rho = self.rho / self.tau
                        print(f'Decreased rho by factor {self.tau}')

                print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')

                if self.converged:
                    break
                            
               
        self.C = C
        end = time.time()    
        print(f'Training time: {(end-start) / 60:.2f}min')
        
        if return_info:
            final_sparsity = np.sum(C != 0) / (C.shape[0] * C.shape[1]) * 100
            if self.check_convergence:
                return final_sparsity, R_primal_norm, R_dual_norm, eps_primal, eps_dual
            else: 
                return final_sparsity
    
    
    def predict(self, data: pd.DataFrame, uid_str: str, sid_str: str, k: int):
        if not set([uid_str, sid_str]) <= set(data.columns):
            raise ValueError(f'{uid_str} or {sid_str} not present in data columns')
            
        self.k = k
        self.X_test = self._parse_data(data, uid_str, sid_str)
        X_rowidx, X_colidx, _ = find(self.X_test)
        X_test_dense = self.X_test.toarray()
        pred = np.dot(X_test_dense, self.C) 
        del X_test_dense 
            
        # we dont want to predict items that the user already interacted with 
        pred[X_rowidx, X_colidx] = -np.inf
        
        # top k items most popular items for each user
        self.topk = argpartition(-pred, self.k - 1, axis=1)[:, :self.k]
        
        # we need to sort manually
        n_test = self.X_test.shape[0]
        sorted_idx = (-pred)[np.arange(n_test)[:, None], self.topk].argsort()
        self.topk = self.topk[np.arange(n_test)[:, None], sorted_idx]
        return self.topk


        

    
    