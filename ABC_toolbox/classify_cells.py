# -*- coding: utf-8 -*-
"""
Utilities for classifying cell types
"""

import numpy as np
import pandas as pd
import scanpy as sc
import scipy as sp

from anndata import AnnData
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from ABC_toolbox import ABC_utils, cell_funcs

# %%

# train a cell type classifier
def train_classifier(X_train, y_train, clf_method='knn'):

    match clf_method:
        case 'knn':
            clf = KNeighborsClassifier(metric='cosine')

    clf.fit(X_train, y_train)

    return clf

# test a cell type classifier
def test_classifier(X_test, y_test, clf, freqs):
        
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    
    labels = clf.classes_
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm = cm.astype(np.float64)
    
    ginis = []
    for i in range(cm.shape[0]):
        ginis.append(gini(cm[i, :]))
    
    sparsity = np.average(ginis, weights=[freqs[label] for label in labels])
    
    return acc, sparsity, cm, labels
    

# make sure bootstrapped data is ready to be input to a classifier
def preprocess_data(exp, meta, freqs, clu_mapping=None):
    
    if isinstance(exp, sp.sparse.sparray):
        exp = exp.todense()
        
    # convert freqs to cluster mapping specific freqs
    if clu_mapping is not None:
        freqs = cell_funcs.freqs_to_cm_freqs(freqs)

    y = meta["cluster"]

    if not isinstance(exp, np.ndarray):
        exp = exp.to_numpy()
    else:
        pass 
        
    exp = pd.DataFrame(data=exp)

    adata = AnnData(exp)

    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    
    all_genes = ABC_utils.load_gene("scRNAseq")
    adata.var_names = all_genes
    
    X = adata

    return X, y, freqs 


def gini(array):
    """Calculate the Gini coefficient of a numpy array."""

    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm

    array = array.flatten()  # all values are treated equally, arrays must be 1d

    if np.amin(array) < 0:
        array -= np.amin(array)  # values cannot be negative

    array += 0.0000001  # values cannot be 0
    array = np.sort(array)  # values must be sorted

    index = np.arange(1, array.shape[0]+1)  # index per array element
    n = array.shape[0]  # number of array elements

    # Gini coefficient
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

# cross-validate a classifier
def cross_val_classifier(meta, exp, freqs,
                         genes=None, clf_method='knn', n_splits=5,
                         n_cells_boot=5000):
    
    """
    Parameters
    ----------
    meta : TYPE
        DESCRIPTION.
    exp : TYPE
        DESCRIPTION.
    freqs : TYPE
        DESCRIPTION.
    genes : TYPE, optional
        DESCRIPTION. The default is None.
    clf_method : TYPE, optional
        DESCRIPTION. The default is 'knn'.
    clu_mapping : TYPE, optional
        DESCRIPTION. The default is None.
    n_splits : TYPE, optional
        DESCRIPTION. The default is 5.

    Returns
    -------
    None.

    """
    
    # restrict to only a subset of genes
    if genes is not None:
        all_genes = ABC_utils.load_gene("scRNAseq")
        idx = np.where(np.isin(all_genes, genes))[0]
        exp = exp[:,idx]

    
    sc.pp.pca(exp)
    
    pcs = exp.obsm['X_pca']

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    splits = skf.split(np.zeros(meta.shape[0]), meta)
    indices = []
    for split in splits:
        indices.append(split[1])
    
    res = {}
    res["accs"] = []
    res["sparsities"] = []
    res["cms"] = []
    res["labels"] = []
    for i in tqdm(range(n_splits), desc='Train-test splits...'):
        train = indices[:i] + indices[i+1:]
        train = np.concatenate(train)
        train = np.sort(train)
        test = indices[i]
        
        X_train = pcs[train,:]
        y_train = meta.iloc[train]
        X_test = pcs[test,:]
        y_test = meta.iloc[test]
        
        clf = train_classifier(X_train, y_train, 
                               clf_method=clf_method)
        
        acc, sparsity, cm, labels = test_classifier(X_test, y_test, clf, freqs)
        
        res["accs"].append(acc)
        res["sparsities"].append(sparsity)
        res["cms"].append(cm)
        res["labels"].append(labels)
        
    return res
