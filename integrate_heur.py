# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 19:30:25 2025

@author: jpv88
"""

import os
import sys

from tqdm import tqdm

import numpy as np
import pandas as pd

ntop = 2000

path = r'C:\Users\jpv88\Documents\GitHub\GenePicker9001'
token = "gene_df_" + str(ntop) +"_"

all_files = os.listdir(path)
gene_df_fnames = [f for f in all_files if token in f]

gene_dfs = []
for fname in gene_df_fnames:
    gene_df = pd.read_csv(os.path.join(path, fname))
    gene_dfs.append(gene_df)
    
rdr_weight = (1/3) 
var_E_weight = (2/3)
weights = [rdr_weight, var_E_weight]

for gene_df in gene_dfs:
    arr = np.column_stack((gene_df["rdr"], gene_df["var_E"]))
    scores = np.average(arr, axis=1, weights=weights)
    gene_df['score'] = scores

def find_tested_genes(gene_df_gas):
        
    tested_genes = []
    for gene_df in gene_df_gas:
        tested_genes.extend(gene_df['gene'].values)
        
    tested_genes = np.unique(tested_genes)
    
    return tested_genes
    
def score_genes(gene_df_gas, tested_genes):
    
    gene_scores = []
    
    for gene in tqdm(tested_genes):
        scores = []
        
        for gene_df in gene_df_gas:
            
            if gene in gene_df['gene'].values:
                scores.append(gene_df['score'][gene_df['gene'] == gene].values[0])
                
        gene_scores.append(np.average(scores))
        
    return gene_scores

tested_genes = find_tested_genes(gene_dfs)
gene_scores = score_genes(gene_dfs, tested_genes)

final_gene_df = {}
final_gene_df['gene'] = tested_genes
final_gene_df['score'] = gene_scores
final_gene_df = pd.DataFrame(data=final_gene_df)

fname = f"heur_gene_df_{ntop}.csv"
final_gene_df.to_csv(os.path.join(path, fname))

final_gene_df_filt = final_gene_df.nlargest(ntop, 'score')
top_genes = final_gene_df_filt["gene"].values
fname = f"top_genes_{ntop}.npy"
np.save(fname, top_genes)