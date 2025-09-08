# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 13:55:11 2025

@author: jpv88
"""

import os

import numpy as np
import pandas as pd

top_n = 100
path = r'C:\Users\jpv88\Documents\GenePicker9001_data'
token = "gene_df_" + str(top_n)

all_files = os.listdir(path)
gene_df_fnames = [f for f in all_files if token in f]

gene_dfs = []
for fname in gene_df_fnames:
    gene_df = pd.read_csv(os.path.join(path, fname))
    gene_dfs.append(gene_df)
    
def find_tested_genes(gene_df_gas):
        
    tested_genes = []
    for gene_df in gene_df_gas:
        tested_genes.extend(gene_df['gene'].values)
        
    tested_genes = np.unique(tested_genes)
    
    return tested_genes
    
def score_genes(gene_df_gas, tested_genes):
    
    gene_scores = []
    
    for gene in tested_genes:
        scores = []
        occurences = []
        
        for gene_df in gene_df_gas:
            
            if gene in gene_df['gene'].values:
                scores.append(gene_df['fitnesses'][gene_df['gene'] == gene].values[0])
                occurences.append(gene_df['occurences'][gene_df['gene'] == gene].values[0])
                
        gene_scores.append(np.average(scores, weights=occurences))
        
    return gene_scores

tested_genes = find_tested_genes(gene_dfs)
gene_scores = score_genes(gene_dfs, tested_genes)

final_gene_df = {}
final_gene_df['gene'] = tested_genes
final_gene_df['score'] = gene_scores
final_gene_df = pd.DataFrame(data=final_gene_df)

fname = f"final_gene_df_{top_n}.csv"
final_gene_df.to_csv(os.path.join(path, fname))

# %%

final_gene_df = pd.read_csv(os.path.join(path, fname))
final_gene_df_filt = final_gene_df.nlargest(50, 'score')
top_genes = final_gene_df_filt['gene'].values



