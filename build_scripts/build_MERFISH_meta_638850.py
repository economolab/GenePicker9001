# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 18:19:56 2024

@author: jpv88
"""

import os
import pickle

import pandas as pd

from tqdm import tqdm

# %%

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# %% load MERFISH metadata

path = r'U:\\eng_research_economo\\JPV\\allen-brain-cell-atlas\\metadata\\'

f_638850 = r'MERFISH-C57BL6J-638850-CCF\\20231215\\views\\cell_metadata_with_parcellation_annotation.csv'

meta_638850 = pd.read_csv(path + f_638850, low_memory=False)
meta_638850.drop(meta_638850.index[meta_638850['parcellation_index'] == 0], 
                  inplace=True)
meta_638850.drop(meta_638850.index[meta_638850['parcellation_index'] == 987], 
                  inplace=True)
meta_638850.reset_index(drop=True, inplace=True)

# %%

path = r'U:\\eng_research_economo\\JPV\\allen-brain-cell-atlas\\metadata\\Allen-CCF-2020\\20230630\\views\\'
f = 'parcellation_to_parcellation_term_membership_acronym.csv'

parcel_table = pd.read_csv(path + f)

# %%

def write_parcellation_data(meta, parcel_table):
    
    parcel_idx = list(parcel_table['parcellation_index'].values)
    
    meta['organ'] = ''
    meta['category'] = ''
    meta['division'] = ''
    meta['structure'] = ''
    meta['substructure'] = '' 
    
    for i in tqdm(range(len(meta))):
        
        table_idx = parcel_idx.index(meta['parcellation_index'][i])
        parcel_data = parcel_table.iloc[table_idx]
        
        meta.at[i, 'organ'] = parcel_data['organ']
        meta.at[i, 'category'] = parcel_data['category']
        meta.at[i, 'division'] = parcel_data['division']
        meta.at[i, 'structure'] = parcel_data['structure']
        meta.at[i, 'substructure'] = parcel_data['substructure']
        
    return meta

def write_cluster_data(meta, cluster_table):
    
    cluster_cell_label = list(cluster_table['cell_label'].values)
    
    meta['class'] = ''
    meta['subclass'] = ''
    meta['supertype'] = ''
    meta['cluster'] = '' 
    
    for i in tqdm(range(len(meta))):
        
        table_idx = cluster_cell_label.index(meta['cell_label'][i])
        cluster_data = cluster_table.iloc[table_idx]
        
        meta.at[i, 'class'] = cluster_data['class']
        meta.at[i, 'subclass'] = cluster_data['subclass']
        meta.at[i, 'supertype'] = cluster_data['supertype']
        meta.at[i, 'cluster'] = cluster_data['cluster']
        
    return meta

columns_map = {'cell_label': 'cell_label',
               'x': 'x_reconstructed',
               'y': 'y_reconstructed',
               'z': 'z_reconstructed',
               'parcellation_index': 'parcellation_index',
               'organ': 'parcellation_organ',
               'category': 'parcellation_category',
               'division': 'parcellation_division',
               'structure': 'parcellation_structure',
               'substructure': 'parcellation_substructure',
               'class': 'class',
               'subclass': 'subclass',
               'supertype': 'supertype',
               'cluster': 'cluster'}

columns_map = {v: k for k, v in columns_map.items()}
    
# %%

meta_638850.rename(columns=columns_map, inplace=True)
meta_638850 = meta_638850[list(columns_map.values())].copy()
meta_638850.to_csv('MERFISH_meta_638550.csv')

