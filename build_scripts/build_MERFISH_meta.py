# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 13:55:06 2023

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
f_Zhuang1 = r'Zhuang-ABCA-1-CCF\\20230830\\ccf_coordinates.csv'
f_Zhuang2 = r'Zhuang-ABCA-2-CCF\\20230830\\ccf_coordinates.csv'
f_Zhuang3 = r'Zhuang-ABCA-3-CCF\\20230830\\ccf_coordinates.csv'
f_Zhuang4 = r'Zhuang-ABCA-4-CCF\\20230830\\ccf_coordinates.csv'

meta_638850 = pd.read_csv(path + f_638850, low_memory=False)
meta_638850.drop(meta_638850.index[meta_638850['parcellation_index'] == 0], 
                  inplace=True)
meta_638850.drop(meta_638850.index[meta_638850['parcellation_index'] == 987], 
                  inplace=True)
meta_638850.reset_index(drop=True, inplace=True)

meta_Zhuang1 = pd.read_csv(path + f_Zhuang1, low_memory=False)
meta_Zhuang1.drop(meta_Zhuang1.index[meta_Zhuang1['parcellation_index'] == 0], 
                  inplace=True)
meta_Zhuang1.drop(meta_Zhuang1.index[meta_Zhuang1['parcellation_index'] == 987], 
                  inplace=True)
meta_Zhuang1.reset_index(drop=True, inplace=True)

meta_Zhuang2 = pd.read_csv(path + f_Zhuang2, low_memory=False)
meta_Zhuang2.drop(meta_Zhuang2.index[meta_Zhuang2['parcellation_index'] == 0], 
                  inplace=True)
meta_Zhuang2.drop(meta_Zhuang2.index[meta_Zhuang2['parcellation_index'] == 987], 
                  inplace=True)
meta_Zhuang2.reset_index(drop=True, inplace=True)

meta_Zhuang3 = pd.read_csv(path + f_Zhuang3, low_memory=False)
meta_Zhuang3.drop(meta_Zhuang3.index[meta_Zhuang3['parcellation_index'] == 0], 
                  inplace=True)
meta_Zhuang3.drop(meta_Zhuang3.index[meta_Zhuang3['parcellation_index'] == 987], 
                  inplace=True)
meta_Zhuang3.reset_index(drop=True, inplace=True)

meta_Zhuang4 = pd.read_csv(path + f_Zhuang4, low_memory=False)
meta_Zhuang4.drop(meta_Zhuang4.index[meta_Zhuang4['parcellation_index'] == 0], 
                  inplace=True)
meta_Zhuang4.drop(meta_Zhuang4.index[meta_Zhuang4['parcellation_index'] == 987], 
                  inplace=True)
meta_Zhuang4.reset_index(drop=True, inplace=True)

# %% load cell type metadata for Zhuang datasets

path_Zhuang1 = r'U:\\eng_research_economo\\JPV\\allen-brain-cell-atlas\\metadata\\Zhuang-ABCA-1\\20231215\\views\\'
path_Zhuang2 = r'U:\\eng_research_economo\\JPV\\allen-brain-cell-atlas\\metadata\\Zhuang-ABCA-2\\20231215\\views\\'
path_Zhuang3 = r'U:\\eng_research_economo\\JPV\\allen-brain-cell-atlas\\metadata\\Zhuang-ABCA-3\\20231215\\views\\'
path_Zhuang4 = r'U:\\eng_research_economo\\JPV\\allen-brain-cell-atlas\\metadata\\Zhuang-ABCA-4\\20231215\\views\\'

cluster_Zhuang1 = pd.read_csv(path_Zhuang1 + 'cell_metadata_with_cluster_annotation.csv')
cluster_Zhuang2 = pd.read_csv(path_Zhuang2 + 'cell_metadata_with_cluster_annotation.csv')
cluster_Zhuang3 = pd.read_csv(path_Zhuang3 + 'cell_metadata_with_cluster_annotation.csv')
cluster_Zhuang4 = pd.read_csv(path_Zhuang4 + 'cell_metadata_with_cluster_annotation.csv')

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

meta_Zhuang1 = write_parcellation_data(meta_Zhuang1, parcel_table)
meta_Zhuang1 = write_cluster_data(meta_Zhuang1, cluster_Zhuang1)

meta_Zhuang2 = write_parcellation_data(meta_Zhuang2, parcel_table)
meta_Zhuang2 = write_cluster_data(meta_Zhuang2, cluster_Zhuang2)

meta_Zhuang3 = write_parcellation_data(meta_Zhuang3, parcel_table)
meta_Zhuang3 = write_cluster_data(meta_Zhuang3, cluster_Zhuang3)

meta_Zhuang4 = write_parcellation_data(meta_Zhuang4, parcel_table)
meta_Zhuang4 = write_cluster_data(meta_Zhuang4, cluster_Zhuang4)

columns_map = {'cell_label': 'cell_label',
               'x': 'x_ccf',
               'y': 'y_ccf',
               'z': 'z_ccf',
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

meta_Zhuang1.rename(columns=columns_map, inplace=True)
meta_Zhuang2.rename(columns=columns_map, inplace=True)
meta_Zhuang3.rename(columns=columns_map, inplace=True)
meta_Zhuang4.rename(columns=columns_map, inplace=True)
    
# %%

meta_638850 = meta_638850[list(meta_Zhuang1.columns.values)].copy()

# %%

master_meta = pd.concat([meta_638850, meta_Zhuang1, meta_Zhuang2, meta_Zhuang3, meta_Zhuang4])
master_meta.reset_index(drop=True, inplace=True)
master_meta.to_csv('MERFISH_meta.csv')

