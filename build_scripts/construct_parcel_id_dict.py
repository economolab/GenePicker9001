# -*- coding: utf-8 -*-
"""
construct_parcel_id_dict

2023-10-17

Jack Vincent

Takes in file CCFv3_annotation_ITKSNAP_labels.txt. Saves a dictionary where 
keys correspond to cluster parcellation indices, and values correspond to the 
associated brain region name.
"""

import os

import pandas as pd

from tqdm import tqdm

# %%

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# %%

path = r'U:\\eng_research_economo\\JPV\\allen-brain-cell-atlas\\metadata\\'
f = r'MERFISH-C57BL6J-638850-CCF\\20230830\\views\\cell_metadata_with_parcellation_annotation.csv'
cell_metadata = pd.read_csv(path + f, low_memory=False)

# %%

parcel_id_dict = {}

# iterate through cells
for i in tqdm(range(len(cell_metadata))):
    
    parcel_id = cell_metadata['parcellation_index'][i]
    
    if parcel_id not in parcel_id_dict:
        
        parcel_substruct = cell_metadata['parcellation_substructure'][i]
        parcel_struct = cell_metadata['parcellation_structure'][i]
        parcel_div = cell_metadata['parcellation_division'][i]
        parcel_cat = cell_metadata['parcellation_category'][i]
        parcel_organ = cell_metadata['parcellation_organ'][i]
        
        parcel_id_dict[parcel_id] = (parcel_substruct, 
                                     parcel_struct, 
                                     parcel_div, 
                                     parcel_cat,
                                     parcel_organ) 
    
# %% write dictionary to pickle file

import pickle

f = open("parcel_id_dict.pkl","wb")
pickle.dump(parcel_id_dict,f)
f.close()