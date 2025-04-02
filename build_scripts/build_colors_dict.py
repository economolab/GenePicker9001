# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 13:30:54 2024

@author: jpv88
"""

import pickle

import matplotlib.colors as mcolors
import numpy as np

from utils import data_nav

meta = data_nav.load_meta('scRNAseq')

colors_dict = {}
class_dict = {}
subclass_dict = {}
supertype_dict = {}
cluster_dict = {}

classes = np.unique(meta['class'])
subclasses = np.unique(meta['subclass'])
supertypes = np.unique(meta['supertype'])
clusters = np.unique(meta['cluster'])

for class_ in classes:
    idx = (meta['class'] == class_).argmax()
    hexa = meta['class_color'][idx]
    class_dict[class_] = mcolors.to_rgb(hexa)
    
for subclass in subclasses:
    idx = (meta['subclass'] == subclass).argmax()
    hexa = meta['subclass_color'][idx]
    subclass_dict[subclass] = mcolors.to_rgb(hexa)
    
for supertype in supertypes:
    idx = (meta['supertype'] == supertype).argmax()
    hexa = meta['supertype_color'][idx]
    supertype_dict[supertype] = mcolors.to_rgb(hexa)
    
for cluster in clusters:
    idx = (meta['cluster'] == cluster).argmax()
    hexa = meta['cluster_color'][idx]
    cluster_dict[cluster] = mcolors.to_rgb(hexa)
    
colors_dict['class_dict'] = class_dict
colors_dict['subclass_dict'] = subclass_dict
colors_dict['supertype_dict'] = supertype_dict
colors_dict['cluster_dict'] = cluster_dict

with open('colors_dict.pkl', 'wb') as handle:
    pickle.dump(colors_dict, handle)

