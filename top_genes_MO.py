# -*- coding: utf-8 -*-
"""
Created on Sat May  2 14:22:25 2026

@author: jpv88
"""

# %%

import numpy as np

# top_genes_MO = ['Alcam', 'Brinp3', 'Cacna2d3', 'Calb1', 'Cbln2', 'Cck', 'Cdh13', 
#                 'Chrm3', 'Cnr1', 'Cpne4', 'Galntl6', 'Grik1', 'Kcnc2', 'Kcnk2', 'Meis2', 
#                 'Pcp4', 'Slc24a2', 'Spon1', 'Stxbp6', 'Thsd7a']

top_genes_MO = ['Cacna2d3', 'Calb1', 'Cbln2', 'Cck', 'Cnr1', 'Cpne4', 
                'Galntl6', 'Grik1', 'Kcnk2', 'Meis2', 'Spon1']

top_genes_MO = np.array(top_genes_MO)

np.save('top_genes_MO.npy', top_genes_MO)


