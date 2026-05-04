# -*- coding: utf-8 -*-
"""
Build local data files for anterior IRN and PARN
"""

import os
import pickle

import numpy as np

import params

from ABC_toolbox import cell_funcs, data_nav, gene_funcs, ABC_utils

# %% load IRN and PARN MERFISH metadata, remove non-neuronal cells

roi = ['MRN', 'RR']
level = 'structure'

MERFISH_meta = data_nav.load_meta('MERFISH')

MERFISH_meta = data_nav.extract_MERFISH_meta(MERFISH_meta, 
                                              roi, 
                                              kind='restrict',
                                              category='anatomy',
                                              level=level)

non_neuronal = ['30 Astro-Epen', '31 OPC-Oligo', '32 OEC', '33 Vascular', '34 Immune']

MERFISH_meta = data_nav.extract_MERFISH_meta(MERFISH_meta, 
                                              non_neuronal, 
                                              kind='remove',
                                              category='taxonomy',
                                              level='class')

# remove junk column
MERFISH_meta.drop(columns=["Unnamed: 0"], inplace=True)

# %% cut rare cell types from MERFISH metadata

# minimum required number of cells for a cluster to be considered legit
# phantom clusters may be present due to cell type misassignment or inaccurate
# allen ccf registration
# default is 3, logic being once is obviously a mistake, twice very well could 
# also happen by mistake, but 3 is when it starts to become possible that the 
# type could legitimately be present 
min_cells = 3

clu, clu_count = np.unique(MERFISH_meta["cluster"], return_counts=True)

bool_mask = (clu_count < min_cells)
bool_mask = np.invert(bool_mask)
keep_clu = clu[bool_mask]
bool_mask = np.isin(MERFISH_meta["cluster"], keep_clu)

MERFISH_meta = MERFISH_meta.iloc[MERFISH_meta.index[bool_mask]]
MERFISH_meta.reset_index(drop=True, inplace=True)

# # %% fetch scRNAseq metadata and raw counts for antIRN-PARN

# clusters = np.unique(MERFISH_meta["cluster"].values)

# scRNAseq_meta = data_nav.load_meta('scRNAseq')
# scRNAseq_data = data_nav.fetch_scRNAseq(clusters, scRNAseq_meta, form='raw', neur_frac=0.1)

# scRNAseq_meta, scRNAseq_raw = scRNAseq_data
# scRNAseq_meta.reset_index(drop=True, inplace=True)
# scRNAseq_raw = scRNAseq_raw.todense()
# scRNAseq_raw = scRNAseq_raw.astype(np.uint16)

# # %% cut cell types with inadequate samples for k-fold cross validation or supercell bootstrapping

# # this should be set to whichever is larger: intended k in k-fold cross 
# # validation, or intended k in supercell bootstrapping
# min_cells = 5

# clu, clu_count = np.unique(scRNAseq_meta["cluster"], return_counts=True)

# bool_mask = (clu_count < min_cells)
# bool_mask = np.invert(bool_mask)
# keep_clu = clu[bool_mask]
# bool_mask = np.isin(scRNAseq_meta["cluster"], keep_clu)

# scRNAseq_raw = scRNAseq_raw[bool_mask,:]
# scRNAseq_meta = scRNAseq_meta.iloc[scRNAseq_meta.index[bool_mask]]
# scRNAseq_meta.reset_index(drop=True, inplace=True)

# # cut those cells from the MERFISH metadata too
# bool_mask = np.isin(MERFISH_meta["cluster"], keep_clu)
# MERFISH_meta = MERFISH_meta.iloc[MERFISH_meta.index[bool_mask]]
# MERFISH_meta.reset_index(drop=True, inplace=True)

# # %% write scRNAseq data and MERFISH frequencies

# # save scRNAseq raw expression data
# np.save(os.path.join(params.local_data_dir, "MO-scRNAseq-raw"), scRNAseq_raw)

# # save scRNAseq metadata
# scRNAseq_meta.to_csv(os.path.join(params.local_data_dir, "MO-scRNAseq-meta.csv"),
#                                   index=False)

# # save MERFISH frequencies
# MERFISH_freqs = cell_funcs.calc_frac_per_type(MERFISH_meta)
# with open(os.path.join(params.local_data_dir, "MO-MERFISH-freqs.pkl"), 'wb') as handle:
#     pickle.dump(MERFISH_freqs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# # save normalized scRNaseq data
# exp_norm = gene_funcs.normalize_counts_to_median(scRNAseq_raw)
# np.save(os.path.join(params.local_data_dir, "MO-scRNAseq-norm"), exp_norm)

# %%

import pandas as pd

MRN_cluster_freqs = cell_funcs.calc_frac_per_type(MERFISH_meta, level='cluster')
MRN_supertype_freqs = cell_funcs.calc_frac_per_type(MERFISH_meta, level='supertype')
MRN_subclass_freqs = cell_funcs.calc_frac_per_type(MERFISH_meta, level='subclass')
MRN_class_freqs = cell_funcs.calc_frac_per_type(MERFISH_meta, level='class')

MRN_cluster_freqs_df = pd.DataFrame(data={'cluster': list(MRN_cluster_freqs.keys()), 'proportion': list(MRN_cluster_freqs.values())})
MRN_supertype_freqs_df = pd.DataFrame(data={'supertype': list(MRN_supertype_freqs.keys()), 'proportion': list(MRN_supertype_freqs.values())})
MRN_subclass_freqs_df = pd.DataFrame(data={'subclass': list(MRN_subclass_freqs.keys()), 'proportion': list(MRN_subclass_freqs.values())})
MRN_class_freqs_df = pd.DataFrame(data={'class': list(MRN_class_freqs.keys()), 'proportion': list(MRN_class_freqs.values())})

MRN_cluster_freqs_df.to_csv("MRN_cluster_freqs.csv", index=False)
MRN_supertype_freqs_df.to_csv("MRN_supertype_freqs.csv", index=False)
MRN_subclass_freqs_df.to_csv("MRN_subclass_freqs.csv", index=False)
MRN_class_freqs_df.to_csv("MRN_class_freqs.csv", index=False)

# %%

thresh = 0.001
good_clusters = []
for key, value in MRN_cluster_freqs.items():
    if value > thresh:
        good_clusters.append(key)
        
# cut those cells from the MERFISH metadata too
bool_mask = np.isin(MERFISH_meta["cluster"], good_clusters)
MERFISH_meta = MERFISH_meta.iloc[MERFISH_meta.index[bool_mask]]
MERFISH_meta.reset_index(drop=True, inplace=True)

freqs = cell_funcs.calc_frac_per_type(MERFISH_meta, level='cluster')
subclass_freqs = cell_funcs.recalc_freqs(freqs, new_level='subclass')

# %%

# inhib = ['06 CTX-CGE GABA', '07 CTX-MGE GABA', '08 CNU-MGE GABA']

# MERFISH_meta_inhib = data_nav.extract_MERFISH_meta(MERFISH_meta, 
#                                                    inhib, 
#                                                    kind='restrict',
#                                                    category='taxonomy',
#                                                    level='class')

# excit = ['01 IT-ET Glut', '02 NP-CT-L6b Glut']

# MERFISH_meta_excit = data_nav.extract_MERFISH_meta(MERFISH_meta, 
#                                                    excit, 
#                                                    kind='restrict',
#                                                    category='taxonomy',
#                                                    level='class')

# MO_cluster_freqs = cell_funcs.calc_frac_per_type(MERFISH_meta_excit, level='cluster')
# MO_supertype_freqs = cell_funcs.calc_frac_per_type(MERFISH_meta_excit, level='supertype')
# MO_subclass_freqs = cell_funcs.calc_frac_per_type(MERFISH_meta_excit, level='subclass')
# MO_class_freqs = cell_funcs.calc_frac_per_type(MERFISH_meta_excit, level='class')

# %%

freqs = cell_funcs.calc_frac_per_type(MERFISH_meta, level='cluster')
old_freqs = freqs.copy()

inhib_kws = ["Vip Gaba", "Sncg Gaba", "Lamp5 Gaba", "Lamp5 Lhx6", 
             "Pvalb chandelier", "Pvalb Gaba", "Sst Gaba", "Sst Chodl"]

inhib_scale = 2

inhib_sum = 0
excit_sum = 0
for key, val in freqs.items():
    
    check_inhib = []
    for inhib_kw in inhib_kws:
        check_inhib.append(inhib_kw in key)
        
    if any(check_inhib):
        freqs[key] = inhib_scale * val
        inhib_sum += inhib_scale * val
    else:
        excit_sum += val
    
excit_target_sum = 1 - inhib_sum
s = excit_target_sum / excit_sum

for key, val in freqs.items():
    
    check_inhib = []
    for inhib_kw in inhib_kws:
        check_inhib.append(inhib_kw in key)
        
    if any(check_inhib):
        pass
    else:
        freqs[key] = s * val


# %% fetch scRNAseq metadata and raw counts for antIRN-PARN

clusters = np.unique(list(freqs.keys()))

scRNAseq_meta = data_nav.load_meta('scRNAseq')
scRNAseq_data = data_nav.fetch_scRNAseq(clusters, scRNAseq_meta, form='raw')

scRNAseq_meta, scRNAseq_raw = scRNAseq_data
scRNAseq_meta.reset_index(drop=True, inplace=True)
scRNAseq_raw = scRNAseq_raw.todense()
scRNAseq_raw = scRNAseq_raw.astype(np.uint16)

# %% cut cell types with inadequate samples for k-fold cross validation or supercell bootstrapping

# this should be set to whichever is larger: intended k in k-fold cross 
# validation, or intended k in supercell bootstrapping
min_cells = 5

clu, clu_count = np.unique(scRNAseq_meta["cluster"], return_counts=True)

bool_mask = (clu_count < min_cells)
bool_mask = np.invert(bool_mask)
keep_clu = clu[bool_mask]
bool_mask = np.isin(scRNAseq_meta["cluster"], keep_clu)

scRNAseq_raw = scRNAseq_raw[bool_mask,:]
scRNAseq_meta = scRNAseq_meta.iloc[scRNAseq_meta.index[bool_mask]]
scRNAseq_meta.reset_index(drop=True, inplace=True)

# cut those cells from the MERFISH metadata too
bool_mask = np.isin(MERFISH_meta["cluster"], keep_clu)
MERFISH_meta = MERFISH_meta.iloc[MERFISH_meta.index[bool_mask]]
MERFISH_meta.reset_index(drop=True, inplace=True)

# %% write scRNAseq data and MERFISH frequencies

# save scRNAseq raw expression data
np.save(os.path.join(params.local_data_dir, "MRN-scRNAseq-raw"), scRNAseq_raw)

# save scRNAseq metadata
scRNAseq_meta.to_csv(os.path.join(params.local_data_dir, "MRN-scRNAseq-meta.csv"),
                                  index=False)

# save MERFISH frequencies
MERFISH_freqs = cell_funcs.calc_frac_per_type(MERFISH_meta)
with open(os.path.join(params.local_data_dir, "MRN-MERFISH-freqs.pkl"), 'wb') as handle:
    pickle.dump(MERFISH_freqs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# save normalized scRNaseq data
exp_norm = gene_funcs.normalize_counts_to_median(scRNAseq_raw)
np.save(os.path.join(params.local_data_dir, "MRN-scRNAseq-norm"), exp_norm)


    

# %% identifying guess of injection coordinates for anterior IRN and PARN

import matplotlib.pyplot as plt
import numpy as np

# there's a hard cutoff in posterior but not anterior, IRN/PARN appears to start in earnest around 10.5
# this aligns visually with disconnection of VII in CCF 
plt.hist(MERFISH_meta['x_ccf'].values, bins=30)

# A to P
x_cent = ((11.1 - 10.5) / 2) + 10.5
x_range = 11.1 - 10.5

# D to V
y_cent = ((max(MERFISH_meta['y_ccf'].values) - min(MERFISH_meta['y_ccf'].values))/2) + min(MERFISH_meta['y_ccf'].values)
y_range = max(MERFISH_meta['y_ccf'].values) - min(MERFISH_meta['y_ccf'].values)

x_cent = x_cent - 5.4
y_cent = y_cent - 0.44

# allen ccf to bregma
AP = x_cent * np.cos(0.0873) - y_cent * np.sin(0.0873)
DV = x_cent * np.sin(0.0873) + y_cent * np.cos(0.0873)

DV = DV * 0.9434

# https://community.brain-map.org/t/how-to-transform-ccf-x-y-z-coordinates-into-stereotactic-coordinates/1858
# https://www.sciencedirect.com/science/article/pii/S0092867420304025

midline = 11.4/2

left_cells = (MERFISH_meta['z_ccf'].values <= midline)
right_cells = (MERFISH_meta['z_ccf'].values > midline)

left_cells = MERFISH_meta['z_ccf'].values[left_cells]
right_cells = MERFISH_meta['z_ccf'].values[right_cells]

z_cent_left = ((max(left_cells) - min(left_cells))/2) + min(left_cells)
z_cent_right = ((max(right_cells) - min(right_cells))/2) + min(right_cells)

dist_left = abs(z_cent_left - midline)
dist_right = abs(z_cent_right - midline)

ML = np.mean([dist_left, dist_right])

# surface of brain at that AP and ML is at roughly 1.18 DV 

DV = DV - 1.18

# %%

bound = data_nav.load_image_boundaries()

with open('parcellation_dict.pickle', 'rb') as handle:
    parc_dict = pickle.load(handle)
    
# %%

import matplotlib.pyplot as plt

# Renormalize ratios using base ratios dict (ratios of all clusters) to any 
# arbitrary combination of taxonomic labels. I.e. ratios will be restricted to 
# only target labels and recalculated to sum to 1. Input labels must all be 
# from the same taxonomic level
def renorm_ratios(ratios, labels):
    
    level = ABC_utils.id_tax(labels)
    level = np.unique(level)
    
    if len(level) > 1:
        raise Exception("All taxonomic labels must be from the same taxonomic level")
        
    keys = list(ratios.keys())
    vals = list(ratios.values())
    
    if level != "cluster":
        
        match level:
            case "supertype":
                idx = 0
            case "subclass":
                idx = 1
            case "class":
                idx = 2
        
        for i, key in enumerate(keys):
            keys[i] = str(ABC_utils.fetch_cluster_tax(key)[idx])
        
    bool_mask = np.isin(keys, labels)

    keys = np.array(keys)[bool_mask]
    vals = np.array(vals)[bool_mask]

    vals = vals/sum(vals)

    merged_list = [(keys[i], vals[i]) for i in range(0, len(keys))]

    from collections import Counter

    c = Counter()
    for k, v in merged_list:
        c[k] += v
        
    renorm_ratios = dict(c)
    
    return renorm_ratios

# classes = [val for val in MO_cluster_freqs.keys() if "L6b CTX" in val] 
classes = [val for val in MO_cluster_freqs.keys()] 

class_ratios = renorm_ratios(MO_cluster_freqs, classes)
class_ratios = {k: v for k, v in sorted(class_ratios.items(), key=lambda item: item[1])}
class_color_dict = ABC_utils.load_colors_dict("cluster")
colors = [class_color_dict[x] for x in class_ratios.keys()]
    
x = np.array(list(class_ratios.values()))

fig, ax = plt.subplots()
ax.pie(x, colors=colors, wedgeprops=dict(width=0.25))
plt.savefig("ratios.png", transparent=True)

# %%

from scipy.stats import entropy
import matplotlib as mpl

MO_inhib_cluster_freqs = cell_funcs.calc_frac_per_type(MERFISH_meta_inhib, level='cluster')
MO_excit_cluster_freqs = cell_funcs.calc_frac_per_type(MERFISH_meta_excit, level='cluster')

S_inhib = entropy(list(MO_inhib_cluster_freqs.values()), base=2)
S_excit = entropy(list(MO_excit_cluster_freqs.values()), base=2)

plt.bar(['Excitatory Neurons', 'Inhibitory Neurons'], [S_excit, S_inhib], color=['deeppink', 'gold'])
plt.ylabel("Entropy (bits)")

mpl.rcParams['image.composite_image'] = False
plt.rcParams['svg.fonttype'] = 'none'
plt.tight_layout()

# %%

excit_prop = 186212/218166
inhib_prop = 31954/218166

excit_prop = excit_prop * 100
inhib_prop = inhib_prop * 100

plt.bar(['Excitatory', 'Inhibitory'], [excit_prop, inhib_prop], color=['deeppink', 'gold'])
plt.ylabel("Prevalence (%)")

mpl.rcParams['image.composite_image'] = False
plt.rcParams['svg.fonttype'] = 'none'
plt.tight_layout()


