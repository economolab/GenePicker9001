# -*- coding: utf-8 -*-
"""
Build local data files for anterior IRN and PARN
"""

import os
import pickle

import numpy as np

import params

from ABC_toolbox import cell_funcs, data_nav, gene_funcs

# %% load IRN and PARN MERFISH metadata, remove non-neuronal cells

roi = ['MOp', 'MOs']
level = 'structure'

MERFISH_meta = data_nav.load_meta('MERFISH')

MERFISH_meta = data_nav.extract_MERFISH_meta(MERFISH_meta, 
                                              roi, 
                                              kind='restrict',
                                              category='anatomy',
                                              level=level)

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

# %% fetch scRNAseq metadata and raw counts for antIRN-PARN

clusters = np.unique(MERFISH_meta["cluster"].values)

scRNAseq_meta = data_nav.load_meta('scRNAseq')
scRNAseq_data = data_nav.fetch_scRNAseq(clusters, scRNAseq_meta, form='raw', neur_frac=0.1, nn_frac=0.1)

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
np.save(os.path.join(params.local_data_dir, "MO-scRNAseq-NN-raw"), scRNAseq_raw)

# save scRNAseq metadata
scRNAseq_meta.to_csv(os.path.join(params.local_data_dir, "MO-scRNAseq-NN-meta.csv"),
                                  index=False)

# save MERFISH frequencies
MERFISH_freqs = cell_funcs.calc_frac_per_type(MERFISH_meta)
with open(os.path.join(params.local_data_dir, "MO-MERFISH-NN-freqs.pkl"), 'wb') as handle:
    pickle.dump(MERFISH_freqs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# save normalized scRNaseq data
exp_norm = gene_funcs.normalize_counts_to_median(scRNAseq_raw)
np.save(os.path.join(params.local_data_dir, "MO-scRNAseq-NN-norm"), exp_norm)

