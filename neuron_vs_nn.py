# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 16:52:53 2025

@author: jpv88
"""

import logging
import os
import pygad
import random

import numpy as np
import pandas as pd

from tqdm import tqdm

import params

from ABC_toolbox import ABC_plot, ABC_utils, cell_funcs, classify_cells, gene_funcs, iterative_reclustering
from gene_filter import filt_genes
from genetic_algo import run_ga

# %%

meta = pd.read_csv(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-NN-meta.csv"), low_memory=False)
exp = np.load(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-NN-raw.npy"))
freqs = pd.read_pickle(os.path.join(params.local_data_dir, "antIRN-PARN-MERFISH-NN-freqs.pkl"))

exp_super, meta_super = cell_funcs.boot_super(exp, meta, k=5)

exp_super = gene_funcs.normalize_counts_to_median(exp_super)

# %%

meta_super.reset_index(drop=True, inplace=True)
non_neuronal = ['30 Astro-Epen', '31 OPC-Oligo', '32 OEC', '33 Vascular', '34 Immune']

nn_mask = np.isin(meta_super["class"], non_neuronal)
nn_index = list(meta_super.index[nn_mask])
neuron_index = list(meta_super.index[~nn_mask])

neur_nn = np.repeat('null', len(meta_super))
neur_nn[neuron_index] = "neuron"
neur_nn[nn_index] = "nn"

meta_super.insert(len(meta_super.columns), 'neur_nn', neur_nn)

# new_freqs = {}
# new_freqs["nn"] = 0
# new_freqs["neuron"] = 0

# # non-neuronal clusters start at 5206
# for key in freqs.keys():
#     if int(key[:4]) >= 5206:
#         new_freqs["nn"] += freqs[key]
#     else:
#         new_freqs["neuron"] += freqs[key]

# %%
gene_df, top_genes = filt_genes(exp_super, meta_super, freqs, top_n=1000, level='neur_nn')

# %%

meta = pd.read_csv(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-NN-meta.csv"), low_memory=False)
exp = np.load(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-NN-raw.npy"))
freqs = pd.read_pickle(os.path.join(params.local_data_dir, "antIRN-PARN-MERFISH-NN-freqs.pkl"))

exp_super, meta_super = cell_funcs.boot_super(exp, meta, k=5)

exp_super = gene_funcs.normalize_counts_to_median(exp_super)

# bootstrap distribution from scRNAseq that matches MERFISH frequencies
exp_boot, meta_boot = cell_funcs.bootstrap_scRNAseq(meta_super, exp_super, freqs, 
                                                    n=10000)

non_neuronal = ['30 Astro-Epen', '31 OPC-Oligo', '32 OEC', '33 Vascular', '34 Immune']

nn_mask = np.isin(meta_boot["class"], non_neuronal)
nn_index = list(meta_boot.index[nn_mask])
neuron_index = list(meta_boot.index[~nn_mask])

# %%

ABC_plot.plot_gene("Pcdh7", meta_boot, exp_boot, level='subclass')

# %%

neuron_exp = []
nn_exp = []
for j in tqdm(range(exp_boot.shape[1])):
    neuron_exp.append(np.mean(exp_boot[neuron_index,j]))
    nn_exp.append(np.mean(exp_boot[nn_index,j]))
    
data = {}
data["gene"] = ABC_utils.load_gene("scRNAseq")
data["neuron_exp"] = neuron_exp
data["nn_exp"] = nn_exp
data["ratio"] = np.array(neuron_exp) / np.array(nn_exp)

gene_df = pd.DataFrame(data=data)
gene_df.set_index('gene', inplace=True)



# %%

pan_neur = ["Cntnap2", "Nrg3", "Rbfox1", "Snhg11", "Atp1b1", "Ttc3", "Snap25",
            "Map1b", "Nap1l5", "Rtn1", "Ccser1", "Peg3", "Dpp6", "Camta1", "Zwint", 
            "Nrg1", "Sgcz", "Lrrtm4", "Kcnip4", "Lingo2", "Asic2", "Dpp10", "Tenm2", "Cdh18", "Ralyl",
            "Syt1", "Galntl6", "Snhg14", "Celf4", "Fgf12", "Rian", "Nsf", "Ywhag", "Cadps", "Atp1a3", "Lrfn5",
            "Cntn5", "Ndrg4", "Sntg1", "Ahi1", "Hs6st3", "Dscam", "Rbms3", "Xkr4", "Dab1",
            "Cntnap5a", "Cdr1os", "Map2", "Grm5", "Rims1", "Stxbp1", "Epha6", "Basp1", "Vsnl1",
            "Hcn1", "Pclo", "Brinp3", "Meis2", "Agbl4", "Tmsb10", "Rnf227"]

exp_boot, meta_boot = cell_funcs.filt_cells(exp_boot, meta_boot)

non_neuronal = ['30 Astro-Epen', '31 OPC-Oligo', '32 OEC', '33 Vascular', '34 Immune']
nn_mask = np.isin(meta_boot["class"], non_neuronal)
nn_index = list(meta_boot.index[nn_mask])
neuron_index = list(meta_boot.index[~nn_mask])

neur_nn = np.repeat('null', len(meta_boot))
neur_nn[neuron_index] = "neuron"
neur_nn[nn_index] = "nn"

meta_boot['cluster'] = neur_nn

freqs = cell_funcs.calc_frac_per_type(meta_boot)

exp_boot, meta_boot, cm_freqs = classify_cells.preprocess_data(exp_boot, meta_boot, freqs)

# %%

gene_df_ga, sol_df_ga, ga_instance = run_ga(meta_boot, exp_boot, cm_freqs, pan_neur, 
                                            init_copies=4,
                                            num_generations=10, 
                                            num_genes=3,
                                            mutation_probability=[0.5, 0])


# %%

subset_genes =['Zfhx3', 'Hoxc4', 'Tenm2','Pcdh9','Hoxb5','Meis2','Nrxn3','Ebf3','Tshz2',
               'Zfhx4', 'Nxph1', 'Dscaml1', 'Hoxb3', 'Pcdh7', 'Rbfox1', 'Pbx3', 'Pcp4',
               'Arpp21', 'Gabra1', 'Negr1', 'Nxph4', 'Pax2', 'Celf2', 'Syt1', 'Shox2',
               'Cdh13', 'Syt2', 'Nrg1', 'Opcml', 'Zfp536', 'Elavl2', 'Ralyl', 'Runx1t1',
               'Phox2b', 'Cadm2', 'Slc8a1','Pdzrn4','Meis1','Lsamp','Nrn1','Cacna2d3',
               'Bcl11b','Rbms3','Lrp1b','Hs6st3','Car10','Ntng1','Onecut2','Rph3a',
               'Kcnip4','Sorcs3','Tafa2','Dpp10','Kirrel3','Atp2b4','Npas3','Cdh8',
               'Ppp3ca','Alcam','Hcn1','Rmst','Lamp5','Slc6a5','A830018L16Rik','Robo1',
               'Lhx1','Bcl11a','Erbb4','Rab3c','Sparcl1','Grm7','Lhx1os','Pbx1',
               'Cplx1','Sema6d','Ptprg','Ahi1','Tmem163','Gad2','Gria3','Slc17a6',
               'Hs3st4','Aldoc','Cacna2d1','Snca','Slc32a1','Kcna2','Pax8','Chrm3',
               'Tcf4','Scn1a','Gm26871','Galnt13','Cntn4','6330411D24Rik','Ebf1','Pcdh17',
               'Mpped2', 'Otp','Grid2','Oip5os1','Auts2','Olfm1','Gm14204','Bend6',
               'Fndc5','Atp2b2','Vamp1','Etl4','Prkg1','Oprm1','Atp1b1','Meg3','Necab1',
               'Slc30a3','Gad1','Vsnl1','Pcdh11x','Scn8a']
subset_df = gene_df.loc[subset_genes]

# %%

meta = pd.read_csv(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-meta.csv"), low_memory=False)
exp = np.load(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-raw.npy"))
freqs = pd.read_pickle(os.path.join(params.local_data_dir, "antIRN-PARN-MERFISH-freqs.pkl"))

exp_super, meta_super = cell_funcs.boot_super(exp, meta, k=5)

exp_super = gene_funcs.normalize_counts_to_median(exp_super)

# %%

meta_super.reset_index(drop=True, inplace=True)
non_neuronal = ['252 DMX VII Tbx20 Chol', '261 HB Calcb Chol']

nn_mask = np.isin(meta_super["subclass"], non_neuronal)
nn_index = list(meta_super.index[nn_mask])
neuron_index = list(meta_super.index[~nn_mask])

neur_nn = np.repeat('null', len(meta_super))
neur_nn[neuron_index] = "neuron"
neur_nn[nn_index] = "nn"

meta_super.insert(len(meta_super.columns), 'neur_nn', neur_nn)

# %%
# bootstrap distribution from scRNAseq that matches MERFISH frequencies
exp_boot, meta_boot = cell_funcs.bootstrap_scRNAseq(meta_super, exp_super, freqs, 
                                                    n=10000)

non_neuronal = ['252 DMX VII Tbx20 Chol', '261 HB Calcb Chol']

nn_mask = np.isin(meta_boot["subclass"], non_neuronal)
nn_index = list(meta_boot.index[nn_mask])
neuron_index = list(meta_boot.index[~nn_mask])

# %%

neuron_exp = []
nn_exp = []
for j in tqdm(range(exp_boot.shape[1])):
    neuron_exp.append(np.mean(exp_boot[neuron_index,j]))
    nn_exp.append(np.mean(exp_boot[nn_index,j]))
    
data = {}
data["gene"] = ABC_utils.load_gene("scRNAseq")
data["neuron_exp"] = neuron_exp
data["nn_exp"] = nn_exp
data["ratio"] = np.array(neuron_exp) / np.array(nn_exp)

gene_df = pd.DataFrame(data=data)
gene_df.set_index('gene', inplace=True)

# %%

ABC_plot.plot_gene("Slc5a7", meta_boot, exp_boot, level='cluster')
