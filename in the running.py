# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 12:19:42 2025

@author: jpv88
"""

import numpy as np
import pandas as pd

heur_2000 = np.load("top_genes_2000.npy", allow_pickle=True)

# %%

cortex = ['Syt6', 'Cpne9', 'Slc32a1', 'Snap25', 'Fam84b', 'Tshz2', 'Slc17a7', 
          'Pvalb', 'Vip', 'Sst', 'Otof', 'Lrg1', 'Hgf', 'Arhgap25', 'Npnt', 
          'Slco2a1', 'Nxph4', 'Tmem215', 'Stard8', 'Ccdc80', 'Rorb', 'Susd5', 'Cbln4',
          'Kcnh5', 'Ctgf', 'Calb1', 'Dkkl1', 'Hkdc1', 'Pamr1', 'Myl4', 'Snap25',
          'Pvalb', 'Sst', 'Rorb', 'Dkkl1', 'Npnt', 'Vip', 'Slc32a1', 'Penk', 'Aldh1l1',
          'Lamp5', 'Col6a1', 'Gad1', 'Gad2', 'Nrn1', 'Stk17b', 'Col8a1', 'Penk']
cortex = np.unique(cortex)

bs = ['Meis2', 'Adarb2', 'Penk', 'Tfap2b', 'Trhr', 'Baiap3', 'Zfhx3', 'Zeb2',
      'Ebf3', 'Cgnl1', 'Htr2c', 'Fxyd6', 'Gpc3', 'Ndnf', 'Nr2f2', 'Rbm20', 'Slc17a6',
      'Slc6a5', 'Nts', 'Sv2b', 'Spp1', 'Cnr1', 'Dlk1', 'Sncg', 'Kcng4', 'Lhx9',
      'Slc6a1', 'Npy', 'Tac1', 'Mog', 'Gpr37l1', 'Reln', 'Tac2', 'Cck', 'Slit2',
      'Rgs6', 'Calb2', 'Rnf220', 'Cartpt', 'Nfib', 'Pcp4', 'Chrm2', 'Erbb4', 'Phox2b',
      'Gal', 'Pde1a', 'Nrp2', 'Pdyn', 'Nnat', 'Adcyap1', 'Gabrq', 'Ecel1', 'Nxph1',
      'Syt10', 'Nmu', 'Ntng2', 'Chat', 'Arpp21', 'Camk2a', 'Lgr5', 'Hoxb5', 'Ebf3',
      'Shox2', 'Tlx1', 'Lbx1', 'Dbh', 'Dmbx1', 'Gbx2', 'Fat2', 'Pax2', 'Tlx3', 'Otp',
      'Ttr', 'Me3', 'Npas1', 'Zfhx4', 'Kirrel3', 'Syt17', 'Snap25', 'Spp1', 'Slc32a1',
      'Hoxc4', 'Hoxd3', 'Meis2', 'Hoxb4', 'Bcl11a', 'Onecut2', 'Nr4a2', 'Zfp536',
      'Hoxa5', 'Nfix', 'Tshz2', 'Tfap2b', 'Prox1', 'Dscaml1', 'Neurod2', 'Prrxl1',
      'Nkx6-1', 'Gata3', 'Tfap2b', 'Spp1', 'Slc17a6', 'Pax2', 'Hoxd3', 'Bcl11a', 'Hoxa5',
      'Neurod2', 'Phox2b', 'Chat', 'Gabra1', 'Rbfox1', 'Nrg1', 'Syt1', 'Cdh13', 'Hcn1',
      'Ralyl', 'Galnt13', 'Rab3c', 'Cdh8', 'Elavl2', 'Car10', 'Cacna2d3', 'Oprm1', 'Pbx3',
      'Rbms3', 'Pcdh7', 'Robo1', 'Dpp10', 'Chrm3', 'Meg3', 'Cntnap2', 'Atp1b1', 'Nap1l5',
      'Selplg', 'Tmem119', 'Aldoc', 'Slc2a5', 'Prph', 'Slc5a7', 'Celf2', 'Vamp1', 'Rph3a',
      'Syt2', 'Grik1', 'Alcam', 'Bend6', 'Ntng1', 'Tenm2', 'Negr1', 'Sdk1', 'Hs6st3']
bs = np.unique(bs)

# %%

isin = np.isin(cortex, heur_2000)
cortex_include = cortex[isin]

isin = np.isin(bs, heur_2000)
bs_include = bs[isin]

all_include = np.concatenate((cortex_include, bs_include))
all_include = np.unique(all_include)

remove = ["Snap25", "Atp1b1", "Phox2b", "Slc5a7", "Slc6a5", "Slc17a6"]

final = np.setdiff1d(all_include, remove)
np.save("top_genes_135.npy", final)

# %%

df_0 = pd.read_csv("final_gene_df_100.csv")
df_1 = pd.read_csv("final_gene_df_100_dupe1.csv")
df_2 = pd.read_csv("final_gene_df_100_dupe2.csv")
df_3 = pd.read_csv("final_gene_df_100_dupe3.csv")
df_4 = pd.read_csv("final_gene_df_100_dupe4.csv")

dfs = [df_0, df_1, df_2, df_3, df_4]

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
        
        for gene_df in gene_df_gas:
            
            if gene in gene_df['gene'].values:
                scores.append(gene_df['score'][gene_df['gene'] == gene].values[0])
                
        gene_scores.append(np.average(scores))
        
    return gene_scores

tested_genes = find_tested_genes(dfs)
gene_scores = score_genes(dfs, tested_genes)

final_gene_df = {}
final_gene_df['gene'] = tested_genes
final_gene_df['score'] = gene_scores
final_gene_df = pd.DataFrame(data=final_gene_df)
