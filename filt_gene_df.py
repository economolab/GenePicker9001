# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 12:42:40 2025

@author: jpv88
"""

import os

import numpy as np
import pandas as pd

gene_df_f = "final_gene_df_100_antIRN_PARN.csv"
n_filt = 100

# %%

gene_df = pd.read_csv(os.path.join(os.getcwd(), gene_df_f))

gene_df_filt = gene_df.nlargest(n_filt, "score")
top_genes = gene_df_filt["gene"].values

remove = ["Alcam", "Dscaml1", "Zfp536", "Bcl11a", "Hoxa5", "Hoxc4", "Nxph1",
          "Pcdh7", "Cntnap2", "Hoxb5", "Kirrel3", "Nr2f2", "Rbfox1", "Bend6",
          "Erbb4", "Syt2", "Tenm2", "Cnr1", "Meis2", "Vamp1", "Aldoc", "Baiap3",
          "Pcp4", "Meg3", "Syt1", "Rgs6", "Rph3a", "Zfhx3", "Tshz2", "Ebf3", "Otp",
          "Chrm2", "Pax2", "Arpp21", "Celf2", "Ralyl", "Robo1", "Negr1"]

final = np.setdiff1d(top_genes, remove)

np.save(f"top_genes_{n_filt}.npy", final)