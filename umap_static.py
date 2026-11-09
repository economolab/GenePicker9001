# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 18:16:00 2025

@author: jpv88
"""

import os

import params

import colorsys
import numpy as np
import pandas as pd
import scanpy as sc

from anndata import AnnData, concat, read_h5ad
from scipy.special import comb
from tqdm import tqdm

import matplotlib.pyplot as plt

from ABC_toolbox import ABC_utils, ABC_plot, cell_funcs, gene_funcs

# %%

meta = pd.read_csv(os.path.join(params.local_data_dir, "IRN-PARN-scRNAseq-meta.csv"), low_memory=False)
exp_raw = np.load(os.path.join(params.local_data_dir, "IRN-PARN-scRNAseq-raw.npy"))
ratios = pd.read_pickle(os.path.join(params.local_data_dir, "IRN-PARN-MERFISH-freqs.pkl"))

exp_norm = gene_funcs.normalize_counts_to_median(exp_raw)
genes = ABC_utils.load_gene("scRNAseq")

boot_mat, boot_meta = cell_funcs.bootstrap_scRNAseq(meta, exp_norm, ratios, n=20000)

# exp_super, meta_super = cell_funcs.boot_super(exp_norm, meta, k=3, boot_factor=1)

adata = AnnData(boot_mat)

# assign cell type
adata.obs["cluster"] = boot_meta["cluster"].values
adata.obs["supertype"] = boot_meta["supertype"].values
adata.obs["subclass"] = boot_meta["subclass"].values
adata.obs["class"] = boot_meta["class"].values

# assign gene names
adata.var_names = genes

#%%

adata = sc.read_h5ad('IRN-PARN-upsampled')

#%%

meta = adata.obs['cluster'].to_frame(name='cluster')
adata_ratios = cell_funcs.calc_frac_per_type(meta)

df = pd.DataFrame({"dict 1": ratios, "dict 2": adata_ratios})
df.plot.bar(rot=0, figsize=(8, 5), xlabel="Key", ylabel="Value")
plt.tight_layout()
plt.show()

#%%

bad_subclasses = ['255 SPVO Mafa Meis2 Glut',
                  '252 DMX VII Tbx20 Chol',
                  '261 HB Calcb Chol',
                  '235 PG-TRN-LRN Fat2 Glut',
              	  '228 PSV Pvalb Lhx2 Glut',
                  '266 PRNc Prox1 Brs3 Gly-Gaba',
              	  '260 MDRNv Crp Glut',
                  '306 SPVI-SPVC Sall3 Lhx1 Gly-Gaba',
                  '305 SPVI-SPVC Sall3 Nfib Gly-Gaba',
              	  '241 NTS Mbnl3 Glut',
                  '293 PAS-MV Ebf2 Gly-Gaba',
              	  '251 NTS Dbh Glut',
                  '256 SPVC Mafa Glut',
                  '216 MB-MY Tph2 Glut-Sero',
                  '296 RPA Pax6 Hoxb5 Gly-Gaba',
                  '244 MV-SPIV Slc6a2 Glut',
                  '246 CU-ECU-SPVI Foxb1 Glut',
                  '247 MV-SPIV Phox2b Ebf3 Lbx1 Glut',
                  '031 CT SUB Glut',
                  '301 MV Nr4a2 Gly-Gaba',
                  '264 PRNc Otp Gly-Gaba',
                  '254 VCO Mafa Meis2 Glut',
                  '225 PRNc-NI-SG-RPO Vsx2 Nr4a2 Glut',
                  '302 MV Xdh Gly-Gaba',
              	  '283 PRP Otp Gly-Gaba',
                  '299 MARN-PPY Ngfr Gly-Gaba',
                  '226 PRNc-PARN Tlx1 Glut',
                  '286 PPY-PGRNl Vip Glyc-Gaba']

good_subclasses = ['282 POR Spp1 Gly-Gaba',
                   '308 DCO Il22 Gly-Gaba',
                   '229 PB-NTS Phox2b Ebf3 Lmx1b Glut',
                   '284 GRN-IRN-MDRNd Ikzf1 Gly-Gaba',
                   '290 MY Prox1 Lmo7 Gly-Gaba',
                   '278 NLL Gata3 Gly-Gaba',
                   '288 MDRN Hoxb5 Ebf2 Gly-Gaba',
                   '289 MDRNd Prox1 Pax6 Gly-Gaba',
                   '291 NTS-MDRNd Prox1 Zic1 Gly-Gaba',
                   '242 PGRNd Dmbx1 Glut',
                   '245 SPVI-SPVC Tlx3 Ebf3 Glut',
                   '236 IRN Vip Glut',
                   '303 IRN Dmbx1 Pax2 Gly-Gaba',
                   '304 NTS-PARN Neurod2 Gly-Gaba',
                   '300 PARN-MDRNd-NTS Gbx2 Gly-Gaba']

remove_bool_bad = np.isin(adata.obs['subclass'], bad_subclasses)
remove_bool_good = np.isin(adata.obs['subclass'], good_subclasses)
remove_bool = np.logical_or(remove_bool_bad, remove_bool_good)

adata = adata[~remove_bool, :].copy()

#%%

level = 'subclass'
meta = adata.obs[level].to_frame(name=level)
adata_ratios = cell_funcs.calc_frac_per_type(meta, level=level)

# %% revert to size normalized counts

print('Normalizing counts...')
sc.pp.normalize_total(adata)

print('Identifying highly variable genes...')
sc.pp.highly_variable_genes(
    adata,
    flavor="seurat_v3",
    n_top_genes=2000
)

print('Basic preprocessing...')
sc.pp.log1p(adata)
sc.pp.scale(adata)

print('Performing PCA...')
sc.pp.pca(adata)

#%%

n_neighbors = 15 # default 15 (generally accepted range: 5 - 50)
min_dist = 0.5 # default 0.5 (generally accepted range 0.0 - 0.5)

print('Computing nearest neighbors...')
sc.pp.neighbors(adata, n_neighbors=n_neighbors)

print('Making UMAP...')
sc.tl.umap(adata, min_dist=min_dist)

#%%

X_umap = adata.obsm['X_umap']

level = "subclass"
colors_dict = ABC_plot.fetch_colors_dict(level)
colors = [colors_dict[el] for el in adata.obs[level]]

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("UMAP 1", fontsize=24)
ax.set_ylabel("UMAP 2", fontsize=24)
ax.spines[['right', 'top']].set_visible(False)
fig.tight_layout()

scat = ax.scatter(X_umap[:,0], X_umap[:,1], s=12, c=colors)

#%%

xlim = [6.05, 8.371]
ylim = [-8.1, -2.947]

in_x = np.logical_and(adata.obsm['X_umap'][:,0] >= xlim[0], adata.obsm['X_umap'][:,0] <= xlim[1])
in_y = np.logical_and(adata.obsm['X_umap'][:,1] >= ylim[0], adata.obsm['X_umap'][:,1] <= ylim[1])

in_box = np.logical_and(in_x, in_y)

box_cells = adata.obs['subclass'][in_box]

# %%

# make boolean array of genes to be included in Umap
# mask_var = ['Phox2b', 'Slc6a5', 'Slc17a6', 'Slc5a7',
#            'Hoxb5', 'Zfhx3', 'Ebf3', 'Gabra1',
#            'Meis2', 'Zfhx4', 'Arpp21', 'Zfhx3']
mask_var = ['Phox2b', 'Slc6a5', 'Slc17a6', 'Slc5a7',
            'Hoxb5', 'Ebf3', 'Pbx3', 'Nxph1',
            'Arpp21', 'Zfhx4', 'Meis2', 'Rbms3',
            'Rbfox1', 'Grin3a', 'Bend6', 'Syt2',
            'Tshz2', 'Tenm2', 'Kcna2', 'Pax2',
            'Cntnap2', 'Pcdh7', 'Sdk1', 'Vamp1']
all_genes = list(adata_us.var_names)
mask_var = [gene in mask_var for gene in all_genes]
mask_var = np.array(mask_var)

sc.pp.pca(adata_us, mask_var=mask_var)
sc.pp.neighbors(adata_us)
sc.tl.umap(adata_us, key_added='X_umap_24gene')

X_umap = adata_us.obsm["X_umap_24gene"]
X_umap_24gene = adata_us_24.obsm["X_umap_new"]

# %%

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_umap = scaler.fit_transform(X_umap)
X_umap_24gene = scaler.fit_transform(X_umap_24gene)

# %%

X_umap = adata_us.obsm["X_umap"]
X_umap_new = adata_us_center.obsm["X_umap_new"]

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_umap = scaler.fit_transform(X_umap)
X_umap_new = scaler.fit_transform(X_umap_new)

# X_umap = X_umap[~np.array(outlier_mask)]

# %%

X_umap = adata_us.obsm["X_umap"]
X_umap_new = adata_us_center.obsm["X_umap_new"]

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_umap = scaler.fit_transform(X_umap)
X_umap_new = scaler.fit_transform(X_umap_new)

X_umap_old = X_umap.copy()
X_umap_old[~np.array(outlier_mask)] = X_umap_new

# %%

exp_super, meta_super = cell_funcs.boot_super(adata_us.X, 
                                              pd.DataFrame(adata_us.obs['cluster']),
                                              k=3,
                                              boot_factor=1)

super_X = np.zeros(adata_us.X.shape)
for clu in np.unique(adata_us.obs['cluster']):
    
    # where this clu is in the upsampled AnnData
    adata_mask = (adata_us.obs['cluster'] == clu)
    adata_mask = adata_mask.values
    
    # where this clu is in the supercells
    super_mask = (meta_super['cluster'] == clu)
    super_mask = super_mask.values

    super_X[adata_mask,:] = exp_super[super_mask,:]  
    
adata_super = AnnData(super_X)

genes = ABC_utils.load_gene("scRNAseq")
adata_super.obs["cluster"] = adata_us.obs["cluster"].values
adata_super.var_names = genes

# make umap from anndata object without overwriting counts info
def make_umap(adata, n_top_genes=2000):
    
    sc.pp.highly_variable_genes(
        adata_us,
        flavor="seurat_v3",
        n_top_genes=n_top_genes
    )
    
    adata_copy = sc.pp.log1p(adata, copy=True)
    sc.pp.scale(adata_copy)
    sc.pp.pca(adata_copy)
    sc.pp.neighbors(adata_copy)
    sc.tl.umap(adata_copy)
    
    adata.obsm['X_umap'] = adata_copy.obsm['X_umap']
    
    return adata

adata_super = make_umap(adata_super)

# %%

start_umap = adata_us.obsm["X_umap"]
end_umap = adata_super.obsm["X_umap"]

from sklearn.preprocessing import MinMaxScaler

def scale_umaps(start_umap, end_umap):
    scaler = MinMaxScaler()

    start_umap = scaler.fit_transform(start_umap)
    end_umap = scaler.fit_transform(end_umap)
    
    return start_umap, end_umap

start_umap, end_umap = scale_umaps(start_umap, end_umap)
    

# %%

fps = 25
duration = 8
num_frames = fps * duration
interval = (1 / fps) * 1000

start_frac = 0.01
end_frac = 0.2

x_courses = []
y_courses = []

# start_umap = X_umap
# end_umap = X_umap

# start_umap = X_umap_24gene
# end_umap = X_umap

for i in tqdm(range(start_umap.shape[0])):
    
    pre_coords = start_umap[i,:]
    post_coords = end_umap[i,:]

    x_course, y_course = calc_point_course(pre_coords, post_coords, num_points=num_frames,
                                           start_frac=start_frac,
                                           end_frac=end_frac)
    
    x_courses.append(x_course)
    y_courses.append(y_course)
    
def build_umap_frame(x_courses, y_courses, frame):
    
    x_umap = [ar[frame] for ar in x_courses]
    y_umap = [ar[frame] for ar in y_courses]
    
    return x_umap, y_umap

level = "cluster"
colors_dict = ABC_plot.fetch_colors_dict(level)
pre_colors = [colors_dict[el] for el in adata_us.obs[level]]

level = "cluster"
colors_dict = ABC_plot.fetch_colors_dict(level)
post_colors = [colors_dict[el] for el in adata_us.obs[level]]

r_courses = []
g_courses = []
b_courses = []

for i in tqdm(range(start_umap.shape[0])):
    
    pre_color = pre_colors[i]
    post_color = post_colors[i]

    r_course, g_course, b_course = calc_color_course(pre_color, post_color, num_points=num_frames,
                                                     start_frac=0.2,
                                                     end_frac=end_frac)
    
    r_courses.append(r_course)
    g_courses.append(g_course)
    b_courses.append(b_course)
    
def build_umap_color(r_courses, g_courses, b_courses, frame):
    
    r_umap = [ar[frame] for ar in r_courses]
    g_umap = [ar[frame] for ar in g_courses]
    b_umap = [ar[frame] for ar in b_courses]
    
    umap_color = list(zip(r_umap, g_umap, b_umap))
    
    return umap_color

pre_alphas = np.repeat(1, start_umap.shape[0])
post_alphas = pre_alphas.copy()
pre_alphas[outlier_mask] = 0

alpha_courses = []

for i in tqdm(range(start_umap.shape[0])):
    
    pre_alpha = pre_alphas[i]
    post_alpha = post_alphas[i]

    alpha_course = calc_alpha_course(pre_alpha, post_alpha, num_points=num_frames,
                                     start_frac=0.2,
                                     end_frac=end_frac)
    
    alpha_courses.append(alpha_course)

def build_umap_alpha(alpha_courses, frame):
    
    alpha_umap = [ar[frame] for ar in alpha_courses]
    alpha_umap = np.array(alpha_umap)
    
    return alpha_umap

# %% debugging

# mask = adata_us.obs["subclass"] == "303 IRN Dmbx1 Pax2 Gly-Gaba"
# mask = mask.values

# fig = plt.figure(figsize=(16, 9))
# ax = fig.add_subplot(111)
# ax.set_xlim(-0.1, 1.1), ax.set_xticks([])
# ax.set_ylim(-0.1, 1.1), ax.set_yticks([])
# ax.scatter(X_umap[mask,0], X_umap[mask,1])

# idx = np.arange(len(alpha_courses))[mask]
# test = alpha_courses[idx[0]]

# %% 

change_colors = False
change_position = True
change_alphas = False

import matplotlib.animation as animation

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111)
ax.set_xlim(-0.1, 1.1), ax.set_xticks([])
ax.set_ylim(-0.1, 1.1), ax.set_yticks([])
ax.set_xlabel("UMAP 1", fontsize=24)
ax.set_ylabel("UMAP 2", fontsize=24)
ax.spines[['right', 'top']].set_visible(False)
fig.tight_layout()

level = "cluster"
colors_dict = ABC_plot.fetch_colors_dict(level)
colors = [colors_dict[el] for el in adata_us.obs[level]]

scat = ax.scatter(start_umap[:,0], start_umap[:,1], s=12, c=colors)

plt.savefig('pre_img', transparent=True)

def ani_func(frame):
    
    if change_position == True:
        x, y = build_umap_frame(x_courses, y_courses, frame)
        scat.set_offsets(np.column_stack((x, y)))
    
    if change_colors == True:
        umap_color = build_umap_color(r_courses, g_courses, b_courses, frame)
        scat.set_color(umap_color)
        
    if change_alphas  == True:
        alpha_umap = build_umap_alpha(alpha_courses, frame)
        scat.set_alpha(alpha_umap)
    
final_frame = num_frames - 1
ani_func(final_frame)

plt.savefig('post_img', transparent=True)

# %%
    
ani = animation.FuncAnimation(fig=fig, 
                              func=ani_func, 
                              frames=num_frames, 
                              interval=interval,
                              repeat=False)


# %%

plt.rcParams['animation.convert_path'] = 'C:/Program Files/ImageMagick-7.1.1-Q16-HDRI/magick.exe'

# writergif = animation.PillowWriter(fps=fps)
# FFwriter = animation.FFMpegWriter(fps=fps)
writerMagick = animation.ImageMagickWriter(fps=fps)
ani.save("ani_test.gif", writer=writerMagick)

# %% cell type pies

import pandas as pd

# filt = True

ratios = pd.read_pickle(os.path.join(params.local_data_dir, "antIRN-PARN-MERFISH-freqs.pkl"))

ratios = {k: v for k, v in sorted(ratios.items(), key=lambda item: item[1])}

cluster_color_dict = ABC_plot.fetch_colors_dict("cluster")
colors = [cluster_color_dict[x] for x in ratios.keys()]

    
x = np.array(list(ratios.values()))


fig, ax = plt.subplots()
ax.pie(x, colors=colors, wedgeprops=dict(width=0.25))

# %%

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


# %%

classes = [str(ABC_utils.fetch_cluster_tax(x)[2]) for x in ratios.keys()]
classes = np.unique(classes)

class_ratios = renorm_ratios(ratios, classes)
class_ratios = {k: v for k, v in sorted(class_ratios.items(), key=lambda item: item[1])}
class_color_dict = ABC_plot.fetch_colors_dict("class")
colors = [class_color_dict[x] for x in class_ratios.keys()]
    
x = np.array(list(class_ratios.values()))

fig, ax = plt.subplots()
ax.pie(x, colors=colors, wedgeprops=dict(width=0.25))

# %%

subclasses = [str(ABC_utils.fetch_cluster_tax(x)[1]) for x in ratios.keys()]
subclasses = np.unique(subclasses)

sub_ratios = renorm_ratios(ratios, subclasses)
sub_ratios = {k: v for k, v in sorted(sub_ratios.items(), key=lambda item: item[1])}
sub_color_dict = ABC_plot.fetch_colors_dict("subclass")
colors = [sub_color_dict[x] for x in sub_ratios.keys()]
    
x = np.array(list(sub_ratios.values()))

fig, ax = plt.subplots()
ax.pie(x, colors=colors, wedgeprops=dict(width=0.25))

# %%

sups = [str(ABC_utils.fetch_cluster_tax(x)[0]) for x in ratios.keys()]
sups = np.unique(sups)

sup_ratios = renorm_ratios(ratios, sups)
sup_ratios = {k: v for k, v in sorted(sup_ratios.items(), key=lambda item: item[1])}
sup_color_dict = ABC_plot.fetch_colors_dict("supertype")
colors = [sup_color_dict[x] for x in sup_ratios.keys()]

x = np.array(list(sup_ratios.values()))

fig, ax = plt.subplots()
ax.pie(x, colors=colors, wedgeprops=dict(width=0.25))
