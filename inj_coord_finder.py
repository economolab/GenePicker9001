# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 16:23:00 2025

@author: jpv88
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ABC_toolbox import ABC_utils, data_nav

# %%

vol = data_nav.load_image_volume()

parc_dict = ABC_utils.load_parcel_dict()

# %%

ROIs = list(parc_dict.values())
ROIs.sort()

ROIs = ['IRN', 'PARN']

inv_parc_dict = {v: k for k, v in parc_dict.items()}

ROIs_idx = [inv_parc_dict[el] for el in ROIs]

# %%

print('Finding ROI voxels...')
ROI_vol = np.logical_or(vol == ROIs_idx[0], vol == ROIs_idx[1])

print('Finding ROI voxel coordinates...')
coord = np.where(ROI_vol)

print('Collating ROI voxel coordinates...')
coordinates = [(x,y,z) for x,y,z in zip(coord[0], coord[1], coord[2])]

print('Unpacking coordinates...')
x, y, z = zip(*coordinates)

print('Repacking into dataframe...')
d = {'x': x, 'y': y, 'z': z}
df = pd.DataFrame(d)

# %%

def ccf_to_breg(x, y, z):
    
    x = x - 540
    y = y - 44
    z = z - 570
    
    AP = x * np.cos(0.0873) - y * np.sin(0.0873)
    DV = x * np.sin(0.0873) + y * np.cos(0.0873)
    ML = z
    
    DV = DV * 0.9434
    
    return AP, DV, ML

# %% restrict to anterior IRN/PARN

mask = df['x'].values <= 1100
df = df[mask]

# there's a hard cutoff (self-made) in posterior but not anterior, IRN/PARN appears to start in earnest around 1050
# this aligns visually with disconnection of VII in CCF 
plt.hist(df['x'].values, bins=40)

# A to P
x_cent = ((1110 - 1050) / 2) + 1050
x_range = 1110 - 1050

# D to V
y_cent = np.mean(df['y'].values)
y_range = max(df['y'].values) - min(df['y'].values)

AP, DV, ML = ccf_to_breg(x_cent, y_cent, 0)

# https://community.brain-map.org/t/how-to-transform-ccf-x-y-z-coordinates-into-stereotactic-coordinates/1858
# https://www.sciencedirect.com/science/article/pii/S0092867420304025

midline = 1140/2

left_vox = (df['z'].values < midline)
left_vox = df['z'].values[left_vox]
z_cent_left = np.mean(left_vox)
dist_left = abs(z_cent_left - midline)
ML = dist_left

AP = round(AP)
DV = round(DV)
ML = round(ML)

DV = 4729 # determined using ImageJ

# %%
# # %% 

# from PIL import Image

# img = Image.fromarray(vol[:,:,int(midline + ML)])
# rot_90 = img.rotate(-90, expand=True)
# rot_5 = rot_90.rotate(5)

# img.show()
# rot_90.show()
# rot_5.show()

# img = np.array(rot_5)

# x_cent_rot = x_cent * np.cos(0.0873) - y_cent * np.sin(0.0873)
# y_cent_rot = x_cent * np.sin(0.0873) + y_cent * np.cos(0.0873)


# # %%

# # dictionary for converting breg to CCF
# breg_coordinates = {}
# for index, row in tqdm(df.iterrows()):
#     AP, DV, ML = ccf_to_breg(row['x'], row['y'], row['z'])
#     AP = round(AP)
#     DV = round(DV)
#     ML = round(ML)
#     breg_coordinates[(AP, DV, ML)] = (row['x'], row['y'], row['z'])

# DV_line = img[:,1300-round(x_cent_rot)]

# fig, ax = plt.subplots()
# ax.imshow(img)
# ax.axvline(1300-round(x_cent_rot))


# # %%

# allen_coords = []
# for i in range(1000):
#     if (AP, i, ML) in breg_coordinates.keys():
#         allen_coords.append(breg_coordinates[(AP, i, ML)])
        
# parc_labels = []
# for i, j, k in allen_coords:
#     parc_labels.append(vol[i,j,k])


# # %% determine coords in allen ccf rotated to match stereotax

# # A to P
# x_cent = ((1110 - 1050) / 2) + 1050

# # D to V
# y_cent = np.mean(df['y'].values)

# # allen ccf to bregma
# AP = x_cent * np.cos(0.0873) - y_cent * np.sin(0.0873)
# DV = x_cent * np.sin(0.0873) + y_cent * np.cos(0.0873)

# DV = DV * 0.9434

