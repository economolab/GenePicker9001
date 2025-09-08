# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 12:39:04 2025

@author: jpv88
"""

import random

genes = ['Phox2b', 'Slc17a6', 'Slc6a5', 'Slc5a7', 'Alcam', 'Nxph1', 'Hoxb5', 'Tenm2', 
         'Meis2', 'Pcp4', 'Syt1', 'Rph3a', 'Zfhx3', 'Tshz2', 'Ebf3', 'Pax2', 'Arpp21', 
         'Robo1', 'Negr1', 'Rab3c']

to_order = ['Alcam']

# order of genes by strength
to_place = random.choice(to_order)

# order: Slc5a7 > Slc6a5 > Slc17a6 > Rph3a > Syt1 > Rab3c > Robo1 > Hoxb5 > Pcp4 > Meis2 > Alcam > Nxph1 > Ebf3 > Negr1 > Arpp21 > Phox2b > Tshz2 > Pax2 > Zfhx3 > Tenm2

# order: Tshz2 > Pax2 > Zfhx3 > Tenm2
