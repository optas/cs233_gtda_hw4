#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os.path as osp
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors  # Students: you can use this implementation to find the 
                                                # Nearest-Neigbors


# In[2]:


# Students: Default location of saved latent codes per last cell of main.ipynb, change appropriately if
# you saved them in another way.
vanilla_ae_emb_file = '../data/out/pc_ae_latent_codes.npz'
part_ae_emb_file = '../data/out/part_pc_ae_latent_codes.npz'


# In[3]:


# Load golden distances (pairwise matrix, or corresponding model/part names in golden_names)
golden_part_dist_file = '../data/golden_dists.npz'
golden_data = np.load(golden_part_dist_file, allow_pickle=True)
golden_part_dist = golden_data['golden_part_dist']
golden_names = golden_data['golden_names']


# In[4]:


# To load vanilla-AE-embeddings (if False will open those of the 2-branch AE).
vanilla = True # or False


# In[5]:


# Load/organize golden part-aware distances.
sn_id_to_parts = defaultdict(list)
id_to_part_loc = dict()

for i, name in enumerate(golden_names):
    # Extract shape-net model ids of golden, map them to their parts.
    sn_id, _, part_id, _, _ = name.split('_')
    sn_id_to_parts[sn_id].append(part_id)
    
    # Map shape-net model id and part_id to location in distance matrix, (the order is the same).
    id_to_part_loc[(sn_id, part_id)] = i


# In[6]:


if vanilla:
    in_d = np.load(vanilla_ae_emb_file)    # Students: assuming you used the numpy.savez        
else:
    in_d = np.load(part_ae_emb_file)
        
latent_codes = in_d['latent_codes']
test_names = in_d['test_names']


# In[10]:


# Use golden distances and matchings to solve question (g)

for i, sn_name in enumerate(test_names):
    parts_of_model = set(sn_id_to_parts[sn_name])
    matched_neighbor = None # Students find the model's name of the Nearest-Neighbor
    parts_of_neighbor = set(sn_id_to_parts[matched_neighbor])
    
    # compute the requested distances.
    # Use id_to_part_loc for each model/part combination

