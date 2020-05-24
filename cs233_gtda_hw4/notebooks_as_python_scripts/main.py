#!/usr/bin/env python
# coding: utf-8

# In[9]:


import torch
import numpy as np
import os.path as osp
import tqdm
import matplotlib.pylab as plt
from torch import nn
from torch import optim
from collections import defaultdict

## Imports based on our ready-to-use code (after you pip-install the cs233_gtda_hw4 package)
from cs233_gtda_hw4.in_out.utils import make_data_loaders
from cs233_gtda_hw4.in_out.utils import save_state_dicts, load_state_dicts
from cs233_gtda_hw4.in_out import pointcloud_dataset
from cs233_gtda_hw4.in_out.plotting import plot_3d_point_cloud


## Imports you might use if you follow are scaffold code (it is OK to use your own stucture of the models)
from cs233_gtda_hw4.models import PointcloudAutoencoder
from cs233_gtda_hw4.models import PartAwarePointcloudAutoencoder
from cs233_gtda_hw4.models.point_net import PointNet
from cs233_gtda_hw4.models.mlp import MLP

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[12]:


##
## Fixed Settings (we do not expect you to change these)
## 

n_points = 1024  # number of points of each point-cloud
n_parts = 4      # max number of parts of each shape
n_train_epochs = 400

# Students: feel free to change below -ONLY- for the bonus Question:
# I.e., use THESE hyper-parameters when you train for the non-bonus questions.

part_lambda = 0.005  # for the part-aware AE you will be using (summing) two losses:
                     # chamfer + cross-entropy
                     # do it like this: chamfer + (part_lambda * cross-entropy), 
                     # i.e. scale-down the cross-entropy
init_lr = 0.009  # initial learning-rate, tested by us with ADAM optimizer (see below)


# In[3]:


## Students: feel free to change below:

# batch-size of data loaders
batch_size = 128 # if you can keep this too as is keep it, 
                 # but if it is too big for your GPU, feel free to change it.

# which device to use: cpu or cuda?
device = 'cuda'  # Note: only the "alternative" (slower) chamfer_loss in losses/nn_distance can run in cpu.

top_in_dir = '../data/'
top_out_dir = '../data/out/'


# In[5]:


# PREPARE DATA:

loaders = make_data_loaders(top_in_dir, batch_size)

for split, loader in loaders.items():
    print('N-examples', split, len(loader.dataset))
    
### Student on your own:

# encoder = PointNet...
# decoder = MLP...
# part_classifier = ...


# In[6]:


part_aware_model = True # or False

if part_aware_model:
    xentropy = nn.CrossEntropyLoss()
#     model = PartAwarePointcloudAutoencoder().to(device) # Students Work here
    model_tag = 'part_pc_ae'
else:
#     model = PointcloudAutoencoder().to(device)  # Students Work here
    model_tag = 'pc_ae'


# In[7]:


# optimizer = optim.Adam(model.parameters(), lr=init_lr)  # Students uncomment once you have defined your model


# In[30]:


## Train for multiple epochs your model.
# Students: the below for-loops are optional, feel free to structure your training 
# differently.

min_val_loss = np.Inf
out_file = osp.join(top_out_dir, model_tag + 'best_model.pth')
start_epoch = 1

for epoch in tqdm.tqdm(range(start_epoch, start_epoch + n_train_epochs)):
    for phase in ['train', 'val', 'test']:
        
        # Students Work Here.
        # epoch_losses = model.train_for_one_epoch  ... 

##       Save model if validation loss improved.
#         if phase == 'val' and recon_loss < min_val_loss:
#             min_val_loss = recon_loss
            
##        If you save the model like this, you can use the next peace to load it. 
#             save_state_dicts(out_file, epoch=epoch, model=model) 


# In[13]:


# Load model with best per-validation loss (uncomment when ready)
# best_epoch = load_state_dicts(out_file, model=model)
# print('per-validation optimal epoch', best_epoch)


# In[8]:


# MAKE your plots and analysis

# 5 examples to visualize per questions (e, f)
examples_to_visualize = ['8a67fd47001e52414c350d7ea5fe2a3a',
                         '1e0580f443a9e6d2593ebeeedbff73b',
                         'd3562f992aa405b214b1fd95dbca05',
                         '4e8d8792a3a6390b36b0f2a1430e993a',
                         '58479a7b7c157865e68f66efebc71317']

# You can (also) use the function for the reconstructions or the part-predictions 
# (for the latter check the kwargs parameter 'c' of matplotlib.
    # plot_3d_point_cloud    


# In[20]:


# Last, save the latent codes of the test data and go to the 
# measuring_part_awareness and tsne_plot_with_latent_codes code.

model.eval()   # Do not forget this.! We are not training any more (OK, since we do not 
               # have batch-norm, drop-out etc. this is not so important, however it is good standard 
               # practice)
latent_codes = []

# Extract the latent codes and save them, so you can analyze them later.

np.savez(osp.join(top_out_dir, model_tag +'_latent_codes'), 
         latent_codes=latent_codes, 
         test_names=test_names)

