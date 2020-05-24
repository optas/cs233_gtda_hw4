## Requirements
Please look at setup.py 
The code has been tested with Python 3.6 (any 3x version should work) and pytorch 1.3 (any 1.1+ version should work).

## STANDALONE Installation Instructions:

Assuming you have already a copy of cs233_gtda_hw4 folder in your hard drive and you are inside that folder with your virtual environment (e.g. conda) active:

1. pip install -e _top-directory-path-of-cs233_gtda_hw4 folder_  # this will install cs233_gtda_hw4 as a python-package with all its default dependencies
2. git submodule update --init --recursive  # this will clone the JIT (fast) Chamfer implementation in cs233_gtda_hw4
3. Start your work at notebooks/main.ipynb  (or notebooks_as_python_scripts/main.py if you are not a fan of notebooks).

------
## Alternatively, inside your active (conda?) environment
Clone the repo from github (https://github.com/optas/cs233_gtda_hw4)
            
    1. git clone https://github.com/optas/cs233_gtda_hw4.git
    
    2. cd cs233_gtda_hw4
    
    3. git submodule add https://github.com/ThibaultGROUEIX/ChamferDistancePytorch cs233_gtda_hw4/losses/ChamferDistancePytorch
        
    4. pip install -e .  # to install the package as a environment-wide module
    
    5. Go to main.ipynb or main.py                

------

Potential Hiccup:

The fast(er) implementation of Chamfer (the one from the submodule above) requires the ninja build system installed.

If you do not have it you can install it like this:
1. wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
2. sudo unzip ninja-linux.zip -d /usr/local/bin/
3. sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force

If you cannot do it, you might have to resort to the ~10x slower provided implementation of Chamfer in losses/nn_distance/chamfer_loss
(see notes inside the models/pointcloud_autoencoder.py).

-----
### (optional) instructions for making a conda environment for this HW.
1. Install anaconda: https://docs.anaconda.com/anaconda/install/
 
2. Create an enviroment: conda create -n name_you_like python=3.6 cudatoolkit=10.0

3. conda activate name_you_like

-----
Best of luck!
The CS233 Instructor/TAs

