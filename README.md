## Requirements
Please look at setup.py 
The code has been tested with Python 3.6 (any 3x version should work).

## Installation Instructions:

#### If you have already an activated virtual-environment:
            
    1. git clone https://github.com/optas/cs233_gtda_hw4.git
    
    2. cd cs233_gtda_hw4
    
    3. git submodule add https://github.com/ThibaultGROUEIX/ChamferDistancePytorch cs233_gtda_hw4/losses/ChamferDistancePytorch
        
    4. pip install -e .  # to install the package as a environment-wide module
    
    5. Go to main.ipynb or main.py                

------
#### If not, please first make such an environment (see instructions below for Conda) 

1. Install anaconda: https://docs.anaconda.com/anaconda/install/
 
2. Create an enviroment: conda create -n name_you_like **python=3.6 cudatoolkit=10.1**

3. conda activate name_you_like

4. conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

(step 4 is optional, but should be done to further increase the chances that you 
get pytorch that sees the GPUs of your system)

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
Best of luck!
The CS233 Instructor/TAs

