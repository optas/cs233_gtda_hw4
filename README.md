## Requirements
Please look at setup.py 
The code has been tested with Python 3.6 (any 3x version should work).

## STANDALONE Installation Instructions:

Assuming you have already a copy of cs233_gtda_hw4 in your hard drive and you are inside a virtual environemnt (e.g. Conda):

1. pip install -e _top-directory-path of cs233_gtda_hw4 folder_  # this will install cs233_gtda_hw4 as a python-package with all its default dependencies
2. git submodule update --init --recursive  # this will clone the JIT (fast) Chamfer implementation in cs233_gtda_hw4
3. Start your work at notebooks/main.ipynb  (or notebooks_as_python_scripts/main.py if you are not a fan of notebooks).

------
## Alternatively, 
if you clone the repo from github (https://github.com/optas/cs233_gtda_hw4)
    do it in a way that will also bring in directly the Chamfer dependency (submodule) i.e.,:
        ```
        git clone --recursive https://github.com/optas/cs233_gtda_hw4.git
        ```
    then go to step3 as above

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

