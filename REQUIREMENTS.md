General Requirements:
* Python 3.5.4 (installed via miniconda3)
* Keras
* Tensorflow 1.3.0-py35_0 (installed via miniconda3)

## Installation (For Linux) - and applicable for user of grid cluster environment 
(in grid cluster environment, packages will be installed under your user account / $HOME directory)

## Install python3 packages via miniconda3
* Link to miniconda site: https://conda.io/miniconda.html
* wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
* chmod +x Miniconda3-latest-Linux-x86_64.sh
* ./Miniconda3-latest-Linux-x86_64.sh

## Create and activate conda virtual environment
* conda create -n tfenv
* source activate tfenv            

## Add conda-forge repository, install tensorflow library
* conda config --add channels conda-forge

If you have GPU card in your computer:

* conda install tensorflow-gpu

Otherwise:

* conda install tensorflow

## install required libraries
* conda install ipython
* conda install scipy
* conda install scikit-learn
* conda install h5py
* pip install wget
* conda install matplotlib
* pip install -U nltk

run the following code to download all nltk_data in your $HOME directory
* python -m nltk.downloader -d (your $HOME directory)/nltk_data all
* conda install gensim

## download and install keras
* cd (your-git-directory)
* git clone https://github.com/fchollet/keras.git
* cd keras
* python setup.py install 
pip install git+git://github.com/fchollet/keras.git --upgrade

**If you are running experiments on GPU clusters, do not forget to check dependencies of the installed libraries with available cuda and gcc modules
## activate modules 
* module load cuda/8.0
* module load cudnn/5.1
* module load gcc/5.2.0
* module load git

## Test Installation
python

```
import tensorflow as tf

graph = tf.constant('Hello world')
session = tf.Session()
print(session.run(graph))
session.close()
```

## deactivate/quit conda tensorflow environment
source deactivate tfenv


