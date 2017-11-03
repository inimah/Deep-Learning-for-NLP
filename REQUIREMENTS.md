General Requirements:
* Python3
* Keras
* Tensorflow

## Installation (For Linux)
## get and install python3 packages via miniconda3
* Link to miniconda site: https://conda.io/miniconda.html
* wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
* chmod +x Miniconda3-latest-Linux-x86_64.sh
* ./Miniconda3-latest-Linux-x86_64.sh

## create and activate conda virtual environment
* conda create -n tfenv
* source activate tfenv            

## Add conda-forge repository, install tensorflow library
* conda config --add channels conda-forge
* conda install tensorflow-gpu

## install required libraries
* conda install ipython
* conda install scipy
* conda install scikit-learn
* conda install h5py
* pip install wget
* conda install matplotlib
* pip install -U nltk
* python -m nltk.downloader -d /(your-home-directory)/nltk_data all
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


