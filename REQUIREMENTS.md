General Requirements:
* Python3
* Keras
* Tensorflow


**If you are running experiments on GPU clusters, do not forget to check dependencies of the installed libraries with available cuda and gcc modules
## activate modules 
* module load cuda/8.0
* module load cudnn/5.1
* module load gcc/5.2.0
* module load git

## get and install conda
* wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
* chmod +x Miniconda2-latest-Linux-x86_64.sh
* ./Miniconda2-latest-Linux-x86_64.sh

## logout - login to activate conda (or source .bashrc)
* exit

## create and active conda virtual environment
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


