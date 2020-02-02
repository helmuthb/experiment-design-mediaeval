# experiment-design-mediaeval-ws2019
Paper2 of Experiment Design Exercise2

# Data

Go to Data section of Mediaeval site, then to Download section. Click Google Drive link.

# Links

[2018-AcousticBrainz-Genre-Task](https://multimediaeval.github.io/2018-AcousticBrainz-Genre-Task/)

[Implementation of paper](https://github.com/MTG/acousticbrainz-mediaeval-baselines)


# Collaborators

(Sergio Oramas)[https://github.com/sergiooramas]

[https://github.com/dbogdanov]

[https://github.com/alastair]

# Step by Step

```
git clone https://github.com/websta/experiment-design-mediaeval-ws2019
```

get data
```
git clone https://github.com/multimediaeval/2017-AcousticBrainz-Genre-Task
cp 2017-AcousticBrainz-Genre-Task/data_stats/* experiment-design-mediaeval-ws2019/data/
```

create environments
```
conda create -n exp-preprocess-python3 python=3.7
conda activate exp-preprocess-python3
conda install h5py
conda install pandas

conda create -n exp-preprocess-python2 python=2.7
```

execute scripts
```
cd experiment-design-mediaeval-ws2019
conda activate exp-preprocess-python2
./split-train-test.sh
conda activate exp-preprocess-python3
./run-preprocess.sh
conda activate exp-preprocess-python2
./run-create-h5.sh
cd ..
```

```
git clone https://github.com/sergiooramas/tartarus
cd tartarus
git checkout b214f66dd4e61e83edc45ffc5c280efe7318a1b6
conda create -n exp-train-python2 python=2.7
conda activate exp-train-python2
pip install -r requirements.txt
cd ..
```

in `src/common.py` change
```
DATA_DIR = "/Users/Sergio/webserver/tartarus/dummy-data"
```

to your data folder, e.g.
```
DATA_DIR = "../../experiment-design-mediaeval-ws2019/data-genres"
```

run experiment
```
cd src
conda activate exp-train-python2
python run_experiments.py genres_allmusic
```

# GPU

not possible in windows 10 neither python 2.7 nor 3.7, pip or conda,..., e.g.:
```
conda install theano pygpu keras scikit-learn pandas
$ python run_experiments.py genres_allmusic
Using Theano backend.
ERROR (theano.gpuarray): Could not initialize pygpu, support disabled
Traceback (most recent call last):
  File "C:\Users\Laszlo\Anaconda3\envs\pygpu_python3\lib\site-packages\theano\gpuarray\__init__.py", line 227, in <module>
    use(config.device)
  File "C:\Users\Laszlo\Anaconda3\envs\pygpu_python3\lib\site-packages\theano\gpuarray\__init__.py", line 214, in use
    init_dev(device, preallocate=preallocate)
  File "C:\Users\Laszlo\Anaconda3\envs\pygpu_python3\lib\site-packages\theano\gpuarray\__init__.py", line 99, in init_dev
    **args)
  File "pygpu\gpuarray.pyx", line 658, in pygpu.gpuarray.init
  File "pygpu\gpuarray.pyx", line 587, in pygpu.gpuarray.pygpu_init
pygpu.gpuarray.GpuArrayException: b'Could not load "nvrtc64\_70.dll": The specified module could not be found.\r\n'
Traceback (most recent call last):
  File "run_experiments.py", line 1, in <module>
    from train import process
  File "M:\projects\tartarus_python3\src\train.py", line 20, in <module>
    import models
  File "M:\projects\tartarus_python3\src\models.py", line 1, in <module>
    from keras.layers import Dense, Dropout, Activation, Flatten, Permute, Lambda, Input, merge, BatchNormalization, Embedding, LSTM, Bidirectional, Reshape, GRU, Merge, ELU
ImportError: cannot import name 'Merge' from 'keras.layers' (C:\Users\Laszlo\Anaconda3\envs\pygpu_python3\lib\site-packages\keras\layers\__init__.py)
(pygpu_python3) 
```
2 errors, 2 fixes:

-> installed latest 10.2 cuda and copied to nvrtc64\_70.dll (https://github.com/Theano/Theano/issues/6681)

-> revert keras latest (2.2.x) to 2.1.5

# python2

```
conda create -n wegwerf python=2.7
pip install keras==2.1.5 pandas theano scikit-learn matplotlib
conda install -c conda-forge pygpu
pip install joblib
pip install h5py
```

# python3

```
conda create -n theano_gpu python=3.6
conda install keras=2.1.5 pandas theano scikit-learn matplotlib
```

python=3.7 was very bad, conda exploded while calculation the dependencies

# Learnings

- python 3 for preprocessing vs python 2 for training
- theano backend: tensorflow vs keras: different orderings of dimensions in nn architecture!
- steps:
  - preprocessing.py python 3
  - participant_split_data.py python 2 or 3 with h5py
  - create_h5.py
    - https://github.com/h5py/h5py/issues/1131
  - run_experiment python 3
    - ValueError: Error when checking target: expected dense_5 to have shape (766,) but got array with shape (1,)