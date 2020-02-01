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