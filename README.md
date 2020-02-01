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