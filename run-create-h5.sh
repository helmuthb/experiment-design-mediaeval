#! /bin/sh

rm -rf data-genres/patches/*.hdf5
rm -rf data-genres/splits/*.tsv
if hash python2 2>/dev/null; then
    python2 create_h5.py
else
    python create_h5.py
fi
