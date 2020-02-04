#! /bin/sh

rm -rf processed 2>/dev/null
if hash python3 2>/dev/null; then
    python3 preprocessing.py
else
    python preprocessing.py
fi
