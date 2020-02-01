#! /bin/sh

cd data
for SOURCE in allmusic discogs lastfm tagtraum
do
    FILE=acousticbrainz-mediaeval-${SOURCE}-train.tsv
    python2 ../participant_split_data.py -m 0 $FILE
done
