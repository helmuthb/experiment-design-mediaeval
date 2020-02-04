#! /bin/sh

cd data
for SOURCE in allmusic discogs lastfm tagtraum
do
    FILE=acousticbrainz-mediaeval-${SOURCE}-train.tsv
    if hash python2 2>/dev/null; then
        python2 ../participant_split_data.py -m 0 $FILE
    else
        python ../participant_split_data.py -m 0 $FILE
    fi
    
done
