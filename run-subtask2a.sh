#!/bin/sh

cd step1
python feed.py --batch_size 5 --block_step 20 --dataset discogs --num_classes 315  
python feed.py --batch_size 5 --block_step 20 --dataset lastfm --num_classes 327  
python feed.py --batch_size 5 --block_step 20 --dataset allmusic --num_classes 766  
python feed.py --batch_size 5 --block_step 20 --dataset tagtraum --num_classes 296  
cd ..