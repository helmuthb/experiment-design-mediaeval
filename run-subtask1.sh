#!/bin/sh

cd step1
python train.py --batch_size 5 --block_step 20 --patience 10 --dataset discogs --num_classes 315  
python train.py --batch_size 5 --block_step 20 --patience 10 --dataset lastfm --num_classes 327  
python train.py --batch_size 5 --block_step 20 --patience 10 --dataset allmusic --num_classes 766  
python train.py --batch_size 5 --block_step 20 --patience 10 --dataset tagtraum --num_classes 296  
cd ..