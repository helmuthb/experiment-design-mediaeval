#!/bin/sh

cd subtasks
python train_subtask1.py --batch_size 32 --block_step 50000 --patience 4 --dataset discogs --num_classes 315  
python train_subtask1.py --batch_size 32 --block_step 50000 --patience 4 --dataset lastfm --num_classes 327  
python train_subtask1.py --batch_size 32 --block_step 50000 --patience 4 --dataset allmusic --num_classes 766  
python train_subtask1.py --batch_size 32 --block_step 50000 --patience 4 --dataset tagtraum --num_classes 296  
cd ..