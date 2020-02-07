#!/bin/sh

cd subtasks
python prepare_subtask2.py --batch_size 32 --block_step 50000 --dataset discogs --num_classes 315  
python prepare_subtask2.py --batch_size 32 --block_step 50000 --dataset lastfm --num_classes 327  
python prepare_subtask2.py --batch_size 32 --block_step 50000 --dataset allmusic --num_classes 766  
python prepare_subtask2.py --batch_size 32 --block_step 50000 --dataset tagtraum --num_classes 296  
cd ..