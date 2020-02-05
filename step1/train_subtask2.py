import click
import h5py
import numpy as np
from pytorchtools import EarlyStopping
import random
from sklearn.metrics import roc_auc_score
import torch
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import common 
from models import Subtask2Model

def batch_block_generator(dataset, block_step, batch_size, y_path, N_train, id2gt):
    hdf5_file = f"{common.PATCHES_DIR}/patches_train_{dataset}_1x1.hdf5"
    f = h5py.File(hdf5_file,"r")
    # block_step = 50000
    block_step = block_step
    batch_size = batch_size
    randomize = True
    # while 1:
    for i in range(0, N_train, block_step):
        x_block = f['features'][i:min(N_train, i+block_step)]
        index_block = f['index'][i:min(N_train, i+block_step)]
        #y_block = f['targets'][i:min(N_train,i+block_step)]
        x_block = np.delete(x_block, np.where(index_block == ""), axis=0)
        index_block = np.delete(index_block, np.where(index_block == ""))
        y_block = np.asarray([id2gt[id.decode('utf-8')] for id in index_block])
        items_list = list(range(x_block.shape[0]))
        if randomize:
            random.shuffle(items_list)
        for j in range(0, len(items_list), batch_size):
            if j+batch_size <= x_block.shape[0]:
                items_in_batch = items_list[j:j+batch_size]
                x_batch = x_block[items_in_batch]
                y_batch = y_block[items_in_batch]
                yield (x_batch, y_batch)

@click.command()
@click.option("--epochs", default=100, help="Number of epochs.")
@click.option("--block_step", default=1, help="Block Step.")
@click.option("--batch_size", default=1, help="Batch Size.")
@click.option("--seed", default=73, help="Seed.")
@click.option("--dataset", default="discogs", help="Dataset: one of allmusic, tagtraum, discogs or lastfm.")
@click.option("--num_classes", default=315, help="Ys: one of 766 for allmusic, 296 for tagtraum, 315 for discogs or 327 for lastfm.")
@click.option("--patience", default=4, help="Patience is an early stopping parameter.")
def train(epochs, block_step, batch_size, seed, dataset, num_classes, patience):
    """
    sample calls:
    python train.py --batch_size 5 --block_step 20 --patience 4 --dataset discogs --num_classes 315  
    python train.py --batch_size 5 --block_step 20 --patience 4 --dataset lastfm --num_classes 327  
    python train.py --batch_size 5 --block_step 20 --patience 4 --dataset allmusic --num_classes 766  
    python train.py --batch_size 5 --block_step 20 --patience 4 --dataset tagtraum --num_classes 296  
    """

    y_path = "class_" + str(num_classes) + "_" + dataset

    torch.manual_seed(seed)

    model = Subtask2Model(num_classes)

    # gpu or cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'using {device}')
    model.to(device)

    print(model.state_dict().keys())
    print([(k, v.shape) for (k, v) in model.state_dict().items()])

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params: {pytorch_total_params}')
    print(f'Trainable params: {pytorch_total_trainable_params}')

if __name__ == "__main__":
    train()
