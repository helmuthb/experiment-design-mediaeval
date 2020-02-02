import click
import h5py
import numpy as np
import torch
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import common 
from models import EmbeddedVectorModel

def load_data_hf5_memory(dataset,val_percent, test_percent, y_path, id2gt):
    hdf5_file = f"{common.PATCHES_DIR}/patches_train_{dataset}_1x1.hdf5"
    f = h5py.File(hdf5_file,"r")
    index_train = f["index"][:]
    index_train = np.delete(index_train, np.where(index_train == ""))

    N_train = index_train.shape[0]

    val_hdf5_file = f"{common.PATCHES_DIR}/patches_val_{dataset}_1x1.hdf5"
    f_val = h5py.File(val_hdf5_file,"r")
    X_val = f_val['features'][:]
    factors_val = np.load(common.DATASETS_DIR+'/y_val_'+y_path+'.npy')
    index_factors_val = open(common.DATASETS_DIR+'/items_index_val_'+dataset+'.tsv').read().splitlines()
    id2gt_val = dict((index,factor) for (index,factor) in zip(index_factors_val,factors_val))
    index_val = [i for i in f_val['index'][:] if i in id2gt_val]
    X_val = np.delete(X_val, np.where(index_val == ""), axis=0)
    index_val = np.delete(index_val, np.where(index_val == ""))                

    Y_val = np.asarray([id2gt_val[id] for id in index_val])

    test_hdf5_file = f"{common.PATCHES_DIR}/patches_test_{dataset}_1x1.hdf5"
    f_test = h5py.File(test_hdf5_file,"r")
    X_test = f_test['features'][:]
    factors_test = np.load(common.DATASETS_DIR+'/y_test_'+y_path+'.npy')
    index_factors_test = open(common.DATASETS_DIR+'/items_index_test_'+dataset+'.tsv').read().splitlines()
    id2gt_test = dict((index,factor) for (index,factor) in zip(index_factors_test,factors_test))
    index_test = [i for i in f_test['index'][:] if i in id2gt_test]
    X_test = np.delete(X_test, np.where(index_test == ""), axis=0)
    index_test = np.delete(index_test, np.where(index_test == ""))                

    Y_test = np.asarray([id2gt_test[id] for id in index_test])

    hdf5_file = f"{common.PATCHES_DIR}/patches_train_{dataset}_1x1.hdf5"
    f = h5py.File(hdf5_file,"r")
    index_all = f["index"][:]
    N = index_all.shape[0]
    train_percent = 1 - val_percent - test_percent
    N_train = int(train_percent * N)
    N_val = int(val_percent * N)
    X_val = f['features'][N_train:N_train+N_val]
    index_val = f['index'][N_train:N_train+N_val]
    X_val = np.delete(X_val, np.where(index_val == ""), axis=0)
    index_val = np.delete(index_val, np.where(index_val == ""))                
    Y_val = np.asarray([id2gt[id] for id in index_val])
    X_test = f['features'][N_train+N_val:N]
    index_test = f['index'][N_train+N_val:N]
    print(index_test.shape)
    print(X_test.shape)
    X_test = np.delete(X_test, np.where(index_test == ""), axis=0)
    index_test = np.delete(index_test, np.where(index_test == ""))                
    print(index_test.shape)
    print(X_test.shape)
    Y_test = np.asarray([id2gt[id] for id in index_test])
    print(Y_test.shape)
    index_train = f['index'][:N_train]
    index_train = np.delete(index_train, np.where(index_train == ""))
    N_train = index_train.shape[0]
    
    return X_val, Y_val, X_test, Y_test, N_train

def batch_block_generator(batch_size, y_path, N_train, id2gt):
    hdf5_file = f"{common.PATCHES_DIR}/patches_train_{dataset}_1x1.hdf5"
    f = h5py.File(hdf5_file,"r")
    block_step = 50000
    batch_size = batch_size
    randomize = True
    with_meta = False
    if X_meta != None:
        with_meta = True
    while 1:
        for i in range(0, N_train, block_step):
            x_block = f['features'][i:min(N_train, i+block_step)]
            index_block = f['index'][i:min(N_train, i+block_step)]
            #y_block = f['targets'][i:min(N_train,i+block_step)]
            x_block = np.delete(x_block, np.where(index_block == ""), axis=0)
            index_block = np.delete(index_block, np.where(index_block == ""))
            y_block = np.asarray([id2gt[id] for id in index_block])
            if params['training']['normalize_y']:
                normalize(y_block, copy=False)
            items_list = range(x_block.shape[0])
            if randomize:
                random.shuffle(items_list)
            for j in range(0, len(items_list), batch_size):
                if j+batch_size <= x_block.shape[0]:
                    items_in_batch = items_list[j:j+batch_size]
                    x_batch = x_block[items_in_batch]
                    y_batch = y_block[items_in_batch]
                    if with_meta:
                        x_batch = [x_batch, X_meta[items_in_batch]]
                    yield (x_batch, y_batch)

@click.command()
@click.option("--epochs", default=2, help="Number of epochs.")
@click.option("--batch_size", default=64, help="Batch Size.")
@click.option("--num_workers", default=2, help="Number of Workers (at least 2 recommended).")
@click.option("--seed", default=73, help="Seed.")
@click.option("--dataset", default="discogs", help="Dataset: one of allmusic, tagtraum, discogs or lastfm.")
@click.option("--x_path", default="unused", help="unused")
@click.option("--y_path", default="class_315_discogs", help="Ys: one of class_766_allmusic, class_296_tagtraum, class_315_discogs or class_327_lastfm.")
def train(epochs, batch_size, num_workers, seed, dataset, x_path, y_path):

    torch.manual_seed(seed)

    model = EmbeddedVectorModel(315)

    # gpu or cpu
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model.to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params: {pytorch_total_params}')
    print(f'Trainable params: {pytorch_total_trainable_params}')

    # # throws warning
    # print(F.binary_cross_entropy(F.softmax(pred), y))
    # # doesn't throw warning
    # print(F.binary_cross_entropy(F.softmax(pred, dim=pred.shape[0]), y))

    sgd_optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=1e-6)
    binary_cross_entropy_loss = nn.BCELoss()


    id2gt = dict()
    factors = np.load(common.DATASETS_DIR+'/y_train_'+y_path+'.npy')
    index_factors = open(common.DATASETS_DIR+'/items_index_train_'+dataset+'.tsv').read().splitlines()
    id2gt = dict((index,factor) for (index,factor) in zip(index_factors,factors))
    X_val, Y_val, X_test, Y_test, N_train = load_data_hf5_memory(dataset, 0.1, 0.1, y_path, id2gt)
    if params['dataset']['nsamples'] != 'all':
        N_train = min(N_train,params['dataset']['nsamples'])


    epochs = model.fit_generator(batch_block_generator(batch_size,config.y_path,N_train,id2gt),
                samples_per_epoch = N_train-(N_train % config.training_params["n_minibatch"]),
                nb_epoch = config.training_params["n_epochs"],
                verbose=1,
                validation_data = (X_val, Y_val),
                callbacks=[early_stopping])
                
    # TODO? early stopping
    for epoch in range(epochs):
        losses = []
        for data in enumerate(batch_block_generator(batch_size,config.y_path,N_train,id2gt), 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            sgd_optimizer.zero_grad()

            outputs = model(inputs)

            loss = binary_cross_entropy_loss(outputs, labels)

            loss.backward()

            optimizer.step()

            # monitor_metric = 'val_loss'
            losses.append(loss.data.mean())


if __name__ == "__main__":
    train()