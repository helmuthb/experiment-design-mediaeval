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
from models import Subtask1Model

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
    # 07c41ad0-3102-40d2-8f93-50fe5c08a3e8              
    Y_val = np.asarray([id2gt[id.decode('utf-8')] for id in index_val])
    X_test = f['features'][N_train+N_val:N]
    index_test = f['index'][N_train+N_val:N]
    # print(index_test.shape)
    # print(X_test.shape)
    X_test = np.delete(X_test, np.where(index_test == ""), axis=0)
    index_test = np.delete(index_test, np.where(index_test == ""))                
    # print(index_test.shape)
    # print(X_test.shape)
    Y_test = np.asarray([id2gt[id.decode('utf-8')] for id in index_test])
    # print(Y_test.shape)
    index_train = f['index'][:N_train]
    index_train = np.delete(index_train, np.where(index_train == ""))
    N_train = index_train.shape[0]
    
    return X_val, Y_val, X_test, Y_test, N_train

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
@click.option("--num_workers", default=2, help="Number of Workers (at least 2 recommended).")
@click.option("--seed", default=73, help="Seed.")
@click.option("--dataset", default="discogs", help="Dataset: one of allmusic, tagtraum, discogs or lastfm.")
@click.option("--num_classes", default=315, help="Ys: one of 766 for allmusic, 296 for tagtraum, 315 for discogs or 327 for lastfm.")
@click.option("--patience", default=5, help="Patience is an early stopping parameter.")
def train(epochs, block_step, batch_size, num_workers, seed, dataset, num_classes, patience):
    """
    sample calls:
    python train.py --batch_size 5 --block_step 20 --patience 10 --dataset discogs --num_classes 315  
    python train.py --batch_size 5 --block_step 20 --patience 10 --dataset lastfm --num_classes 327  
    python train.py --batch_size 5 --block_step 20 --patience 10 --dataset allmusic --num_classes 766  
    python train.py --batch_size 5 --block_step 20 --patience 10 --dataset tagtraum --num_classes 296  
    """

    y_path = "class_" + str(num_classes) + "_" + dataset

    torch.manual_seed(seed)

    model = Subtask1Model(num_classes)

    # gpu or cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'using {device}')
    # device = torch.device("cpu")
    model.to(device)

    print(model.state_dict().keys())
    print([(k, v.shape) for (k, v) in model.state_dict().items()])

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params: {pytorch_total_params}')
    print(f'Trainable params: {pytorch_total_trainable_params}')

    # sgd = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=1e-6)
    optimizer = optim.Adam(params=model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    criterion = nn.BCELoss()

    id2gt = dict()
    factors = np.load(common.DATASETS_DIR+'/y_train_'+y_path+'.npy')
    index_factors = open(common.DATASETS_DIR+'/items_index_train_'+dataset+'.tsv').read().splitlines()
    id2gt = dict((index,factor) for (index,factor) in zip(index_factors,factors))
    X_val, Y_val, X_test, Y_test, N_train = load_data_hf5_memory(dataset, 0.1, 0.1, y_path, id2gt)

    validation_set = list(zip(X_val[:], Y_val[:]))
    test_set = list(zip(X_test[:], Y_test[:]))

    # see https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    train_losses = []
    valid_losses = []
    test_losses = []
    avg_train_losses = []
    avg_valid_losses = [] 
    avg_test_losses = [] 

    for epoch in range(1, epochs + 1):

        model.train()
        for index, data in enumerate(batch_block_generator(dataset, block_step, batch_size,y_path,N_train,id2gt), 0):

            inputs, labels = torch.Tensor(data[0]).to(device), torch.Tensor(data[1]).to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        for data, target in validation_set:

            output = model(torch.Tensor(data).to(device))

            loss = criterion(output, torch.Tensor(target).to(device))

            valid_losses.append(loss.item())

        for data, target in test_set:

            output = model(torch.Tensor(data).to(device))

            loss = criterion(output, torch.Tensor(target).to(device))

            test_losses.append(loss.item())

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        test_loss = np.average(test_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        avg_test_losses.append(test_loss)
        
        epoch_len = len(str(epochs))
        
        print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f} ' +
                     f'test_loss: {test_loss:.5f} ')

        print(print_msg)

        train_losses = []
        valid_losses = []
        test_losses = []
        
        early_stopping(valid_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load best model
    model.load_state_dict(torch.load('checkpoint.pt'))
    file = f'{common.SAVED_MODELS_DIR}/subtask1_{dataset}.pt'
    torch.save(model.state_dict(), file)
    print(f'best model for {dataset} saved to {file}')

    # TODO save activations from feeding OTHER datasets (whatever that means)
    # weights = model.state_dict()['input.weight']
    # file = f'{common.EMBEDDED_DIR}/embedded_vector_{dataset}.npy'
    # np.save(file, weights.cpu().numpy())
    # print(f'best embedded vector for {dataset} saved to {file}')


if __name__ == "__main__":
    train()