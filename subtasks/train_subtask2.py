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
            else:
                items_in_batch = items_list[j:]
            x_batch = x_block[items_in_batch]
            y_batch = y_block[items_in_batch]
            yield (x_batch, y_batch)

def batch_block_generator_x_y(dataset, block_step, batch_size, x, y):
    block_step = block_step
    batch_size = batch_size
    N_train = len(x)
    randomize = True
    # while 1:
    for i in range(0, N_train, block_step):
        x_block = x[i:min(N_train, i+block_step)]
        y_block = y[i:min(N_train, i+block_step)]
        items_list = list(range(x_block.shape[0]))
        if randomize:
            random.shuffle(items_list)        
        for j in range(0, len(items_list), batch_size):
            if j+batch_size <= x_block.shape[0]:
                items_in_batch = items_list[j:j+batch_size]
            else:
                items_in_batch = items_list[j:]
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
    python train_subtask2.py --batch_size 5 --block_step 20 --patience 4 --dataset discogs --num_classes 315  
    python train_subtask2.py --batch_size 5 --block_step 20 --patience 4 --dataset lastfm --num_classes 327  
    python train_subtask2.py --batch_size 5 --block_step 20 --patience 4 --dataset allmusic --num_classes 766  
    python train_subtask2.py --batch_size 5 --block_step 20 --patience 4 --dataset tagtraum --num_classes 296  
    """

    y_path = "class_" + str(num_classes) + "_" + dataset

    torch.manual_seed(seed)

    model = Subtask2Model(num_classes)

    # gpu or cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f'using {device}')
    model.to(device)

    print(model.state_dict().keys())
    print([(k, v.shape) for (k, v) in model.state_dict().items()])

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params: {pytorch_total_params}')
    print(f'Trainable params: {pytorch_total_trainable_params}')

    optimizer = optim.Adam(params=model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    criterion = nn.BCELoss()

    X_train1, X_val1, X_test1 = common.load_x_data_preprocesed('discogs', dataset)
    X_train2, X_val2, X_test2 = common.load_x_data_preprocesed('allmusic', dataset)
    X_train3, X_val3, X_test3 = common.load_x_data_preprocesed('lastfm', dataset)
    X_train4, X_val4, X_test4 = common.load_x_data_preprocesed('tagtraum', dataset)

    X_train = np.concatenate([np.array(X_train1), np.array(X_train2), np.array(X_train3), np.array(X_train4)], axis=1)
    X_val = np.concatenate([np.array(X_val1), np.array(X_val2), np.array(X_val3), np.array(X_val4)], axis=1)
    X_test = np.concatenate([np.array(X_test1), np.array(X_test2), np.array(X_test3), np.array(X_test4)], axis=1)

    Y_train, Y_val, Y_test = common.load_y_data_preprocesed(y_path)

    print(np.asarray(X_train).shape)
    print(np.asarray(Y_train).shape)

    train_set = list(zip(X_train[:], Y_train[:]))
    print(np.asarray(train_set).shape)
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
    avg_valid_roc_aucs = []

    for epoch in range(1, epochs + 1):

        # set model mode to "training"
        model.train()
        # get blocks of specified batch size and use it to train the model
        for data, target in batch_block_generator_x_y(dataset, block_step, batch_size, X_train, Y_train):
            optimizer.zero_grad()
            output = model(torch.Tensor(data).to(device))
            loss = criterion(output, torch.Tensor(target).to(device))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # set model mode to "evaluation"
        model.eval()
        # calculate loss and AUC-ROC
        predicted_y = []
        true_y = []
        for data, target in validation_set:
            output = model(torch.Tensor(data).to(device))
            loss = criterion(output, torch.Tensor(target).to(device))
            valid_losses.append(loss.item())
            predicted_y.append(output.detach().cpu().numpy())
            true_y.append(target)

        try:
            valid_roc_auc = roc_auc_score(np.array(true_y), np.array(predicted_y))
        except:
            valid_roc_auc = 0.0

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
        avg_valid_roc_aucs.append(valid_roc_auc)
        
        epoch_len = len(str(epochs))
        
        print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f} ' +
                     f'valid_roc_auc: {valid_roc_auc:.5f} ' +
                     f'test_loss: {test_loss:.5f}')

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
    file = f'{common.SAVED_MODELS_DIR}/subtask2_{dataset}.pt'
    torch.save(model.state_dict(), file)
    print(f'best model for {dataset} saved to {file}')

if __name__ == "__main__":
    train()
