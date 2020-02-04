import click
import h5py
import numpy as np
import torch
from torchvision import datasets

import common 
from models import Subtask1Model

def get_activation(batch_size, model):
    for index, data in enumerate(batch_block_generator(dataset,batch_size,y_path,N_train,id2gt), 0):
        # get activations
        model.get_activation()

def batch_block_generator(dataset, block_step, batch_size, y_path, N_train, id2gt):
    hdf5_file = f"{common.PATCHES_DIR}/patches_train_{dataset}_1x1.hdf5"
    f = h5py.File(hdf5_file,"r")
    # block_step = 50000
    block_step = block_step
    batch_size = batch_size
    # while 1:
    for i in range(0, N_train, block_step):
        x_block = f['features'][i:min(N_train, i+block_step)]
        index_block = f['index'][i:min(N_train, i+block_step)]
        #y_block = f['targets'][i:min(N_train,i+block_step)]
        x_block = np.delete(x_block, np.where(index_block == ""), axis=0)
        index_block = np.delete(index_block, np.where(index_block == ""))
        y_block = np.asarray([id2gt[id.decode('utf-8')] for id in index_block])
        items_list = list(range(x_block.shape[0]))
        for j in range(0, len(items_list), batch_size):
            if j+batch_size <= x_block.shape[0]:
                items_in_batch = items_list[j:j+batch_size]
                x_batch = x_block[items_in_batch]
                y_batch = y_block[items_in_batch]
                yield (x_batch, y_batch)

@click.command()
@click.option("--batch_size", default=1, help="Batch Size.")
@click.option("--block_step", default=1, help="Block Step.")
@click.option("--dataset", default="discogs", help="Dataset: one of allmusic, tagtraum, discogs or lastfm.")
@click.option("--num_classes", default=315, help="Ys: one of 766 for allmusic, 296 for tagtraum, 315 for discogs or 327 for lastfm.")
def feed(batch_size, block_step, dataset, num_classes):
    # gpu or cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'using {device}')

    # load dataset
    id2gt = dict()
    y_path = "class_" + str(num_classes) + "_" + dataset
    factors = np.load(common.DATASETS_DIR+'/y_train_'+y_path+'.npy')
    index_factors = open(common.DATASETS_DIR+'/items_index_train_'+dataset+'.tsv').read().splitlines()
    id2gt = dict((index,factor) for (index,factor) in zip(index_factors,factors))
    hdf5_file = f"{common.PATCHES_DIR}/patches_train_{dataset}_1x1.hdf5"
    f = h5py.File(hdf5_file,"r")
    index_train = f["index"][:]
    index_train = np.delete(index_train, np.where(index_train == ""))
    N_train = index_train.shape[0]

    #########
    # discogs
    #########
    discogs_model = Subtask1Model(315)
    model_name = 'discogs'
    # load state
    file = f'{common.SAVED_MODELS_DIR}/subtask1_discogs.pt'
    discogs_model.load_state_dict(torch.load(file))
    discogs_model.to(device)
    discogs_model_activations = []

    ##########
    # tagtraum
    ##########
    tagtraum_model = Subtask1Model(296)
    model_name = 'tagtraum'
    # load state
    file = f'{common.SAVED_MODELS_DIR}/subtask1_tagtraum.pt'
    tagtraum_model.load_state_dict(torch.load(file))
    tagtraum_model.to(device)
    tagtraum_model_activations = []

    ########
    # lastfm
    ########
    lastfm_model = Subtask1Model(327)
    model_name = 'lastfm'
    # load state
    file = f'{common.SAVED_MODELS_DIR}/subtask1_lastfm.pt'
    lastfm_model.load_state_dict(torch.load(file))
    lastfm_model.to(device)
    lastfm_model_activations = []

    ##########
    # allmusic
    ##########
    allmusic_model = Subtask1Model(766)
    model_name = 'allmusic'
    # load state
    file = f'{common.SAVED_MODELS_DIR}/subtask1_allmusic.pt'
    allmusic_model.load_state_dict(torch.load(file))
    allmusic_model.to(device)
    allmusic_model_activations = []

    # get the activation froem each model for given dataset
    for _, data in enumerate(batch_block_generator(dataset, block_step, batch_size, y_path, N_train, id2gt), 0):
        discogs_model_activations.append(discogs_model.get_activation(torch.Tensor(data[0]).to(device)).detach().numpy())
        tagtraum_model_activations.append(tagtraum_model.get_activation(torch.Tensor(data[0]).to(device)).detach().numpy())
        lastfm_model_activations.append(lastfm_model.get_activation(torch.Tensor(data[0]).to(device)).detach().numpy())
        allmusic_model_activations.append(allmusic_model.get_activation(torch.Tensor(data[0]).to(device)).detach().numpy())

    file = f'{common.TRAINDATA_DIR}/X_train__discogs_{dataset}.npy'
    np.save(file, discogs_model_activations)
    print(f'discogs model for dataset {dataset} saved to {file}')

    file = f'{common.TRAINDATA_DIR}/X_train__tagtraum_{dataset}.npy'
    np.save(file, tagtraum_model_activations)
    print(f'tagtraum model for dataset {dataset} saved to {file}')

    file = f'{common.TRAINDATA_DIR}/X_train__lastfm_{dataset}.npy'
    np.save(file, lastfm_model_activations)
    print(f'lastfm model for dataset {dataset} saved to {file}')

    file = f'{common.TRAINDATA_DIR}/X_train__allmusic_{dataset}.npy'
    np.save(file, allmusic_model_activations)
    print(f'allmusic model for dataset {dataset} saved to {file}')

if __name__ == "__main__":
    feed()