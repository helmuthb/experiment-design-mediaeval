import numpy as np
import h5py

DATA_DIR = "../data-genres"
PATCHES_DIR = DATA_DIR+"/patches"
DATASETS_DIR = DATA_DIR+"/splits"
TRAINDATA_DIR = DATA_DIR+"/train_data"
SAVED_MODELS_DIR = DATA_DIR+"/saved_models"

def load_data_hf5_memory(dataset,val_percent, test_percent, y_path, id2gt):
    hdf5_file = f"{PATCHES_DIR}/patches_train_{dataset}_1x1.hdf5"
    f = h5py.File(hdf5_file,"r")
    index_train = f["index"][:]
    index_train = np.delete(index_train, np.where(index_train == ""))

    N_train = index_train.shape[0]

    val_hdf5_file = f"{PATCHES_DIR}/patches_val_{dataset}_1x1.hdf5"
    f_val = h5py.File(val_hdf5_file,"r")
    X_val = f_val['features'][:]
    factors_val = np.load(DATASETS_DIR+'/y_val_'+y_path+'.npy')
    index_factors_val = open(DATASETS_DIR+'/items_index_val_'+dataset+'.tsv').read().splitlines()
    id2gt_val = dict((index,factor) for (index,factor) in zip(index_factors_val,factors_val))
    index_val = [i for i in f_val['index'][:] if i in id2gt_val]
    X_val = np.delete(X_val, np.where(index_val == ""), axis=0)
    index_val = np.delete(index_val, np.where(index_val == ""))                

    Y_val = np.asarray([id2gt_val[id] for id in index_val])

    test_hdf5_file = f"{PATCHES_DIR}/patches_test_{dataset}_1x1.hdf5"
    f_test = h5py.File(test_hdf5_file,"r")
    X_test = f_test['features'][:]
    factors_test = np.load(DATASETS_DIR+'/y_test_'+y_path+'.npy')
    index_factors_test = open(DATASETS_DIR+'/items_index_test_'+dataset+'.tsv').read().splitlines()
    id2gt_test = dict((index,factor) for (index,factor) in zip(index_factors_test,factors_test))
    index_test = [i for i in f_test['index'][:] if i in id2gt_test]
    X_test = np.delete(X_test, np.where(index_test == ""), axis=0)
    index_test = np.delete(index_test, np.where(index_test == ""))                

    Y_test = np.asarray([id2gt_test[id] for id in index_test])

    hdf5_file = f"{PATCHES_DIR}/patches_train_{dataset}_1x1.hdf5"
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
    Y_val = np.asarray([id2gt[id.decode('utf-8')] for id in index_val])
    X_test = f['features'][N_train+N_val:N]
    index_test = f['index'][N_train+N_val:N]
    X_test = np.delete(X_test, np.where(index_test == ""), axis=0)
    index_test = np.delete(index_test, np.where(index_test == ""))                
    Y_test = np.asarray([id2gt[id.decode('utf-8')] for id in index_test])
    index_train = f['index'][:N_train]
    index_train = np.delete(index_train, np.where(index_train == ""))
    N_train = index_train.shape[0]
    
    return X_val, Y_val, X_test, Y_test, N_train

