import numpy as np
import h5py

DATA_DIR = "../data-genres"
PATCHES_DIR = DATA_DIR+"/patches"
DATASETS_DIR = DATA_DIR+"/splits"
TRAINDATA_DIR = DATA_DIR+"/train_data"
SAVED_MODELS_DIR = DATA_DIR+"/saved_models"

def load_data_hf5_memory(dataset, y_path, id2gt):
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
    index_val = [i.decode('utf-8') for i in f_val['index'][:] if i.decode('utf-8') in id2gt_val]
    X_val = np.delete(X_val, np.where(index_val == ""), axis=0)
    index_val = np.delete(index_val, np.where(index_val == ""))                

    Y_val = np.asarray([id2gt_val[id] for id in index_val])

    test_hdf5_file = f"{PATCHES_DIR}/patches_test_{dataset}_1x1.hdf5"
    f_test = h5py.File(test_hdf5_file,"r")
    X_test = f_test['features'][:]
    factors_test = np.load(DATASETS_DIR+'/y_test_'+y_path+'.npy')
    index_factors_test = open(DATASETS_DIR+'/items_index_test_'+dataset+'.tsv').read().splitlines()
    id2gt_test = dict((index,factor) for (index,factor) in zip(index_factors_test,factors_test))
    index_test = [i.decode('utf-8') for i in f_test['index'][:] if i.decode('utf-8') in id2gt_test]
    X_test = np.delete(X_test, np.where(index_test == ""), axis=0)
    index_test = np.delete(index_test, np.where(index_test == ""))                

    Y_test = np.asarray([id2gt_test[id] for id in index_test])
    
    return X_val, Y_val, X_test, Y_test, N_train

def load_x_data_preprocesed(dataset, metadata_source):
    all_X_meta = np.load(TRAINDATA_DIR+'/X_train_%s_%s.npy' % (metadata_source,dataset))
    all_X = all_X_meta
    print(all_X.shape)
    X_val = np.load(TRAINDATA_DIR+'/X_val_%s_%s.npy' % (metadata_source,dataset))
    X_test = np.load(TRAINDATA_DIR+'/X_test_%s_%s.npy' % (metadata_source,dataset))
    X_train = all_X

    return X_train, X_val, X_test

def load_y_data_preprocesed(Y_path):
    factors = np.load(DATASETS_DIR+'/y_train_'+Y_path+'.npy')
    all_Y = factors
    print(all_Y.shape)
    Y_val = np.load(DATASETS_DIR+'/y_val_'+Y_path+'.npy')
    Y_test = np.load(DATASETS_DIR+'/y_test_'+Y_path+'.npy')
    Y_train = all_Y

    return Y_train, Y_val, Y_test