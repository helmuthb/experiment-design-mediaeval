import torch
import torchvision
from torchvision import transforms, datasets

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class EmbeddedVectorModel(nn.Module):

    def __init__(self, output_dim):
        super().__init__()
        
        # SGD hyper parameters
        self.learning_rate = 0.1
        self.momentum = 0.95
        # network hyper parameters
        self.dropout_factor = 0.5

        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(2669, 256),
            nn.Dropout(p=self.dropout_factor),
            nn.ReLU(),
            nn.Linear(256, self.output_dim),
            nn.Sigmoid()
        )

model = EmbeddedVectorModel(315)
print(model)
print(model.state_dict().keys())
print([(k, v.shape) for (k, v) in model.state_dict().items()])
dense_vector = model.state_dict()['net.0.weight'][:]
print(type(dense_vector))
print(dense_vector.shape)
dense_vector = model.state_dict()['net.3.weight'][:]
print(type(dense_vector))
print(dense_vector.shape)
pytorch_total_params = sum(p.numel() for p in model.parameters())
pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total params: {pytorch_total_params}')
print(f'Trainable params: {pytorch_total_trainable_params}')

class Net(nn.Module):

    def __init__(self, output_dim):
        super(Net, self).__init__()
        # network hyper parameters
        self.dropout_factor = 0.5

        # network architecture
        self.input = nn.Linear(2669, 256)
        self.dense = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.dropout(x, p=self.dropout_factor)
        x = F.sigmoid(self.dense(x))
        return x

net = Net(315)
pytorch_total_params = sum(p.numel() for p in model.parameters())
pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total params: {pytorch_total_params}')
print(f'Trainable params: {pytorch_total_trainable_params}')

torch.manual_seed(73)
pred = torch.randn(1, 10)
y = torch.randn(1, 10)
# throws warning
print(F.binary_cross_entropy(F.softmax(pred), y))
# doesn't throw warning
print(F.binary_cross_entropy(F.softmax(pred, dim=pred.shape[0]), y))

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=1e-6)

        # >>> optimizer.zero_grad()
        # >>> loss_fn(model(input), target).backward()
        # >>> optimizer.step()

    # nparams = copy.deepcopy(models.params_1)
    # nparams["dataset"]["evaluation"] = 'multilabel' # binary/multilabel/multiclass/recommendation
    # nparams["dataset"]["fact"] = 'class'
    # nparams["dataset"]["npatches"] = 1
    # nparams["dataset"]["with_metadata"] = False
    # nparams["dataset"]["only_metadata"] = False
    # nparams["dataset"]["configuration"] = suffix
    # nparams["dataset"]["sparse"] = False
    # nparams["dataset"]["window"] = 1
    # nparams["training"]["val_from_file"] = True
    # nparams["dataset"]["dim"] = 315
    # nparams["training"]["loss_func"] = 'binary_crossentropy'
    # nparams["training"]["optimizer"] = 'adam'
    # nparams["training"]["normalize_y"] = False
    # nparams["cnn"]["architecture"] = '3'
    # nparams["cnn"]["n_dense"] = 256
    # nparams["cnn"]["n_dense_2"] = 0
    # nparams["cnn"]["dropout_factor"] = 0.5
    # nparams["cnn"]["final_activation"] = 'sigmoid'
    # nparams['cnn']['n_metafeatures'] = 2669
    # nparams["dataset"]["nsamples"] = 'all'
    # nparams["dataset"]["dataset"] = 'discogs'
    # nparams["dataset"]["meta-suffix"] = meta_suffix
    # add_extra_params(nparams, extra_params)
    # params['genres_discogs'] = copy.deepcopy(nparams)

    # inputs2 = Input(shape=(params["n_metafeatures"],))
    # x2 = Dropout(params["dropout_factor"])(inputs2)

    # if params["n_dense"] > 0:
    #     dense2 = Dense(output_dim=params["n_dense"], init="uniform", activation='relu')
    #     x2 = dense2(x2)
    #     logging.debug("Output CNN: %s" % str(dense2.output_shape))

    #     x2 = Dropout(params["dropout_factor"])(x2)

    # if params["n_dense_2"] > 0:
    #     dense3 = Dense(output_dim=params["n_dense_2"], init="uniform", activation='relu')
    #     x2 = dense3(x2)
    #     logging.debug("Output CNN: %s" % str(dense3.output_shape))

    #     x2 = Dropout(params["dropout_factor"])(x2)

    # dense4 = Dense(output_dim=params["n_out"], init="uniform", activation=params['final_activation'])
    # xout = dense4(x2)
    # logging.debug("Output CNN: %s" % str(dense4.output_shape))

    # if params['final_activation'] == 'linear':
    #     reg = Lambda(lambda x :K.l2_normalize(x, axis=1))
    #     xout = reg(xout)

    # model = Model(input=inputs2, output=xout)

    # t_params = config.training_params
    # sgd = SGD(lr=t_params["learning_rate"], decay=t_params["decay"],
    #           momentum=t_params["momentum"], nesterov=t_params["nesterov"])
    # adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # optimizer = eval(t_params['optimizer'])
    # metrics = ['mean_squared_error']
    # if config.model_arch["final_activation"] == 'softmax':
    #     metrics.append('categorical_accuracy')
    # if t_params['loss_func'] == 'cosine':
    #     loss_func = eval(t_params['loss_func'])
    # else:
    #     loss_func = t_params['loss_func']
    # model.compile(loss=loss_func, optimizer=optimizer,metrics=metrics)
