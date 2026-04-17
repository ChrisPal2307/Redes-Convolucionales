import functions
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print(torch.cuda.get_device_name(0))


def basic_conv_torch(
    epochs=60,
    mini_batch_size=10,
    eta=0.1,
    lmbda=0.0,
    device="cpu"
):
    training_data, validation_data, test_data = functions.load_data_torch(device=device)

    X_train, y_train = training_data
    X_val, y_val     = validation_data
    X_test, y_test   = test_data

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=mini_batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=mini_batch_size
    )

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=mini_batch_size
    )


    print("CONV-1")

    model = functions.Network([
        # 784 -> 28 * 28
        nn.Unflatten(1, (1, 28, 28)),

        functions.ConvPoolLayer(
            filter_shape=(32, 1, 3, 3),  #32 filtros, 1 canal (b/n), alto, ancho
            poolsize=(2, 2)
        ),
        ## -> (32, 13, 13)
        functions.Flatten(),
        #32 * 13 * 13
        functions.FullyConnectedLayer(
            n_in=32 * 13 * 13,  # 26/2 = 13
            n_out=100  #Neuronas de la capa FC
        ),

        functions.SoftmaxLayer(
            n_in=100,
            n_out=10
        )
    ]).to(device)

    # entrenamiento:
    functions.Network.SGD_torch(
        model,
        train_loader,
        val_loader,
        test_loader,
        epochs=epochs,
        eta=eta,
        lmbda=lmbda,
        device=device
    )
    return model


def dbl_conv_torch(
        epochs=60,
        mini_batch_size=10,
        eta=0.1,
        lmbda=0.0,
        device="cpu"
):
    training_data, validation_data, test_data = functions.load_data_torch(device=device)

    X_train, y_train = training_data
    X_val, y_val = validation_data
    X_test, y_test = test_data

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=mini_batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=mini_batch_size
    )

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=mini_batch_size
    )

    print("CONV-2")

    model = functions.Network([
        nn.Unflatten(1, (1, 28, 28)),

        functions.ConvPoolLayer(
            filter_shape=(32, 1, 3, 3),
            poolsize=(2, 2)
        ),

        functions.ConvPoolLayer(
            filter_shape=(64, 32, 3, 3),
            poolsize=(2, 2)
        ),

        functions.Flatten(),

        functions.FullyConnectedLayer(
            n_in=64 * 5 * 5,
            n_out=128
        ),

        functions.SoftmaxLayer(
            n_in=128,
            n_out=10
        )
    ]).to(device)

    functions.Network.SGD_torch(
        model,
        train_loader,
        val_loader,
        test_loader,
        epochs=epochs,
        eta=eta,
        lmbda=lmbda,
        device=device
    )
    return model


dbl_conv_torch(25,64,0.05,0.0,device)
