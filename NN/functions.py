import gzip
import pickle
import torch
import torch.nn as nn

def load_data_torch(filename="data/mnist.pkl.gz", device="cpu"):
    with gzip.open(filename, "rb") as f:
        training_data, validation_data, test_data = pickle.load(
            f, encoding="latin1"
        )

    def tensorize(data):
        x = torch.tensor(data[0], dtype=torch.float32, device=device) #imágenes (28*28px -> 784)
        y = torch.tensor(data[1], dtype=torch.long, device=device) #etiquetas (0-9)
        return x, y

    return (
        tensorize(training_data),
        tensorize(validation_data),
        tensorize(test_data)
    )

class Network(nn.Module):

    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    #FLujo completo de la red:
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def SGD_torch(
        model,
        train_loader,
        val_loader,
        test_loader,
        epochs,
        eta, #lr
        lmbda=0.0,
        device="cpu"
    ):
        criterion = nn.CrossEntropyLoss()  #función de pérdida para clasificación -> loss = -log(softmax(p))

        #Descenso de gradiente:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=eta,
            weight_decay=lmbda   #  regularización -> w=w−η(dL+lmbda*w)
        )

        best_val_acc = 0.0

        for epoch in range(epochs):
            model.train()

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()    #Calcular gradientes (dL/dw)
                optimizer.step()   #Actualización de los pesos

            model.eval()
            correct, total = 0, 0

            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    output = model(x)
                    preds = output.argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)

            val_acc = correct / total
            print(f"Epoch {epoch}: validation accuracy {val_acc:.2%}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print("This is the best validation accuracy to date.")
                torch.save(
                    model.state_dict(),
                    "best_cnn_relu(2).pth"
                )

                if test_loader:
                    correct, total = 0, 0
                    with torch.no_grad():
                        for x, y in test_loader:
                            x, y = x.to(device), y.to(device)
                            output = model(x)
                            preds = output.argmax(dim=1)
                            correct += (preds == y).sum().item()
                            total += y.size(0)

                    test_acc = correct / total
                    print(f"Corresponding test accuracy: {test_acc:.2%}")

        print("Finished training network.")


class ConvPoolLayer(nn.Module):  #Conv + pooling

    def __init__(
        self,
        filter_shape,
        poolsize=(2, 2),
        activation_fn=None
    ):
        super().__init__()

        num_filters, in_channels, filter_h, filter_w = filter_shape
        self.poolsize = poolsize

        # Capa Convolucional:
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_filters,
            kernel_size=(filter_h, filter_w),
            bias=True
        )

        # Funcion de activación:
        self.activation = activation_fn if activation_fn is not None else nn.ReLU()

        #Inicialización de pesos de la capa Conv (distribución normal)
        nn.init.kaiming_normal_(
            self.conv.weight,
            mode='fan_in',
            nonlinearity='relu'
        )
        nn.init.zeros_(self.conv.bias)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=poolsize)

    def forward(self, x):
        x = self.conv(x)       # Convolución
        x = self.pool(x)       # Max Pooling
        x = self.activation(x) # ReLU
        return x


class FullyConnectedLayer(nn.Module):

    def __init__(self, n_in, n_out, activation_fn=None, p_dropout=0.0):
        super().__init__()

        self.fc = nn.Linear(n_in, n_out)
        self.activation = activation_fn if activation_fn is not None else nn.ReLU()
        self.dropout = nn.Dropout(p_dropout)

        #Inicialización de pesos de la capa FullyConnected (distribución normal)
        nn.init.kaiming_normal_(
            self.fc.weight,
            mode='fan_in',
            nonlinearity='relu'
        )
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class SoftmaxLayer(nn.Module):
    def __init__(self, n_in, n_out, p_dropout=0.5):
        super().__init__()
        self.fc = nn.Linear(n_in, n_out)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)