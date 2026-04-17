import functions
import imgTrans
from PIL import Image
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

device = torch.device("cpu")
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


model.load_state_dict(torch.load("best_cnn_relu(2).pth", map_location=device))
model.eval()

x = torch.randn(1, 784)
y = model(x)


transform = imgTrans.mnist_transform()
img = Image.open("imgs/dos.png").convert("L")
x = transform(img)
x = 1.0 - x
x = x.view(1, -1)
x = x.to(device)

with torch.no_grad():
    output = model(x)
    pred = output.argmax(dim=1).item()

plt.imshow(x.view(1, 28, 28)[0].cpu(), cmap="gray")
plt.title(f"Predicción: {pred}")
plt.axis("off")
plt.show()

