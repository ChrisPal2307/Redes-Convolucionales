
import torchvision.transforms as transforms


def mnist_transform():

    return transforms.Compose([
        # 1. Convertir a escala de grises (1 canal)
        transforms.Grayscale(num_output_channels=1),

        # 2. Redimensionar a 28x28 (MNIST)
        transforms.Resize((28, 28)),

        # 3. Convertir a tensor y normalizar a [0,1]
        transforms.ToTensor(),

    ])