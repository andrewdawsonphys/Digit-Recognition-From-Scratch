from src.utils import load_mnist
from src.train import train

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_mnist()

    # shape of normalised_x_train: (60000, 28, 28), dtype: float64
    # 60000 samples, each sample is a 28x28 image,
    # pixel values are normalized to [0, 1] range
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # reshape normalised_x_train to (60000, 784) for visualization
    x_train = x_train.reshape(-1, 28*28)
    x_test = x_test.reshape(-1, 28*28)


    model = train(x_train, y_train)
