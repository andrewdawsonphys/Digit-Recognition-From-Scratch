"""
train.py

Contains the training logic: 
- loops over batches
- computes loss
- updates weights
"""

import numpy as np
import matplotlib.pyplot as plt
from .utils import batch_generator
from .model import NeuralNetwork

def train(x_train, y_train):

    W1 = np.random.randn(784, 64) * 0.01
    b1 = np.zeros((1, 64))
    W2 = np.random.randn(64, 10) * 0.01
    b2 = np.zeros((1, 10))
    learning_rate = 0.01
    num_epochs = 10

    nn = NeuralNetwork(W1, W2, b1, b2, learning_rate)
    batch_losses = []
    batch_indices = []
    batch_count = 0
    for epoch in range(num_epochs):
        for x_batch, y_batch in batch_generator(x_train, y_train, 50):
            z1, a1, z2, a2 = nn.forward(x_batch)

            eps = 1e-12
            num_classes = a2.shape[1]
            y_batch_onehot = np.eye(num_classes)[y_batch]
            batch_loss = -np.sum(y_batch_onehot * np.log(a2 + eps)) / x_batch.shape[0]

            print(f"Epoch {epoch+1}, Batch {batch_count+1}, Loss: {batch_loss:.4f}")
            batch_losses.append(batch_loss)
            batch_indices.append(batch_count)
            batch_count += 1
            dw1, db1, dw2, db2 = nn.backward(a1, a2, z1, z2, x_batch, y_batch_onehot)
            nn.update_parameters(dw1, db1, dw2, db2)

    # Plot batch loss vs. batch number
    plt.figure(figsize=(10, 5))
    plt.plot(batch_indices, batch_losses, label="Batch Loss")
    plt.xlabel("Batch Number")
    plt.ylabel("Loss")
    plt.title("Batch Loss vs. Batch Number")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return nn
