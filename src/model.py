"""
model.py
Defines the Neural Network:
    - forward
    - back
    - weights
    - initialisation
"""
import numpy as np
from .utils import relu, softmax

class NeuralNetwork():

    def __init__(self, w1, w2, b1, b2, learning_rate):
        self.w1 = w1
        self.w2 = w2
        self.b1 = b1
        self.b2 = b2
        self.learning_rate = learning_rate

    def forward(self, x: np.ndarray):

        z1 = x@self.w1+self.b1
        a1 = relu(z1)
        z2 = a1@self.w2+self.b2
        # activation layer
        a2 = softmax(z2)

        return z1, a1, z2, a2

    def backward(self, a1, a2, z1, z2, x, y_onehot):
        # output layer gradient
        dz2 =  a2 - y_onehot
        dw2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)
        dz1 = dz2 @ self.w2.T
        dz1[z1 <= 0] = 0
        dw1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return dw1, db1, dw2, db2

    def update_parameters(self, dw1, db1, dw2, db2):
        self.w1 -= self.learning_rate*dw1
        self.w2 -= self.learning_rate*dw2
        self.b1 -= self.learning_rate*db1
        self.b2 -= self.learning_rate*db2
