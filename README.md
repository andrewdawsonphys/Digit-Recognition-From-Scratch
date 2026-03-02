# Digit Recognition From Scratch

This project implements a handwritten digit classifier from first principles using only NumPy.

No PyTorch. No TensorFlow. No automatic differentiation.

The goal is to deeply understand how neural networks actually work by manually deriving and implementing:

- Forward propagation
- ReLU activation
- Softmax output layer
- Cross-entropy loss
- Backpropagation via the chain rule
- Gradient descent parameter updates

The model is trained on the MNIST dataset and achieves competitive accuracy using a fully connected feedforward architecture.

---

## Why This Project?

Modern ML frameworks abstract away the mechanics of training neural networks.  
This project removes that abstraction and rebuilds everything from scratch to understand:

- How gradients flow through a network
- Why non-linear activations are necessary
- How softmax and cross-entropy interact
- How weight updates improve predictions

---

## Architecture

Input → Linear → ReLU → Linear → Softmax

All gradients are derived manually and implemented explicitly.

---

## Tech Stack

- Python
- NumPy
- Matplotlib (for visualisation)

---

## Future Improvements

- Add L2 regularization
- Implement momentum / Adam
- Extend to CNNs
- Improve training stability

---

This repository focuses on clarity and mathematical understanding over framework convenience.
