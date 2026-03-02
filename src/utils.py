import os
import kagglehub
import numpy as np
import struct

def download_data() -> str:
    """Download MNIST Dataset and return path to data"""
    return kagglehub.dataset_download("hojjatk/mnist-dataset")


def load_idx_images(file_path: str) -> np.ndarray:
    """Load MNIST images from IDX file."""
    with open(file_path, 'rb') as f:
        _, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape((size, nrows, ncols))
    return data

def load_idx_labels(file_path: str) -> np.ndarray:
    """Load MNIST labels from IDX file."""
    with open(file_path, 'rb') as f:
        _, size = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def load_mnist(data_dir: str = None):
    """
    Download (if needed) and load MNIST dataset.
    Returns:
        (x_train, y_train), (x_test, y_test) as numpy arrays
    """
    if data_dir is None:
        data_dir = download_data()
    # MNIST files
    files = {
        'train_images': os.path.join(data_dir, 'train-images.idx3-ubyte'),
        'train_labels': os.path.join(data_dir, 'train-labels.idx1-ubyte'),
        'test_images': os.path.join(data_dir, 't10k-images.idx3-ubyte'),
        'test_labels': os.path.join(data_dir, 't10k-labels.idx1-ubyte'),
    }
    x_train = load_idx_images(files['train_images'])  # type: np.ndarray  # shape: (60000, 28, 28), dtype: np.uint8
    y_train = load_idx_labels(files['train_labels'])
    x_test = load_idx_images(files['test_images'])
    y_test = load_idx_labels(files['test_labels'])
    return (x_train, y_train), (x_test, y_test)


def batch_generator(x: np.ndarray, y: np.ndarray, batch_size: int):
    """Yield batches of data."""
    n_samples = x.shape[0]
    indices = np.arange(n_samples)
    # Shuffle indices to ensure random batches each epoch
    np.random.shuffle(indices)
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        yield x[batch_indices], y[batch_indices]

def relu(x: np.ndarray):
    return np.where(x > 0, x, 0)

def softmax(x: np.ndarray):
    """Compute softmax values for each set of scores in x (vectorized, numerically stable)."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)
