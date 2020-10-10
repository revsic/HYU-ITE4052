import numpy as np
from scipy import signal


def gradient(img):
    """Compute gradient from image with convolution of
    vertical, horizontal edge filters.
    
    Args:
        img: np.ndarray, [H, W], gray scale image.
    Returns:
        np.ndarray, [H, W], gradient computed by pixel-by-pixel.
    """
    dx = signal.convolve2d(img, [[-1, 1]])[..., :-1]
    dy = signal.convolve2d(img, [[-1], [1]])[:-1]
    return dx, dy


def harris_matrix(img, size):
    """Compute harris matrix from image with given window size.
    Args:
        img: np.ndarray, [H, W], gray scale image.
        size: int, size of the window.
    Returns:
        np.ndarray, [H - size + 1, W - size + 1, 2, 2], harris matrix.
    """
    # [H, W]
    dx, dy = gradient(img)
    # for summing gradients in window
    boxfilter = np.ones((size, size))
    # [H - size + 1, W - size + 1]
    ixx = signal.convolve2d(dx * dx, boxfilter, 'valid')
    ixy = signal.convolve2d(dx * dy, boxfilter, 'valid')
    iyy = signal.convolve2d(dy * dy, boxfilter, 'valid')
    # [2, 2, H - size + 1, W - size + 1]
    harris = np.array([[ixx, ixy], [ixy, iyy]])
    # [H - size + 1, W - size + 1, 2, 2]
    return harris.transpose(2, 3, 0, 1)


def harris_resp(matrix, eps=1e-8):
    """Compute harris response from matrix.
    Args:
        matrix: np.ndarray, [H', W', 2, 2], harris matrices.
        eps: float, small value for preventing zero-division.
    Returns:
        tuple,
            eigval: np.ndarray, [H', W', 2], eigen values.
            eigvec: np.ndarray, [H', W', 2, 2], normalized eigen vectors.
            resp: np.ndarray, [H', W'], harris response.
    """
    # [H', W', 2], [H', W', 2, 2]
    eigval, eigvec = np.linalg.eig(matrix)
    resp = eigval.prod(axis=-1) / np.maximum(eigval.sum(axis=-1), eps)
    return eigval, eigvec, resp
