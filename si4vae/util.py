import numpy as np
import tensorflow as tf

def make_gaussian_filter_kernel(kernel_size : int, sigma : float) -> tf.Tensor:
    """Make gaussian filter kernel for one channel image

    Parameters
    ----------
    kernel_size : int
        width and height of kernel
    sigma : float
        sigma of gaussian kernel

    Returns
    ----------
    kernle : np.ndarray
        gaussian filter kernel
    """
    x = np.linspace(-(kernel_size-1)/2, (kernel_size-1)/2, kernel_size)
    vx, vy = np.meshgrid(x,x)
    kernel = tf.exp(-0.5 * ((np.square(vx) + np.square(vy))/ np.square(sigma)))
    kernel = kernel / np.sum(kernel)
    kernel = tf.reshape(tf.cast(kernel, dtype=tf.float64),[kernel_size, kernel_size, 1, 1])

    return kernel

def make_mean_filter_kernel(kernel_size: int) -> np.ndarray:
    """Make mean filter kernel for one channel image

    Parameters
    ----------
    kernel_size : int
        width and height of kernel

    Returns
    ----------
    kernel : np.ndarray
        mean filter kernel
    """
    kernel = tf.ones([kernel_size, kernel_size, 1, 1]) / (kernel_size * kernel_size)
    kernel = tf.cast(kernel,dtype=tf.float64)

    return kernel

def apply_kernel(original_image, kernel, borderType="SAME"):
    """The function applying filtering with given kernel

    Parameters
    ----------
    original_image : np.ndarray
        Image applied kernel
    kernel : np.ndarray
        Kernel matrix used in filtering
    borderType : int
        Padding method

    Returns
    -------
    result_image : np.ndarray
    Image applied filtering
    """

    # apply kernel with replicate padding
    # TODO : 要検証
    # if borderType is constant int value, use use padding with specified value?
    result_image = tf.nn.depthwise_conv2d(original_image, kernel, strides=[1, 1, 1, 1], padding=borderType)

    return result_image
