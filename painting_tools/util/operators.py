from jax.lax import conv
import jax.numpy as jnp

def gradient(input):
    """ Computes the gradient for x and y directions and stacks them in the first (0th) axis.
    """
    return jnp.stack((gradient_x(input), gradient_y(input)), axis=0)


def gradient_x(input):
    """ Applies a gradient operator to an image to compute the gradient in the x direction.
    """
    gradient_x = jnp.array([
        [ 0., 0., 0.],
        [-1., 1., 0.],
        [ 0., 0., 0.]
    ])
    return convolve(gradient_x, input)


def gradient_y(input):
    """ Applies a gradient operator to an image to compute the gradient in the y direction.
    """
    gradient_y = jnp.array([
        [ 0.,-1., 0.],
        [ 0., 1., 0.],
        [ 0., 0., 0.]
    ])
    return convolve(gradient_y, input)


def convolve(filter, input, padding_mode='edge'):
    """ Convolves a K x K filter with an input image of size H x W.
    The input image can also have a channel dimension, but this is not required.
    """
    input_dim = len(input.shape)
    if input_dim < 3:
        input = input[..., jnp.newaxis]
    input = jnp.pad(input, pad_width=[[filter.shape[1] // 2]*2] * 2 + [[0, 0]], mode=padding_mode)
    input = jnp.transpose(input, [2, 0, 1])[jnp.newaxis]
    filter = filter[jnp.newaxis, jnp.newaxis] * jnp.eye(input.shape[1])[..., jnp.newaxis, jnp.newaxis]
    input_filter = conv(input, filter, window_strides=(1, 1), padding='VALID')
    output = jnp.transpose(input_filter[0], [1, 2, 0])
    if input_dim < 3:
        return output[..., 0]
    return output