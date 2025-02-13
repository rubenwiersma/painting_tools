from typing import Optional, Union
import haiku as hk
import jax.numpy as jnp

from . import PaintLayer


def set_painting_params(params, weights=None, thickness=None):
    params = hk.data_structures.to_mutable_dict(params)
    for l, layer_params in enumerate(params.values()):
        if weights is not None:
            layer_params['w'] = weights[l]

        if thickness is not None:
            if type(thickness) is int or type(thickness) is float:
                t_l = jnp.ones(layer_params['t'].shape) * thickness
            else:
                t_l = thickness[l]
            layer_params['t'] = t_l
    return hk.data_structures.to_haiku_dict(params)


class Painting(hk.Module):
    """ Encodes a layered painting where each layer represents
    a wet-on-wet (alla prima) layer, which is then dried.
    The painting can be rendered to multiple modalities with the render_X methods,
    each of which is differentiable with jax autograd.

    Args:
        height (int): Height of the painting in pixels.
        width (int): Width of the painting in pixels.
        pigments (ndarray): Set of P pigment measurements with L wavelengths,
            which are used to render the painting.
            In an inverse rendering situation, the set of pigments is assumed
            to be the set of pigments used to paint the original painting.
            Size [P x L x 3]: pigments x wavelengths x [wavelength, absorption, scatter].
        n_layers (int, optional): Number of layers in the painting.
            Each layer is 'separated' by drying. (default: 1)
        pigment_layer_mask (ndarray): Masks the pigments available in each layer. (default: None)
        substrate (ndarray): Reflectance of the substrate (e.g., canvas). (default: None)
        name (str): Name of the painting, used for string representation.
    """
    def __init__(
        self,
        height: int,
        width: int,
        pigments: jnp.ndarray,
        n_layers: Optional[int] = 1,
        substrate: Optional[jnp.ndarray] = None,
        name: Optional[str] = None,
        opaque: bool = False
    ):
        super().__init__(name=name)
        self.height = height
        self.width = width
        self.pigments = pigments
        spectra = self.pigments[0][:, 0]
        self.n_spectra = spectra.shape[0]

        self.substrate = jnp.ones([height, width, self.n_spectra]) if substrate is None else substrate
        
        layers = []

        for l in range(n_layers):
            layer_pigments = pigments
            layers.append(PaintLayer(width, height, pigments=layer_pigments, name='PaintLayer_%d' % l, opaque=opaque))

        self.layers = tuple(layers)

    def spectral(self) -> jnp.ndarray:
        out = self.substrate
        
        for layer in self.layers:
            out = layer.spectral(out)

        return out

    def depth(self) -> jnp.ndarray:
        out = jnp.zeros((self.height, self.width))

        for layer in self.layers:
            out = out + layer.depth()
        
        return out

    def __repr__(self):
        return 'Painting {}'.format(self.name)
