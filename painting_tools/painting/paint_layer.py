from typing import Optional
import haiku as hk
import jax.numpy as jnp

from ..rendering.kubelka_munk import kubelka_munk, kubelka_munk_opaque, mix_pigments

class PaintLayer(hk.Module):
    def __init__(
        self,
        width: int,
        height: int,
        pigments: list,
        name: Optional[str] = None,
        opaque: bool = False
    ):
        super().__init__(name)
        self.width = width
        self.height = height
        self.n_pigments = len(pigments)
        self.pigments_spectra = jnp.stack(pigments, axis=0)
        self.opaque = opaque

    def pigment_weights(self) -> jnp.ndarray:
        P, H, W = self.n_pigments, self.height, self.width
        w_init = hk.initializers.RandomUniform(0.001, 1.)
        w = hk.get_parameter("w", shape=[H, W, P], init=w_init)
    
        # No pixel can have zero total pigment
        w_sum = w.sum(axis=-1, keepdims=True)
        w = w + (w_sum == 0) * 0.1

        # And every pixel's pigment ratios should sum to one
        w = w / w.sum(axis=-1, keepdims=True)

        return w

    def depth(self) -> jnp.ndarray:
        H, W = self.height, self.width
        t_init = hk.initializers.RandomUniform(0.1, 1.)
        return hk.get_parameter("t", shape=[H, W], init=t_init)

    def spectral(
        self,
        substrate: jnp.ndarray
    ) -> jnp.ndarray:
        w = self.pigment_weights() 
        t = self.depth()
        layer_pigments = mix_pigments(w, self.pigments_spectra[..., 1:])
        if self.opaque:
            return kubelka_munk_opaque(layer_pigments[..., 0], layer_pigments[..., 1])
        return kubelka_munk(t, layer_pigments[..., 0], layer_pigments[..., 1], substrate)
