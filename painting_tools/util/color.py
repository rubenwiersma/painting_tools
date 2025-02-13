import jax.numpy as jnp
from jax import lax
import numpy as np

from scipy.interpolate import interp1d

from ..rendering.kubelka_munk import kubelka_munk, mix_pigments
from ..measurements import cie1931raw, illuminantraw
from ..measurements.pigments import basic


xyztorgb = jnp.array([
                        [3.2404542, -1.5371385, -0.4985314],
                        [-0.9692660,  1.8760108,  0.0415560],
                        [0.0556434, -0.2040259,  1.0572252]
                     ])


def spectral_to_rgb(spectral_reflection, wavelengths=basic.spectra, rgb_response=cie1931raw, illuminant=illuminantraw, apply_gamma=True, gamma=2.4):
    """ Converts a multispectral reflectance to an RGB reflectance.

    Args:
        multispectral_reflection (ndarray): The reflectance per pixel
            for L wavelengths. Size [H, W x L].
        spectra (ndarray): The wavelengths used. Size [L]. (default: None)
        rgb_response (ndarray): The response for red, blue, and yellow
            for L* wavelengths (L* >= L). Size [L* x 4]. (default: cie1931raw)
        light (ndarray): The illuminant used in the simulated setup,
            defined for L* wavelengts (L* >= L). Size [L*]. (default: illuminantraw)
    """
    if (wavelengths[1] - wavelengths[0]) % 10 != 0:
        spectral_reflection, wavelengths = resample_wavelengths(spectral_reflection, wavelengths)
    if wavelengths is None:
        mask = jnp.arange(rgb_response.shape[0])
    else:
        mask = ((wavelengths - rgb_response[0, 0]) / (rgb_response[1, 0] - rgb_response[0, 0])).astype(jnp.int64)

    norm_factor = jnp.mean(illuminant[mask, 1:].T @ rgb_response[mask, 1:])
    spectral_reflection = spectral_reflection * illuminant[jnp.newaxis, jnp.newaxis, mask, 1] / norm_factor
    x = spectral_reflection @ rgb_response[mask, 1, jnp.newaxis]
    y = spectral_reflection @ rgb_response[mask, 2, jnp.newaxis]
    z = spectral_reflection @ rgb_response[mask, 3, jnp.newaxis]
    xyz = jnp.concatenate((x, y, z), axis=-1)
    rgb = xyz_to_rgb(xyz).clip(0, 1)
    if apply_gamma:
        return linear_to_gamma(rgb, gamma=gamma)
    return rgb


def resample_wavelengths(spectral_reflection, wavelengths, new_wavelengths=np.arange(410, 740, 10)):
    mask_wavelengths = wavelengths < 835
    wavelengths = wavelengths[mask_wavelengths]
    f_linear = interp1d(wavelengths, spectral_reflection[..., mask_wavelengths], 'linear')
    return f_linear(new_wavelengths), new_wavelengths


def xyz_to_rgb(xyz):
    H, W, _ = xyz.shape
    return (xyztorgb @ xyz.reshape(-1, 3).T).T.reshape(H, W, 3)


def rgb_to_luminance(rgb):
    rgb_coefficients = jnp.array([0.2126, 0.7152, 0.0722])
    rgb_linear = gamma_to_linear(rgb)
    return jnp.sum(rgb_linear * rgb_coefficients[None, None], axis=-1, keepdims=True) 


def linear_to_gamma(rgb_linear, threshold=0.0031308, a=0.055, gamma=2.4):
    return jnp.where(rgb_linear > threshold, (1 + a) * (rgb_linear ** (1.0/gamma)) - a, 12.92 * rgb_linear)


def gamma_to_linear(rgb_gamma, threshold=0.0031808, a=0.055, gamma=2.4):
    return jnp.where(rgb_gamma > 12.92 * threshold, ((rgb_gamma + a) / (1 + a)) ** gamma, rgb_gamma / 12.92)


def pigments_to_rgb(pigment_weights, pigments, t=None):
    t = jnp.ones(pigment_weights.shape[:-1]) if t is None else t                            # [W x H]
    pigment_spectra = jnp.stack([p['spectra'] for p in pigments], axis=0)                   # [P x L x 3]
    substrate = jnp.ones(pigment_weights.shape[:-1] + (pigments[0]['spectra'].shape[0],))   # [W x H x L]
    s_a = mix_pigments(pigment_weights, pigment_spectra[..., 1:])
    spectral = kubelka_munk(t, s_a[..., 0], s_a[..., 1], substrate)
    return spectral_to_rgb(spectral, pigments[0]['spectra'][:, 0])


def cmy_to_rgb(cmy, t=None):
    return pigments_to_rgb(cmy, basic.get_pigments('cmy'), t)
