import jax.numpy as jnp
from jax import lax
from ..util.math import coth

EPS = 1e-5

def kubelka_munk(t, absorption, scatter, substrate_reflectance, eps=EPS):
    """ Uses Kubelka-Munk to model the reflectance of a layer of paint
    applied on top of a substrate. The layer of paint is given by the
    absorption and scattering components of the paint at each pixel.

    Assumptions:
    - The reflectance is captured at a 90 degree angle with a completely diffuse light
      or the incoming light hits the surface at a 60 degree angle.
    - The pigments are all contained in the same medium and there is no other medium between
      the surface and the sensor. If there is air, use the Saunderson equations
      to counteract the effect of air (see Zhao, Berns, Taplin et al. 2008).
    
    Args:
        t (ndarray): Thickness of the layer applied on top. Size [H x W].
        absorption (ndarray): Absorption of pigment at each pixel for 
            L wavelengths. Size [H x W x L].
        scatter (ndarray): Scatter of pigment at each pixel for 
            L wavelengths. Size [H x W x L].
        substrate_reflectance (ndarray): The reflectance of the substrate
            at each pixel for L wavelengths. Size [H x W x L]
    """
    a, s = absorption, scatter
    xi = substrate_reflectance
    a_over_s = a / s.clip(eps)
    x = (1.0 + a_over_s).clip(1.)
    y = lax.sqrt(lax.pow(x, 2.) - 1.)
    coth_yst = coth(y * s * jnp.expand_dims(t, -1) + eps)
    y_coth_yst = y * coth_yst
    return (1.0 - xi * (x - y_coth_yst)) / (x - xi + y_coth_yst + eps)

def kubelka_munk_opaque(absorption, scatter, eps=EPS):
    """ Uses Kubelka-Munk to model the reflectance of an opaque layer of paint.
    KM is based on the following assumptions:
    - The layer of paint is opaque.
    - The scattering component is dominated by white (i.e., all 1).
    - The standard KM assumptions also hold.
    
    Args:
        t (ndarray): Thickness of the layer applied on top. Size [H x W].
        absorption (ndarray): Absorption of pigment at each pixel for 
            L wavelengths. Size [H x W x L].
        scatter (ndarray): Scatter of pigment at each pixel for 
            L wavelengths. Size [H x W x L].
        substrate_reflectance (ndarray): The reflectance of the substrate
            at each pixel for L wavelengths. Size [H x W x L]
    """
    a_over_s = absorption / scatter.clip(eps)
    return 1 + a_over_s - jnp.sqrt(jnp.square(a_over_s) + 2 * a_over_s)

def saunderson_correction(r, k1, k2):
    return k1 + (1 - k1) * (1 - k2) * (r / (1 - k2 * r))

def reflectance_to_ks(reflection):
    return jnp.square(1 - reflection) / (2 * reflection)

def mix_pigments(weights, measurements, eps=EPS):
    """ Mixes a set of pigments linearly, given a set of
    P weights and P pigment measurements.
    
    Args:
        weights (ndarray): The weight for each pigment. Size [H x W x P].
        measurements (ndarray): The measurements for each pigment,
            given by absorption and scatter for L wavelengths. Size [P x L x 2]
    """
    weights = weights[..., None, None]                                    # [H x W x P x 1 x 1]
    # weights_norm = weights.sum(axis=2, keepdims=True)
    # weights = jnp.where(weights_norm == 0, jnp.ones_like(weights) / weights.shape[2], weights / weights_norm)
    mixed_measurements = (weights * measurements[None, None]).sum(axis=2) # [H x W x L x 2]

    return mixed_measurements