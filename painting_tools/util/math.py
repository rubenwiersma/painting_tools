from jax import lax


def coth(x):
    return lax.cosh(x) / lax.sinh(x)