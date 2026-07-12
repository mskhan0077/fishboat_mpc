import numpy as np
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array


def unicycle_ext(x, cmd, dt, tau_r, tau_u):
    px, py, heading, u, r = x
    uc, rc = cmd
    u = u + dt * (uc - u) / tau_u
    r = r + dt * (rc - r) / tau_r
    heading = heading + dt * r
    px = px + dt * u * jnp.cos(heading)
    py = py + dt * u * jnp.sin(heading)
    return jnp.array([px, py, heading, u, r])