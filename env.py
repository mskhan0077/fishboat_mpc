import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.typing import ArrayLike
from jax import Array
import math


class Env:
    def __init__(self,
                 env_dim: int,
                 dt: float,
                 vehicle_l: float,
                 vehicle_w: float,
                 goal_tol: float) -> None:
        self.env_dim = env_dim
        self.dt = dt
        self.lh = vehicle_l / 2.0
        self.wh = vehicle_w / 2.0
        self.goal_tol = goal_tol


    def reset(self,
              start: ArrayLike,
              goal: ArrayLike) -> tuple[Array, Array, Array]:
        veh_corners = jnp.array(object=[[start[0] - self.lh, start[1] - self.wh],
                                        [start[0] + self.lh, start[1] - self.wh],
                                        [start[0] + self.lh, start[1] + self.wh],
                                        [start[0] - self.lh, start[1] - self.wh]],
                                dtype=float)
        start_state = start
        goal_state = goal
        
        return start_state, goal_state, veh_corners


    def dynamics(self,
                 x: ArrayLike,
                 u: ArrayLike) -> Array:
        return jnp.array([u[0] * math.cos(x[2]),
                          u[0] * math.sin(x[2]),
                          u[1]], dtype=float)


def main():
    env_dim = 50
    dt = 0.1
    vehicle_l = 2.0
    vehicle_w = 1.0

    start = jnp.array(object=[5.0, 5.0, 0.0],
                      dtype=float)
    goal = jnp.array(object=[40.0, 40.0, 0.0],
                     dtype=float)
    # obstacles = jnp.array(object=[]) # include as per need
    goal_tol = 0.2

    env = Env(env_dim=env_dim,
             dt=dt,
             vehicle_l=vehicle_l,
             vehicle_w=vehicle_w,
             goal_tol=goal_tol)
    start_state, goal_state, veh_corners = env.reset(start=start,
                                                     goal=goal)


if __name__=='__main__':
    main()