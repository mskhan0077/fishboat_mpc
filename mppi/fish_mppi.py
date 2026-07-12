import numpy as np
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array
import math
from env import Env


class MPPI:
    def __init__(self):
        pass




def main():
    env_dim = 50
    dt = 0.1

    hull_radius = 0.5
    inflation = 1.2
    cam_range = 10.0
    half_fov_deg = jnp.deg2rad(45.0)

    start = jnp.array(object=[5.0, 5.0, 0.0],
                      dtype=float)
    goal = jnp.array(object=[40.0, 40.0, 0.0],
                     dtype=float)
    
    r_obs_locs = jnp.array(object=[[10.0, 20.0, 2.5, 5.0, 0.0],
                                    [22.0, 26.0, 2.0, 5.0, 0.0],
                                    [31.0, 28.0, 2.0, 2.0, 0.0],
                                    [38.0, 39.0, 2.5, 2.5, 0.0]], dtype=float) # (x, y, width/2, height/2, yaw), rectangular obstacles
    
    goal_tol = 0.2

    env = Env(env_dim=env_dim,
             dt=dt,
             hull_radius=hull_radius,
             inflation=inflation,
             cam_range=cam_range,
             fov_deg=half_fov_deg,
             goal_tol=goal_tol,
             r_obs_locs=r_obs_locs)
    
    
    start_state, goal_state = env.reset(start=start,
                                        goal=goal)
    
    current_state = start_state
    initial_cost = env.build_costmap(current_state=current_state)

    while jnp.linalg.norm(current_state[:2], goal_state[:2]) > goal_tol:


if __name__=='__main__':
    main()