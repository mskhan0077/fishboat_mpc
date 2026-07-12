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
                 hull_radius: float,
                 inflation: float,
                 cam_range: float,
                 fov_deg: float,
                 goal_tol: float,
                 r_obs_locs: ArrayLike) -> None:
        self.env_dim = env_dim
        self.dt = dt

        self.hull_radius = hull_radius
        self.inflation = inflation
        self.cam_range = cam_range
        self.fov_deg = fov_deg
        self.goal_tol = goal_tol

        self.r_obs = r_obs_locs

        self.x = jnp.linspace(start=0.0,
                              stop=env_dim,
                              num=env_dim)
        self.y = jnp.linspace(start=0.0,
                              stop=env_dim,
                              num=env_dim)
        self.xx, self.yy = jnp.meshgrid(self.x, self.y)
        self.grid_pts = jnp.stack(arrays=[self.xx.ravel(), self.yy.ravel()],
                                  axis=1)
        self.unknown_cost = 0.0
        self.costmap = jnp.zeros(shape=(self.env_dim, self.env_dim),
                                 dtype=int)
        


    def reset(self,
              start: ArrayLike,
              goal: ArrayLike) -> tuple[Array, Array, Array]:
        
        start_state = start
        goal_state = goal
        
        return start_state, goal_state


    def dynamics(self,
                 x: ArrayLike,
                 u: ArrayLike) -> Array:
        return jnp.array([u[0] * math.cos(x[2]),
                          u[0] * math.sin(x[2]),
                          u[1]], dtype=float)
    
    def obs_props(self):
        obs_centers = jnp.zeros(shape=(jnp.shape(self.r_obs)[0], 2),
                                dtype=float)
        obs_halfs = jnp.zeros(shape=(jnp.shape(self.r_obs)[0], 2),
                              dtype=float)
        obs_yaws = jnp.zeros(shape=(jnp.shape(self.r_obs)[0],),
                             dtype=float)
        
        for i in range(jnp.shape(self.r_obs)[0]):
            obs_centers = obs_centers.at[i].set(self.r_obs[i][:2])
            obs_halfs = obs_halfs.at[i].set(self.r_obs[i][2:4])
            obs_yaws = obs_yaws.at[i].set(self.r_obs[i][4])
        return obs_centers, obs_halfs, obs_yaws
    

    def box_sdf(self,
                points: ArrayLike,
                center: ArrayLike,
                halfs: ArrayLike,
                yaw: ArrayLike):
        c, s = jnp.cos(yaw), jnp.sin(yaw)
        Rt = jnp.array([[c, -s], [s, c]])
        p_local = (points - center) @ Rt
        d = jnp.abs(p_local) - halfs
        outside = jnp.linalg.norm(jnp.maximum(d, 0.0), axis=-1)
        inside = jnp.minimum(jnp.max(d, axis=-1), 0.0)
        return outside + inside
        
    
    def scene_sdf(self,
                  points: ArrayLike,
                  centers: ArrayLike,
                  halfs: ArrayLike,
                  yaws: ArrayLike):

        per_obs = jax.vmap(fun=self.box_sdf,
                           in_axes=(None, 0, 0, 0))(points, centers, halfs, yaws)
        return jnp.min(per_obs, axis=0)
    
    def sdf_to_cost(self,
                    sdf: ArrayLike,
                    hull_radius: float,
                    inflation: float,
                    lethal=1e3):
        clearance = jnp.maximum(sdf - hull_radius, 0.0)
        graded = jnp.exp(-clearance / inflation)
        return jnp.where(sdf < hull_radius, lethal, graded)
    

    def in_fov(self,
               points: ArrayLike,
               current_state: ArrayLike,
               cam_range: float,
               fov_deg: float):
        rel = points - current_state[:2]
        rng = jnp.linalg.norm(rel, axis=-1)
        bearing=jnp.arctan2(rel[:, 1], rel[:, 0])
        err = jnp.arctan2(jnp.sin(bearing - current_state[2]),
                        jnp.cos(bearing - current_state[2]))
        return (rng <= cam_range) & (jnp.abs(err) <= fov_deg)
        
    
    def build_costmap(self, current_state):
        centers, halfs, yaws = self.obs_props()
        sdf = self.scene_sdf(self.grid_pts,
                             centers,
                             halfs,
                             yaws)
        obs_cost = self.sdf_to_cost(sdf,
                                    self.hull_radius,
                                    self.inflation)
        known = self.in_fov(self.grid_pts,
                            current_state,
                            self.cam_range,
                            self.fov_deg)
        return jnp.where(known, obs_cost, self.unknown_cost)
    

    def plot_costmap(self,
                     cost: ArrayLike):
        cost_img = np.asarray(cost).reshape(self.xx.shape)
        fig, ax = plt.subplots(figsize=(11, 7))
        im = ax.imshow(cost_img, origin="lower",
                       extent=[float(self.x[0]), float(self.x[-1]), float(self.y[0]), float(self.y[-1])],
                       cmap="magma", aspect="equal", vmax=3.0)
        fig.colorbar(im, ax=ax, label="cost (clipped at 3 for display)")
        plt.show()
    