import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import casadi as ca
import math
from typing import Literal as L


ObstacleArray = np.ndarray[tuple[int, L[4]], np.dtype[np.float16]]
Pose = np.ndarray[tuple[L[3]], np.dtype[np.float16]]

class FishBoat:
    def __init__(self,
                 timesteps: int,
                 min_turn_radius: float,
                 wheelbase: float,
                 vehicle_l: float,
                 cam_range: float,
                 map_dim: int,
                 safe_dist: float,
                 start_state: Pose,
                 goal_state: Pose,
                 obstacle_locs: ObstacleArray,
                 control_horizon: int) -> None:
        
        self.timesteps = timesteps

        self.min_turn_radius = min_turn_radius
        self.wheelbase = wheelbase
        self.vehicle_l = vehicle_l
        self.cam_range = cam_range

        self.map_dim = map_dim
        self.safe_dist = safe_dist
        self.start_state = start_state
        self.goal_state = goal_state
        self.obstacle_locs = obstacle_locs

        self.control_horizon = control_horizon


        self.dt = 0.1
        self.state_dim = np.shape(start_state)
        self.fig, self.ax = plt.subplots()


    def collision_check(self):
        pass

    def reset(self):
        traj = np.zeros(shape=(self.timesteps, self.state_dim))
        
        obs_corners = np.zeros(shape=(np.shape(self.obstacle_locs)[0], 8), dtype=np.float16)
        for i, params in enumerate(self.obstacle_locs):
            blx, bly = params[0] - (params[2] / 2), params[1] - (params[3] / 2)
            ulx, uly = params[0] - (params[2] / 2), params[1] + (params[3] / 2)
            brx, bry = params[0] + (params[2] / 2), params[1] - (params[3] / 2)
            urx, ury = params[0] + (params[2] / 2), params[1] + (params[3] / 2)

            obs_corners[i] = [blx, bly, ulx, uly, brx, bry, urx, ury]
        
        return traj, obs_corners

    def mpc(self):
        pass

    def dynamics_model(self, x, u) -> Pose:
        return np.array([u[0] + math.cos(x[2]),
                         u[1] + math.sin(x[2]),
                         u[1]])

    def step(self):
        pass

    def render(self):           
        def make_obstacles():
            for center in self.obstacle_locs:
                left_x, left_y = center[0] - (center[2] / 2), center[1] - (center[3] / 2)
                rect = patches.Rectangle(xy=(left_x, left_y), 
                                         width=center[2], 
                                         height=center[3])
                self.ax.add_patch(rect)


def main():
    total_timsteps = 500

    min_turn_radius = 1.0
    wheelbase = 1.0
    vehicle_l = 2.0
    cam_range = 5.0
    v_max = 1.0
    w_max = 1.5

    map_dim = 50
    safe_dist = 0.5
    start_state = np.array([5.0, 5.0, 0.0])
    goal_state = np.array([45.0, 45.0, 0.0])
    obstacle_locs = np.array([[],
                              [],
                              []]) # define obstacles centers with width and height; [x, y, width, height]
    
    control_horizon = 30
    Q = ca.diag([8, 8, 0.3])
    R = ca.diag([0.1, 0.05])
    Qf = ca.diag([40, 40, 1])

    boat = FishBoat(timesteps=total_timsteps,
                    min_turn_radius=min_turn_radius,
                    wheelbase=wheelbase,
                    vehicle_l=vehicle_l,
                    cam_range=cam_range,
                    map_dim=map_dim,
                    safe_dist=safe_dist,
                    start_pos=start_state,
                    goal_pos=goal_state,
                    obstacle_locs=obstacle_locs,
                    control_horizon=control_horizon)




if __name__=='__main__':
    main()