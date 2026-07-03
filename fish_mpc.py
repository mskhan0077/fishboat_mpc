import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
import numpy as np
import casadi as ca
import math
from typing import Literal as L



class FishBoat:
    def __init__(self,
                 timesteps: int,
                 vehicle_l: float,
                 vehicle_w: float,
                 cam_range: float,
                 fov_deg: float,
                 v_max: float,
                 w_max: float,
                 map_dim: int,
                 safe_dist: float,
                 start_state: np.ndarray,
                 goal_state: np.ndarray,
                 obstacle_locs: np.ndarray,
                 control_horizon: int) -> None:
        
        self.timesteps = timesteps
        self.lh, self.lw = vehicle_l / 2.0, vehicle_w / 2.0   # vehicle half-dims
        self.cam_range = cam_range
        self.fov_half = np.deg2rad(fov_deg) / 2.0
        self.v_max, self.w_max = v_max, w_max
        self.map_dim = map_dim
        self.D_MIN = safe_dist
        self.W_SLACK = 1e4
 
        self.start_state = start_state.astype(float)
        self.goal_state = goal_state.astype(float)
        self.obstacle_locs = obstacle_locs.astype(float)   # rows: [x, y, w, h]
        self.control_horizon = control_horizon
        self.dt = 0.1
 
        self.nx, self.nu = 3, 2
        self.Q  = ca.diag([8, 8, 0.3])
        self.R  = ca.diag([0.1, 0.05])
        self.Qf = ca.diag([40, 40, 1])
 
        self.obs_corners = self._obstacle_corners()     # (M, 4, 2)
        self.known_idx = set()                           # obstacle memory
        self._cache_key = None                           # rebuild solver on change
        self._solver = None
        self._Xg = None                                  # warm-start state guess
        self._Ug = None                                  # warm-start control guess
 
        self.x = self.start_state.copy()
        self.fig, self.ax = plt.subplots(figsize=(7, 7))


    def collision_check(self):
        pass

    def _obstacle_corners(self) -> np.ndarray:
        """World-frame corners of every obstacle -> (M, 4, 2)."""
        M = self.obstacle_locs.shape[0]
        corners = np.zeros((M, 4, 2))
        for i, (cx, cy, w, h) in enumerate(self.obstacle_locs):
            hw, hh = w / 2.0, h / 2.0
            corners[i] = [[cx - hw, cy - hh], [cx + hw, cy - hh],
                          [cx + hw, cy + hh], [cx - hw, cy + hh]]
        return corners

    def dynamics_model(self, x, u) -> np.ndarray:
        return ca.vertcat(u[0] * ca.cos(x[2]),
                          u[0] * ca.sin(x[2]),
                          u[1])

    
    def v_corners(self, xk):
        c, th = xk[0:2], xk[2]
        Rm = ca.vertcat(ca.horzcat(ca.cos(th),-ca.sin(th)),
                        ca.horzcat(ca.sin(th), ca.cos(th)))
        return [c + ca.mtimes(Rm, ca.vertcat(sx, sy))
                for sx, sy in [(self.lh, self.lw), (self.lh, -self.lw),
                               (-self.lh, self.lw), (-self.lh, -self.lw)]]

    def build(self, active):
        N, dt = self.control_horizon, self.dt
        opti = ca.Opti()
        X = opti.variable(self.nx, N+1)
        U = opti.variable(self.nu, N)
        x0 = opti.parameter(self.nx)
        xref = opti.parameter(self.nx)
        M = len(active)
        A = opti.variable(2*M, N) if M else None
        B = opti.variable(M, N) if M else None
        S = opti.variable(M, N) if M else None

        cost = 0
        for k in range(N):
            e = X[:, k] - xref
            cost += ca.mtimes([e.T, self.Q, e]) + ca.mtimes([U[:, k].T, self.R, U[:, k]])
            f1 = self.dynamics_model(X[:, k],             U[:, k])
            f2 = self.dynamics_model(X[:, k] + dt/2 * f1, U[:, k])
            f3 = self.dynamics_model(X[:, k] + dt/2 * f2, U[:, k])
            f4 = self.dynamics_model(X[:, k] + dt   * f3, U[:, k])
            opti.subject_to(X[:, k+1] == X[:, k] + dt/6 * (f1 + 2*f2 + 2*f3 + f4))
        
            if M:
                vv = self.v_corners(X[:, k])
                for m in range(M):
                    a, b, s = A[2*m:2*m+2, k], B[m, k], S[m, k]
                    cost += self.W_SLACK*s**2
                    for v in vv:
                        opti.subject_to(ca.dot(a, v) <= b)
                    for w in active[m]:
                        opti.subject_to(ca.dot(a, ca.DM(w)) >= b + self.D_MIN - s)
                    opti.subject_to(ca.dot(a, a) <= 1)
                    opti.subject_to(s >= 0)
        
        cost += ca.mtimes([(X[:, N] - xref).T, self.Qf, (X[:, N] - xref)])
        opti.minimize(cost)
        opti.subject_to(X[:, 0] == x0)
        opti.subject_to(opti.bounded(-self.v_max, U[0, :], self.v_max))
        opti.subject_to(opti.bounded(-self.w_max, U[1, :], self.w_max))
        opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": 0,
                              "ipopt.sb": "yes", "ipopt.max_iter": 200})
        return dict(opti=opti, X=X, U=U, x0=x0, xref=xref, A=A, B=B, S=S, M=M)


    def mpc(self, fov_obs, subgoal):
        active = [self.obs_corners[i] for i in fov_obs]
        key = tuple(fov_obs)
        if key != self._cache_key:
            self.solver = self.build(active)
            self._cache_key = key
        s = self.solver
        opti = s["opti"]
        opti.set_value(s["x0"], self.x)
        opti.set_value(s["xref"], subgoal)
        if self._Xg is not None:
            opti.set_initial(s["X"], self._Xg)
            opti.set_initial(s["U"], self._Ug)
        try:
            sol = opti.solve()
            Xs, Us = sol.value(s["X"]), sol.value(s["U"])
        except RuntimeError:
            Xs = opti.debug.value(s["X"]); Us = opti.debug.value(s["U"])
        
        self._Xg = np.hstack([Xs[:, 1:], Xs[:, -1:]])
        self._Ug = np.hstack([Us[:, 1:], Us[:, -1:]])
        return Us[:, 0], Xs



    def update_fov(self):
        px, py, th = self.x
        for i, (cx, cy, *_) in enumerate(self.obstacle_locs):
            if i in self.known_idx:
                continue
            dx, dy = cx - px, cy - py
            if math.hypot(dx, dy) > self.cam_range:
                continue
            rel = np.arctan2(dy, dx) - th
            rel = (rel + np.pi) % (2*np.pi) - np.pi
            if abs(rel) <= self.fov_half:
                self.known_idx.add(i)

        return sorted(self.known_idx)
        
    def carrot(self):
        reach = self.control_horizon * self.dt * self.v_max
        to = self.goal_state[:2] - self.x[:2]
        d = np.linalg.norm(to)
        if d <= 0.9 * reach:
            return self.goal_state
        direction = to / d
        p = self.x[:2] + direction*0.9*reach
        return np.array([p[0], p[1], np.arctan2(direction[1], direction[0])])
    
    def rk4(self, x, u):
        f = lambda s: np.array([u[0] * np.cos(s[2]), u[0] * np.sin(s[2]), u[1]])
        k1 = f(x); k2 = f(x + self.dt/2 * k1)
        k3 = f(x + self.dt/2 * k2); k4 = f(x + self.dt * k3)
        return x + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    def _draw_vehicle(self):
        rect = patches.Rectangle((-self.lh, -self.lw), 2*self.lh, 2*self.lw,
                                 facecolor="tab:blue", edgecolor="navy", alpha=0.8)
        rect.set_transform(Affine2D().rotate(self.x[2])
                           .translate(self.x[0], self.x[1]) + self.ax.transData)
        self.ax.add_patch(rect)
 
    def render(self, traj, Xpred, subgoal, t):
        ax = self.ax
        ax.clear()
        ax.set_xlim(0, self.map_dim); ax.set_ylim(0, self.map_dim)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        for i, (cx, cy, w, h) in enumerate(self.obstacle_locs):
            known = i in self.known_idx
            ax.add_patch(patches.Rectangle(
                (cx - w/2, cy - h/2), w, h,
                facecolor="tab:red" if known else "lightgray",
                edgecolor="k", alpha=0.6 if known else 0.35))
        th_deg = np.degrees(self.x[2])
        ax.add_patch(patches.Wedge(self.x[:2], self.cam_range,
                                   th_deg - np.degrees(self.fov_half),
                                   th_deg + np.degrees(self.fov_half),
                                   color="tab:blue", alpha=0.12))
        if Xpred is not None:
            ax.plot(Xpred[0], Xpred[1], "c.-", ms=2, lw=1, label="MPC plan")
        ax.plot(traj[:, 0], traj[:, 1], "k-", lw=1.2, label="path")
        self._draw_vehicle()
        ax.plot(*self.goal_state[:2], "g*", ms=16, label="goal")
        ax.plot(*subgoal[:2], "m+", ms=11, mew=2, label="carrot")
        ax.set_title(f"step {t}   known obstacles: {len(self.known_idx)}")
        ax.legend(loc="upper left", fontsize=8)

    def run(self, live, gif_path):
        if live: 
            plt.ion()
        
        self.x = self.start_state.copy()
        traj = [self.x.copy()]
        frames = []

        for t in range(self.timesteps):
            fov_obs = self.update_fov()
            sub_goal = self.carrot()
            u, X_pred = self.mpc(fov_obs, sub_goal)
            self.x = self.rk4(self.x, u)
            traj.append(self.x.copy())
            self.render(np.array(traj), X_pred, sub_goal, t)
            if live:
                plt.pause(0.001)
            if gif_path:
                self.fig.canvas.draw()
                frames.append(np.asarray(self.fig.canvas.buffer_rgba()).copy())
            if np.linalg.norm(self.x[:2] - self.goal_state[:2]) < 0.6:
                break
        if live:
            plt.ioff(); plt.show()
        if gif_path:
            from PIL import Image
            imgs = [Image.fromarray(f) for f in frames]
            imgs[0].save(gif_path, save_all=True, append_images=imgs[1:],
                         duration=40, loop=0)
        return np.array(traj)


def main():
    total_timsteps = 500

    vehicle_l = 2.0
    vehicle_w = 1.0

    cam_range = 10.0
    fov_deg = 90.0

    v_max = 2.0
    w_max = 1.5

    map_dim = 50
    safe_dist = 2.0


    start_state = np.array([5.0, 5.0, np.pi/4])
    goal_state = np.array([47.0, 20.0, 0.0])
    obstacle_locs = np.array([[40.0, 20.0, 5.0, 10.0],
                                [22.0, 26.0, 4.0, 10.0],
                                [31.0, 28.0, 8.0, 4.0],
                                [38.0, 39.0, 5.0, 5.0]]) # define obstacles centers with width and height; [x, y, width, height]
    
    control_horizon = 10

    boat = FishBoat(timesteps=total_timsteps,
                    vehicle_l=vehicle_l,
                    vehicle_w=vehicle_w,
                    cam_range=cam_range,
                    fov_deg = fov_deg,
                    v_max=v_max,
                    w_max=w_max,
                    map_dim=map_dim,
                    safe_dist=safe_dist,
                    start_state=start_state,
                    goal_state=goal_state,
                    obstacle_locs=obstacle_locs,
                    control_horizon=control_horizon)
    
    boat.run(live=True, gif_path='boat_mpc.gif')




if __name__=='__main__':
    main()