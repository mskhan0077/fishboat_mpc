import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from jax.scipy.ndimage import map_coordinates
from env import Env
from dynamics import unicycle_ext


class P:
    dt = 0.2
    T = 30
    K = 2000
    temp = 1.0
    tau_u = 0.8
    tau_r = 0.5
    u_min, u_max = 0.5, 2.0
    r_min, r_max = -0.9, 0.9
    sig_u = 1.2
    sig_r = 2.0
    w_goal = 4.0
    w_term = 30.0
    w_obs = 120.0
    w_ctrl = 0.02
    w_rate = 0.05
    max_range = 9.0
    half_fov = jnp.deg2rad(45)
    hull_r = 0.4
    inflation = 0.45
    unknown_cost = 0.1


# ---- pure rollout (uses module constants, no env object -> jittable) ----
def _rollout(x0, V):
    def f(x, cmd):
        nx = unicycle_ext(x, cmd, P.dt, P.tau_r, P.tau_u)   # FIX: dt supplied
        return nx, nx[:2]
    _, pos = jax.lax.scan(f, x0, V)
    return pos


@partial(jax.jit, static_argnames=("x0g", "y0g", "dxg", "dyg"))
def _mppi_step(x0, U_bar, G, key, goal, x0g, y0g, dxg, dyg):
    key, sk = jax.random.split(key)
    sig = jnp.array([P.sig_u, P.sig_r])
    eps = jax.random.normal(sk, (P.K, P.T, 2)) * sig
    dU = jnp.cumsum(eps * P.dt, axis=1)
    V = U_bar[None] + dU
    V = V.at[:, :, 0].set(jnp.clip(V[:, :, 0], P.u_min, P.u_max))
    V = V.at[:, :, 1].set(jnp.clip(V[:, :, 1], P.r_min, P.r_max))   # FIX: was clipping ch0 twice; now ch1 with r limits

    # FIX: vmap over 3 conceptual args reduced to 2 (env removed)
    pos = jax.vmap(_rollout, in_axes=(None, 0))(x0, V)

    # costmap query (world -> fractional grid index -> bilinear sample)
    cols = (pos[:, :, 0] - x0g) / dxg
    rows = (pos[:, :, 1] - y0g) / dyg
    coords = jnp.stack([rows.ravel(), cols.ravel()])
    c_obs = map_coordinates(G, coords, order=1, mode="constant",
                            cval=P.unknown_cost).reshape(pos.shape[:2])

    d_goal = jnp.linalg.norm(pos - goal[:2], axis=-1)
    stage = P.w_obs * c_obs + P.w_goal * d_goal
    ctrl = P.w_ctrl * jnp.sum(V**2, axis=-1)
    rate = P.w_rate * jnp.sum(eps**2, axis=-1)
    S = jnp.sum(stage + ctrl + rate, axis=1) + P.w_term * d_goal[:, -1]

    w = jax.nn.softmax(-(S - jnp.min(S)) / P.temp)       
    U_new = jnp.sum(w[:, None, None] * V, axis=0)

    best = jnp.argmin(S)
    brake = jnp.max(c_obs[best]) > 500.0
    U_new = jnp.where(brake, jnp.zeros_like(U_new), U_new)
    return U_new, key, brake


class MPPI:
    def __init__(self, env, goal):
        self.env = env
        self.goal = goal
        self.x0g = float(env.x[0]); self.y0g = float(env.y[0])
        self.dxg = float(env.x[1] - env.x[0]); self.dyg = float(env.y[1] - env.y[0])

    def reveal(self, G, pose, true_cost):
        known = self.env.in_fov(pose, P.max_range, P.half_fov).reshape(self.env.xx.shape)
        return jnp.where(known, true_cost, G)

    def mppi_step(self, x0, U_bar, G, key):
        return _mppi_step(x0, U_bar, G, key, self.goal,
                          self.x0g, self.y0g, self.dxg, self.dyg)

    def run(self, start_pos, start_vel, true_cost, max_steps=250):
        x = jnp.concatenate([start_pos, start_vel], axis=0)
        U_bar = jnp.zeros((P.T, 2))
        G = jnp.full(self.env.xx.shape, P.unknown_cost)
        key = jax.random.PRNGKey(42)

        traj = [np.array(x)]; cmds = []; brakes = []; Gs = []
        for step in range(max_steps):           
            G = self.reveal(G, x[:3], true_cost)
            Gs.append(np.array(G))
            U_bar, key, brake = self.mppi_step(x, U_bar, G, key)
            u_apply = U_bar[0]
            x = unicycle_ext(x, u_apply, P.dt, P.tau_r, P.tau_u)
            U_bar = jnp.concatenate([U_bar[1:], U_bar[-1:]])
            traj.append(np.array(x)); cmds.append(np.array(u_apply)); brakes.append(bool(brake))
            if float(jnp.linalg.norm(x[:2] - self.goal[:2])) < 0.8:
                print(f"reached goal in {step+1} steps ({(step+1)*P.dt:.1f}s)")
                break
        return np.array(traj), np.array(cmds), Gs


def main():
    env_dim = 50
    start_pos = jnp.array([5.0, 5.0, 0.0])
    start_vel = jnp.zeros((2,))
    goal = jnp.array([45.0, 45.0, 0.0])          # FIX: was (40,40) which is INSIDE obstacle 4

    r_obs_locs = jnp.array([[10.0, 20.0, 2.5, 5.0, 0.0],
                            [22.0, 10.0, 4.0, 6.0, 0.0],
                            [31.0, 28.0, 8.0, 2.0, 0.0],
                            [38.0, 45.0, 2.5, 2.5, 0.0]])

    env = Env(env_dim=env_dim, dt=P.dt, hull_radius=P.hull_r, inflation=P.inflation,
              cam_range=P.max_range, fov_deg=P.half_fov, r_obs_locs=r_obs_locs)

    centers, halfs, yaws = env.obs_props()
    sdf = env.scene_sdf(centers, halfs, yaws).reshape(env.xx.shape)
    true_cost = env.sdf_to_cost(sdf, P.hull_r, P.inflation)

    controller = MPPI(env, goal)
    traj, cmds, Gs = controller.run(start_pos, start_vel, true_cost)
    print("path length:", round(float(np.sum(np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1))), 2), "m")

    # ---------------- animation ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Wedge
    from matplotlib.animation import FuncAnimation, PillowWriter

    xe = [float(env.x[0]), float(env.x[-1]), float(env.y[0]), float(env.y[-1])]
    hf = float(P.half_fov)

    def corners(c, h, yaw):
        cc, ss = np.cos(yaw), np.sin(yaw)
        R = np.array([[cc, -ss], [ss, cc]])
        loc = np.array([[-h[0], -h[1]], [h[0], -h[1]], [h[0], h[1]], [-h[0], h[1]]])
        return loc @ R.T + c

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    im = ax.imshow(Gs[0], origin="lower", extent=xe, cmap="Greys",
                   vmax=1.2, alpha=0.6, zorder=0)
    for loc in np.array(r_obs_locs):
        ax.add_patch(Polygon(corners(loc[:2], loc[2:4], loc[4]), closed=True,
                     facecolor="crimson", edgecolor="darkred", alpha=0.85, zorder=2))
    ax.plot(float(start_pos[0]), float(start_pos[1]), "o", color="limegreen", ms=12, zorder=5, label="start")
    ax.plot(float(goal[0]), float(goal[1]), "*", color="gold", ms=22, mec="k", zorder=5, label="goal")
    traj_line, = ax.plot([], [], color="dodgerblue", lw=2.5, zorder=4)
    boat_pt, = ax.plot([], [], "o", color="black", ms=8, zorder=6)
    ax.set_xlim(0, 50); ax.set_ylim(0, 50); ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.legend(loc="lower right")
    holder = {"p": []}

    def update(i):
        for p in holder["p"]:
            p.remove()
        holder["p"] = []
        im.set_data(np.asarray(Gs[i]))
        px, py, psi = traj[i, 0], traj[i, 1], traj[i, 2]
        traj_line.set_data(traj[:i + 1, 0], traj[:i + 1, 1])
        boat_pt.set_data([px], [py])
        wdg = Wedge((px, py), P.max_range, np.rad2deg(psi - hf), np.rad2deg(psi + hf),
                    alpha=0.18, color="green", zorder=1)
        ax.add_patch(wdg); holder["p"].append(wdg)
        arr = ax.arrow(px, py, 2.2 * np.cos(psi), 2.2 * np.sin(psi),
                       head_width=1.0, color="black", zorder=6)
        holder["p"].append(arr)
        ax.set_title(f"MPPI  |  t = {i * P.dt:4.1f} s")
        return []

    frames = range(0, len(Gs), 2)       # every 2nd step keeps the gif light
    anim = FuncAnimation(fig, update, frames=frames, interval=90, blit=False)
    anim.save("mppi_anim.gif", writer=PillowWriter(fps=12))
    print("animation frames:", len(list(frames)))


if __name__ == "__main__":
    main()