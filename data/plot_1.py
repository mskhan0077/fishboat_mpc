import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "plots_1"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_time(df):
    """Find ROS time column."""
    for c in df.columns:
        if "time" in c.lower() or "%" in c.lower():
            t = df[c] * 1e-9      # nanoseconds -> seconds
            return t - t.iloc[0]
    return np.arange(len(df))


def find_column(df, keywords):
    """Find column containing all keywords."""
    for c in df.columns:
        name = c.lower()
        if all(k in name for k in keywords):
            return c
    return None


def plot_xyz(df, t, prefix, ylabel, savefile):
    x = find_column(df, [prefix, "x"])
    y = find_column(df, [prefix, "y"])
    z = find_column(df, [prefix, "z"])

    if x is None:
        return

    plt.figure(figsize=(12,5))
    plt.plot(t, df[x], label="X", linewidth=1.5)
    if y:
        plt.plot(t, df[y], label="Y", linewidth=1.5)
    if z:
        plt.plot(t, df[z], label="Z", linewidth=1.5)

    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(savefile.replace("_", " ").title())
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savefile, dpi=250)
    plt.close()


def plot_magnitude(df, t, prefix, ylabel, savefile):
    x = find_column(df, [prefix, "x"])
    y = find_column(df, [prefix, "y"])
    z = find_column(df, [prefix, "z"])

    if x is None:
        return

    mag = np.sqrt(df[x]**2 + df[y]**2 + df[z]**2)

    plt.figure(figsize=(12,4))
    plt.plot(t, mag, linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title("Magnitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(savefile, dpi=250)
    plt.close()


# -------------------------------------------------------
# IMU
# -------------------------------------------------------
for file in glob.glob("imu*.csv"):

    df = pd.read_csv(file)
    t = find_time(df)

    folder = os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(file))[0])
    os.makedirs(folder, exist_ok=True)

    plot_xyz(
        df,
        t,
        "linear_acceleration",
        "Acceleration (m/s²)",
        os.path.join(folder, "linear_acceleration.png")
    )

    plot_xyz(
        df,
        t,
        "angular_velocity",
        "Angular Velocity (rad/s)",
        os.path.join(folder, "angular_velocity.png")
    )

    plot_magnitude(
        df,
        t,
        "linear_acceleration",
        "Acceleration Magnitude (m/s²)",
        os.path.join(folder, "acceleration_magnitude.png")
    )

    plot_magnitude(
        df,
        t,
        "angular_velocity",
        "Angular Velocity Magnitude (rad/s)",
        os.path.join(folder, "angular_velocity_magnitude.png")
    )


# -------------------------------------------------------
# Velocity
# -------------------------------------------------------
for file in glob.glob("velocity*.csv"):

    df = pd.read_csv(file)
    t = find_time(df)

    folder = os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(file))[0])
    os.makedirs(folder, exist_ok=True)

    plot_xyz(
        df,
        t,
        "linear",
        "Velocity (m/s)",
        os.path.join(folder, "linear_velocity.png")
    )

    plot_magnitude(
        df,
        t,
        "linear",
        "Speed (m/s)",
        os.path.join(folder, "speed.png")
    )


# -------------------------------------------------------
# RC Channels
# -------------------------------------------------------
for file in glob.glob("rc*.csv"):

    df = pd.read_csv(file)
    t = find_time(df)

    folder = os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(file))[0])
    os.makedirs(folder, exist_ok=True)

    channel_cols = [c for c in df.columns if "channel" in c.lower()]

    if len(channel_cols) > 0:

        plt.figure(figsize=(12,6))

        for c in channel_cols:
            plt.plot(t, df[c], label=c.split(".")[-1])

        plt.xlabel("Time (s)")
        plt.ylabel("PWM")
        plt.title("RC Channels")
        plt.grid(True)
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(folder, "rc_channels.png"), dpi=250)
        plt.close()

print("Plots saved to:", OUTPUT_DIR)