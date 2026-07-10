import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
OUTPUT_DIR = "plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_csv(csv_file, category):
    print(f"Processing {csv_file}")

    df = pd.read_csv(csv_file)

    # Find the time column
    time_col = None
    for c in df.columns:
        if "%" in c.lower() or "time" in c.lower():
            time_col = c
            break

    if time_col is None:
        time_col = df.columns[0]

    t = df[time_col]

    # Output folder
    out_dir = os.path.join(OUTPUT_DIR, category)
    os.makedirs(out_dir, exist_ok=True)

    # Plot every numeric column
    for col in df.columns:

        if col == time_col:
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        plt.figure(figsize=(12,4))
        plt.plot(t, df[col], linewidth=1)
        plt.title(col)
        plt.xlabel("Time")
        plt.ylabel(col)
        plt.grid(True)

        filename = os.path.splitext(os.path.basename(csv_file))[0]
        save_name = os.path.join(
            out_dir,
            f"{filename}_{col.replace('/','_').replace('[','').replace(']','')}.png"
        )

        plt.tight_layout()
        plt.savefig(save_name, dpi=200)
        plt.close()

    print(f"Saved plots to {out_dir}")


# -----------------------------
# Process IMU files
# -----------------------------
for f in glob.glob("imu*.csv"):
    plot_csv(f, "imu")

# -----------------------------
# Process Velocity files
# -----------------------------
for f in glob.glob("velocity*.csv"):
    plot_csv(f, "velocity")

# -----------------------------
# Process RC files
# -----------------------------
for f in glob.glob("rc*.csv"):
    plot_csv(f, "rc")

print("\nDone!")
print(f"Plots saved under '{OUTPUT_DIR}/'")