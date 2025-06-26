import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import matplotlib

# === 设置字体和字号 ===
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 24
plt.rcParams['axes.labelsize'] = 26
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams['legend.fontsize'] = 24

def parse_args():
    parser = argparse.ArgumentParser(description='Plot lane vehicle trajectories with S-G smoothing and strict interpolation.')
    parser.add_argument('--trajectory_file', type=str, required=True, help='Path to the trajectory data CSV file')
    parser.add_argument('--steps', type=int, default=None, help='Number of steps to plot (default: all steps)')
    args = parser.parse_args()
    return args

def load_data(trajectory_file, max_steps=None):
    try:
        df = pd.read_csv(trajectory_file)
        required_cols = ['step','vehid','vtype','lane_id','distance_to_intersection','speed','signal_phase']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        if max_steps is not None:
            df = df[df['step'] <= max_steps]
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def savgol_smooth_distance(veh_df, window_length=20, polyorder=2):
    if len(veh_df) < window_length:
        return veh_df
    veh_df = veh_df.sort_values('step').reset_index(drop=True)
    dist_array = veh_df['distance_to_intersection'].values
    dist_smooth = savgol_filter(dist_array, window_length=window_length, polyorder=polyorder)
    dist_smooth = np.clip(dist_smooth, 0, None)
    veh_df['distance_to_intersection'] = dist_smooth
    return veh_df

def strict_interpolate_if_missing_next(veh_df):
    if veh_df.empty:
        return veh_df

    veh_df = veh_df.sort_values('step').reset_index(drop=True)
    new_rows = []
    n = len(veh_df)
    for i in range(n):
        row_i = veh_df.iloc[i]
        dist_i = row_i['distance_to_intersection']
        speed_i = row_i['speed']
        step_i = row_i['step']

        if i < n-1:
            new_rows.append(row_i)
        else:
            new_rows.append(row_i)
            dist_next = dist_i - speed_i
            if dist_next < 5 and speed_i > 1e-9:
                t_cross = step_i + dist_i / speed_i
                interpolated_row = row_i.copy()
                interpolated_row['step'] = t_cross
                interpolated_row['distance_to_intersection'] = 0.0
                interpolated_row['speed'] = speed_i
                new_rows.append(interpolated_row)
            break

    out_df = pd.DataFrame(new_rows).reset_index(drop=True)
    return out_df

def plot_line(ax, veh_df, line_color='blue'):
    if len(veh_df) < 2:
        return
    veh_df = veh_df.sort_values('step').reset_index(drop=True)

    segments = []
    current_seg = [veh_df.iloc[0]]
    for i in range(1, len(veh_df)):
        prev_step = veh_df['step'].iloc[i - 1]
        curr_step = veh_df['step'].iloc[i]
        if (curr_step - prev_step) > 5:
            segments.append(pd.DataFrame(current_seg))
            current_seg = [veh_df.iloc[i]]
        else:
            current_seg.append(veh_df.iloc[i])
    if current_seg:
        segments.append(pd.DataFrame(current_seg))

    for seg in segments:
        ax.plot(seg['step'], seg['distance_to_intersection'],
                color=line_color, linewidth=2.0, alpha=0.9)

def get_phase_color(phase):
    phase = str(phase).lower()
    if '0.0' in phase:
        return 'red'
    elif '1.0' in phase:
        return 'green'
    elif '0.5' in phase:
        return 'yellow'
    return 'gray'

def plot_trajectories_lane(lane_data, output_path, lane_id):
    fig, ax = plt.subplots(figsize=(12, 8))

    vtype0_data = lane_data[lane_data['vtype'] == 'vtype0']
    vtype1_data = lane_data[lane_data['vtype'] == 'vtype1']

    for vehid, veh_df in vtype0_data.groupby('vehid'):
        veh_df = strict_interpolate_if_missing_next(veh_df)
        plot_line(ax, veh_df, line_color='blue')

    for vehid, veh_df in vtype1_data.groupby('vehid'):
        veh_df = strict_interpolate_if_missing_next(veh_df)
        plot_line(ax, veh_df, line_color='red')

    phase_data = lane_data[['step', 'signal_phase']].drop_duplicates().sort_values('step')
    phase_y = np.zeros_like(phase_data['step'])
    colors = [get_phase_color(ph) for ph in phase_data['signal_phase']]
    ax.scatter(phase_data['step'], phase_y, c=colors, marker='|', s=150)

    import matplotlib.lines as mlines
    cav_line = mlines.Line2D([], [], color='red', linewidth=2.0, label='CAV')
    hv_line = mlines.Line2D([], [], color='blue', linewidth=2.0, label='HV')
    #ax.legend(handles=[cav_line, hv_line], loc='lower right')
    ax.legend(handles=[hv_line], loc='lower right')


    ax.set_xlabel("Time Step")
    ax.set_ylabel("Distance to Intersection")
    ax.grid(True)
    ax.set_ylim(250, 0)

    fig.tight_layout()
    plt.savefig(output_path, format='pdf')  # <-- 输出为 PDF
    plt.close(fig)
    print(f"Saved to {output_path}")

def main():
    args = parse_args()
    df = load_data(args.trajectory_file, args.steps)
    if df is None or df.empty:
        print("No data.")
        return

    lane_ids = df['lane_id'].unique()
    for lane_id in lane_ids:
        lane_data = df[df['lane_id'] == lane_id]
        if lane_data.empty:
            continue
        safe_lane_id = lane_id.replace('/', '_').replace('\\', '_')
        out_path = os.path.join(os.path.dirname(args.trajectory_file),
                                f"lane_{safe_lane_id}_trajectory.pdf")  # <-- 保存为 PDF
        plot_trajectories_lane(lane_data, out_path, lane_id)

if __name__ == "__main__":
    main()
