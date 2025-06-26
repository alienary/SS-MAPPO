import numpy as np
import matplotlib.pyplot as plt

# 设置统一字体样式
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
# 渗透率标签与对应位置（从0%开始）
penetration_rates = ["0%", "20%", "40%", "60%", "100%"]
x = np.arange(len(penetration_rates))

# 方法标签和颜色
methods = ["MAPPO", "MAA2C", "MADDPG"]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# 指标标签
y_labels = ["Waiting Time (s)", "Delay (s)", "Fuel Consumption (ml)", "Stop And Go"]

# IDM数据作为起点
idm_values = {
    "Waiting Time (s)": 60.9,
    "Delay (s)": 161.79,
    "Fuel Consumption (ml)": 103.8,
    "Stop And Go": 1.94
}

# 三种方法的数据（不含0%，我们手动在前面加IDM值）
raw_data = {
    "MAPPO": {
        "Waiting Time (s)": [33.65, 20.71, 16.95, 9.98],
        "Delay (s)": [138.83, 128.89, 121.83, 109.77],
        "Fuel Consumption (ml)": [89.33, 83.93, 81.73, 75.91],
        "Stop And Go": [1.02, 0.87, 0.83, 0.81]
    },
    "MAA2C": {
        "Waiting Time (s)": [39.83, 23.23, 17.64, 9.97],
        "Delay (s)": [143.85, 143.42, 132.52, 122.15],
        "Fuel Consumption (ml)": [95.48, 90.86, 86.58, 82.36],
        "Stop And Go": [1.21, 1.03, 0.89, 0.86]
    },
    "MADDPG": {
        "Waiting Time (s)": [35.77, 23.05, 17.9, 11.22],
        "Delay (s)": [147.54, 142.43, 141.68, 134.4],
        "Fuel Consumption (ml)": [92.28, 90.66, 90.17, 87.37],
        "Stop And Go": [1.12, 1.03, 1.01, 0.90]
    }
}
label_map = {
    "MAPPO": "SS-MAPPO",
    "MAA2C": "SS-MAA2C",
    "MADDPG": "SS-MADDPG"
}

for i, metric in enumerate(y_labels):
    fig, ax = plt.subplots(figsize=(8, 6))

    for j, method in enumerate(methods):
        values = [idm_values[metric]] + raw_data[method][metric]
        ax.plot(
            x, values,
            marker='o', linestyle='-', linewidth=2, markersize=6,
            color=colors[j], label=label_map[method]  # 使用新图例标签
        )

    ax.set_xticks(x)
    ax.set_xticklabels(penetration_rates)
    ax.tick_params(axis='both', labelsize=18)
    ax.set_xlabel("Penetration Rate", fontsize=20)
    ax.set_ylabel(metric, fontsize=20)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=18, loc='best')  # 添加图例

    fig.tight_layout()
    fig.savefig(f"{metric}.pdf", format='pdf')
    plt.close(fig)
