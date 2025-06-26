import matplotlib.pyplot as plt
import numpy as np
import os

# 设置字体为 Times New Roman，调优字号
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['legend.fontsize'] = 20

# 方法和渗透率
methods = ["Lane Only","Accel Only","Full Control" ]
penetrations = ["20%", "60%", "100%"]

# 色彩方案（Set2 风格）
method_colors = {
    "Baseline": "#999999",
    "Full Control": "#66c2a5",
    "Accel Only": "#fc8d62",
    "Lane Only": "#8da0cb"
}

# 各指标数据
metrics_data = {
    "Waiting Time (s)": {
        "baseline": 60.9,
        "data": [
            [61.01,38.15,33.56],
            [59.8, 20.27,16.95 ],
            [58.92, 11.8,9.43 ]
        ]
    },
    "Delay (s)": {
        "baseline": 161.79,
        "data": [
            [159.63, 157.35,149.01 ],
            [159.39, 135.37,127.03 ],
            [159.36, 119.82,116.75 ]
        ]
    },
    "Fuel Consumption (ml)": {
        "baseline": 103.8,
        "data": [
            [103.22, 96.91,93.05 ],
            [103.21, 87.23,84.16 ],
            [103.09, 81.79,78.7 ]
        ]
    },
    "Stop And Go": {
        "baseline": 1.94,
        "data": [
            [1.92, 1.29, 1.04],
            [1.92, 1.23,0.83 ],
            [1.91, 1.13,0.81 ]
        ]
    }
}

# 绘图参数
bar_width = 0.22
group_gap = 0.8
y_axis_starts = {
    "Waiting Time (s)": 0,
    "Delay (s)": 110,
    "Fuel Consumption (ml)": 75,
    "Stop And Go": 0.75
}

# 输出路径
output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)

# 生成图像
for metric_name, metric_info in metrics_data.items():
    baseline = metric_info["baseline"]
    data = metric_info["data"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x_baseline = -group_gap
    x = np.arange(len(penetrations)) * group_gap

    # baseline 柱加入图例
    ax.bar(x_baseline, baseline, width=bar_width, color=method_colors["Baseline"], label="Baseline")

    # 方法柱组
    for i, method in enumerate(methods):
        values = [data[j][i] for j in range(len(penetrations))]
        offsets = x + (i - 1) * bar_width
        ax.bar(offsets, values, width=bar_width, color=method_colors[method], label=method)

    # 设置 x 轴标签等
    ax.set_xticks(np.concatenate([[x_baseline], x]))
    ax.set_xticklabels(["0%"] + penetrations)  # 修改这里
    ax.set_xlabel("Penetration Rate")          # 添加这行
    ax.set_ylabel(metric_name)
    ax.set_ylim(bottom=y_axis_starts[metric_name])
    ax.grid(axis='y', linestyle='--', alpha=0.4)

 
    # 或者用这行替代，放在图下方：
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)

    plt.tight_layout(rect=[0, 0.01, 1, 1])
    # 图标题（放图下方）
    #fig.text(0.5, 0.045, metric_name, ha='center', fontsize=14, weight='bold')

    plt.tight_layout(rect=[0, 0.01, 1, 1])

    # 保存为 PDF（矢量图）
    filename = metric_name.replace(" ", "_").replace("(", "").replace(")", "") + ".pdf"
    fig.savefig(os.path.join(output_dir, filename), format='pdf', bbox_inches='tight')
    plt.close(fig)
