import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import warnings

warnings.filterwarnings("ignore")


def load_and_prepare_data(csv_path, gemm_type="trivial"):
    """加载并准备数据，支持trivial和grouped两种类型"""
    try:
        df = pd.read_csv(csv_path)

        if gemm_type == "trivial":
            required_cols = [
                "m",
                "k",
                "n",
                "time_us",
                "throughput_TFLOPS",
                "bandwidth_GBs",
            ]
        else:  # grouped
            required_cols = [
                "num_groups",
                "valid_m",
                "k",
                "n",
                "time_us",
                "throughput_TFLOPS",
                "bandwidth_GBs",
            ]

        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")

        return df
    except Exception as e:
        raise Exception(f"Failed to load data: {e}")


def interpolate_metric(data, metric, grid_size=50):
    """对单个指标进行插值"""
    k_values = sorted(data["k"].unique())
    n_values = sorted(data["n"].unique())

    if len(k_values) < 2 or len(n_values) < 2:
        return None, None, None

    k_grid = np.linspace(min(k_values), max(k_values), grid_size)
    n_grid = np.linspace(min(n_values), max(n_values), grid_size)
    K, N = np.meshgrid(k_grid, n_grid)

    points = data[["k", "n"]].values
    values = data[metric].values

    try:
        Z = griddata(points, values, (K, N), method="cubic")
        return K, N, Z
    except Exception as e:
        raise Exception(f"Failed to interpolate data: {e}")


def plot_3d_surface(ax, K, N, Z, title, metric_label, cmap="coolwarm"):
    """绘制3D曲面图"""
    if K is None or N is None or Z is None:
        ax.text(
            0.5,
            0.5,
            0.5,
            "Insufficient data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return None

    surf = ax.plot_surface(K, N, Z, cmap=cmap, linewidth=0, antialiased=True, alpha=0.8)

    z_range = Z.max() - Z.min()
    z_offset = Z.min() - 0.1 * z_range if z_range > 0 else Z.min() - 1
    ax.contourf(K, N, Z, zdir="z", offset=z_offset, cmap=cmap, alpha=0.5)

    ax.set_xlabel("k", fontsize=8)
    ax.set_ylabel("n", fontsize=8)
    ax.set_zlabel(metric_label, fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_zlim(z_offset, Z.max())

    return surf


def plot_trivial_gemm(csv_path, output_dir="./results/deep_gemm/trivial"):
    """绘制trivial GEMM性能图表"""
    df = load_and_prepare_data(csv_path, "trivial")

    import os

    os.makedirs(output_dir, exist_ok=True)

    metrics_info = {
        "time_us": "Execution Time (μs)",
        "throughput_TFLOPS": "Throughput (TFLOPS)",
        "bandwidth_GBs": "Bandwidth (GB/s)",
    }

    for m in df["m"].unique():
        m_data = df[df["m"] == m]

        if len(m_data) < 4:
            print(f"Skipping m={m}: insufficient data points")
            continue

        fig = plt.figure(figsize=(18, 6))
        fig.suptitle(f"Performance Metrics for m={m}", fontsize=16)

        for i, (metric, title) in enumerate(metrics_info.items()):
            ax = fig.add_subplot(1, 3, i + 1, projection="3d")
            K, N, Z = interpolate_metric(m_data, metric)
            surf = plot_3d_surface(ax, K, N, Z, title, title)
            if surf:
                fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        plt.tight_layout()
        output_path = f"{output_dir}/contour_plot_m_{m}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")


def plot_grouped_gemm(
    csv_path, output_dir="./results/deep_gemm/grouped", layout="auto"
):
    """绘制grouped GEMM性能图表 - 灵活布局"""
    df = load_and_prepare_data(csv_path, "grouped")

    import os

    os.makedirs(output_dir, exist_ok=True)

    metrics_info = {
        "time_us": "Time (μs)",
        "throughput_TFLOPS": "Throughput (TFLOPS)",
        "bandwidth_GBs": "Bandwidth (GB/s)",
    }

    for valid_m in sorted(df["valid_m"].unique()):
        m_data = df[df["valid_m"] == valid_m]
        num_groups_list = sorted(m_data["num_groups"].unique())
        n_groups = len(num_groups_list)

        if n_groups == 0:
            continue

        if isinstance(layout, tuple):
            rows, cols = layout
        else:
            rows, cols = determine_layout(n_groups, len(metrics_info), layout)

        fig_width = min(6 * cols, 30)
        fig_height = min(5 * rows, 25)

        fig = plt.figure(figsize=(fig_width, fig_height))
        fig.suptitle(
            f"Performance Metrics for valid_m={valid_m} ({n_groups} groups)",
            fontsize=16,
        )

        plot_idx = 1
        for i, num_groups in enumerate(num_groups_list):
            group_data = m_data[m_data["num_groups"] == num_groups]

            for j, (metric, title) in enumerate(metrics_info.items()):
                ax = fig.add_subplot(rows, cols, plot_idx, projection="3d")

                if len(group_data) < 4:
                    ax.text(
                        0.5,
                        0.5,
                        0.5,
                        f"Insufficient data\nfor num_groups={num_groups}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=8,
                    )
                    ax.set_title(f"G={num_groups}: {title}", fontsize=9)
                else:
                    K, N, Z = interpolate_metric(group_data, metric)
                    surf = plot_3d_surface(
                        ax, K, N, Z, f"G={num_groups}: {title}", title
                    )

                    if surf:
                        cbar = fig.colorbar(surf, ax=ax, shrink=0.4, aspect=5, pad=0.1)
                        cbar.ax.tick_params(labelsize=6)

                ax.view_init(elev=20, azim=45)
                ax.tick_params(axis="both", which="major", labelsize=6)

                plot_idx += 1

        while plot_idx <= rows * cols:
            ax = fig.add_subplot(rows, cols, plot_idx)
            ax.axis("off")
            plot_idx += 1

        plt.tight_layout()
        output_path = f"{output_dir}/contour_plot_valid_m_{valid_m}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path} (layout: {rows}x{cols})")


def determine_layout(n_groups, n_metrics, mode="auto"):
    """智能确定子图布局"""
    total_plots = n_groups * n_metrics

    if mode == "compact":
        cols = n_metrics
        rows = n_groups
    elif mode == "square":
        cols = int(np.ceil(np.sqrt(total_plots)))
        rows = int(np.ceil(total_plots / cols))
    else:  # auto
        if n_groups <= 4:
            cols = n_metrics
            rows = n_groups
        elif n_groups <= 8:
            cols = n_metrics * 2
            rows = int(np.ceil(n_groups / 2))
        else:
            cols = int(np.ceil(np.sqrt(total_plots)))
            rows = int(np.ceil(total_plots / cols))

    return rows, cols


def plot_comparison(
    trivial_csv, grouped_csv, output_dir="./results/deep_gemm/comparison"
):
    """对比trivial和grouped GEMM的性能差异"""
    import os

    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    trivial_df = load_and_prepare_data(trivial_csv, "trivial")
    grouped_df = load_and_prepare_data(grouped_csv, "grouped")

    metrics_info = {
        "time_us": ("Time Difference (%)", "RdBu_r", "lower"),  # 时间越低越好
        "throughput_TFLOPS": (
            "Throughput Difference (%)",
            "RdBu",
            "higher",
        ),  # 吞吐量越高越好
        "bandwidth_GBs": ("Bandwidth Difference (%)", "RdBu", "higher"),  # 带宽越高越好
    }

    # 找出共同的m值
    trivial_m_values = set(trivial_df["m"].unique())
    grouped_m_values = set(grouped_df["valid_m"].unique())
    common_m_values = sorted(trivial_m_values.intersection(grouped_m_values))

    if not common_m_values:
        print("No common m values found between trivial and grouped data")
        return

    for m in common_m_values:
        # 获取trivial数据
        trivial_data = trivial_df[trivial_df["m"] == m]
        grouped_m_data = grouped_df[grouped_df["valid_m"] == m]

        num_groups_list = sorted(grouped_m_data["num_groups"].unique())
        n_groups = len(num_groups_list)

        if n_groups == 0 or len(trivial_data) < 4:
            continue

        # 确定布局
        rows = n_groups
        cols = 3  # 三个指标

        fig_width = min(6 * cols, 24)
        fig_height = min(5 * rows, 25)

        fig = plt.figure(figsize=(fig_width, fig_height))
        fig.suptitle(
            f"Performance Comparison: Grouped vs Trivial (m={m})\n"
            + "Positive values (red) indicate grouped is better",
            fontsize=16,
        )

        plot_idx = 1
        for num_groups in num_groups_list:
            group_data = grouped_m_data[grouped_m_data["num_groups"] == num_groups]

            if len(group_data) < 4:
                for j in range(3):
                    ax = fig.add_subplot(rows, cols, plot_idx)
                    ax.text(
                        0.5,
                        0.5,
                        f"Insufficient data\nfor num_groups={num_groups}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(f"G={num_groups}")
                    plot_idx += 1
                continue

            # 合并数据以确保相同的(k,n)点
            merged = pd.merge(
                trivial_data[
                    ["k", "n", "time_us", "throughput_TFLOPS", "bandwidth_GBs"]
                ],
                group_data[["k", "n", "time_us", "throughput_TFLOPS", "bandwidth_GBs"]],
                on=["k", "n"],
                suffixes=("_trivial", "_grouped"),
            )

            if len(merged) < 4:
                for j in range(3):
                    ax = fig.add_subplot(rows, cols, plot_idx)
                    ax.text(
                        0.5,
                        0.5,
                        f"Insufficient matching data\nfor num_groups={num_groups}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(f"G={num_groups}")
                    plot_idx += 1
                continue

            # 计算并绘制每个指标的差异
            for metric, (title, cmap, better) in metrics_info.items():
                ax = fig.add_subplot(rows, cols, plot_idx, projection="3d")

                # 计算百分比差异
                if metric == "time_us":
                    # 时间: (trivial - grouped) / trivial * 100
                    # 正值表示grouped更快
                    merged["diff"] = (
                        (merged[f"{metric}_trivial"] - merged[f"{metric}_grouped"])
                        / merged[f"{metric}_trivial"]
                        * 100
                    )
                else:
                    # 吞吐量和带宽: (grouped - trivial) / trivial * 100
                    # 正值表示grouped更好
                    merged["diff"] = (
                        (merged[f"{metric}_grouped"] - merged[f"{metric}_trivial"])
                        / merged[f"{metric}_trivial"]
                        * 100
                    )

                # 插值差异数据
                diff_data = merged[["k", "n", "diff"]]
                K, N, Z = interpolate_metric(
                    diff_data.rename(columns={"diff": "metric"}), "metric"
                )

                if K is not None:
                    surf = plot_3d_surface(
                        ax, K, N, Z, f"G={num_groups}: {title}", title, cmap=cmap
                    )

                    if surf:
                        cbar = fig.colorbar(surf, ax=ax, shrink=0.4, aspect=5, pad=0.1)
                        cbar.ax.tick_params(labelsize=6)

                ax.view_init(elev=20, azim=45)
                ax.tick_params(axis="both", which="major", labelsize=6)

                plot_idx += 1

        plt.tight_layout()
        output_path = f"{output_dir}/comparison_m_{m}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path} (layout: {rows}x{cols})")
