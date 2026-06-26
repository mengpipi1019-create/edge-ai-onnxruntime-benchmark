from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def save_line_chart(df, x_col, y_cols, labels, title, ylabel, output_path):
    """
    画折线图并保存。
    df: 读取到的 CSV 数据
    x_col: 横轴列名
    y_cols: 纵轴列名列表
    labels: 图例名称列表
    title: 图标题
    ylabel: 纵轴名称
    output_path: 图片保存路径
    """
    plt.figure(figsize=(8, 5))

    for y_col, label in zip(y_cols, labels):
        plt.plot(df[x_col], df[y_col], marker="o", label=label)

    plt.xlabel("Batch Size")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(df[x_col])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print("Saved:", output_path)


def main():
    # 1. 输入 CSV 路径
    csv_path = Path("results/day6_batch_benchmark.csv")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}\n"
            "请先确认 Day 6 是否已经生成 results/day6_batch_benchmark.csv"
        )

    # 2. 输出图片目录
    output_dir = Path("results/day7_figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. 读取 CSV
    df = pd.read_csv(csv_path)

    print("Loaded CSV:", csv_path)
    print("\nCSV columns:")
    print(df.columns.tolist())

    print("\nData preview:")
    print(df)

    # 4. 图 1：batch size vs batch latency
    save_line_chart(
        df=df,
        x_col="batch_size",
        y_cols=["pytorch_batch_latency_ms", "onnx_batch_latency_ms"],
        labels=["PyTorch", "ONNX Runtime"],
        title="Batch Size vs Batch Latency",
        ylabel="Batch Latency (ms)",
        output_path=output_dir / "day7_batch_latency_curve.png"
    )

    # 5. 图 2：batch size vs image latency
    save_line_chart(
        df=df,
        x_col="batch_size",
        y_cols=["pytorch_image_latency_ms", "onnx_image_latency_ms"],
        labels=["PyTorch", "ONNX Runtime"],
        title="Batch Size vs Image Latency",
        ylabel="Image Latency (ms)",
        output_path=output_dir / "day7_image_latency_curve.png"
    )

    # 6. 图 3：batch size vs FPS
    save_line_chart(
        df=df,
        x_col="batch_size",
        y_cols=["pytorch_fps", "onnx_fps"],
        labels=["PyTorch", "ONNX Runtime"],
        title="Batch Size vs FPS",
        ylabel="FPS",
        output_path=output_dir / "day7_fps_curve.png"
    )

    # 7. 图 4：batch size vs speedup
    save_line_chart(
        df=df,
        x_col="batch_size",
        y_cols=["speedup"],
        labels=["ONNX Runtime Speedup"],
        title="ONNX Runtime Speedup over PyTorch",
        ylabel="Speedup (x)",
        output_path=output_dir / "day7_speedup_curve.png"
    )

    print("\nAll figures saved to:", output_dir)


if __name__ == "__main__":
    main()