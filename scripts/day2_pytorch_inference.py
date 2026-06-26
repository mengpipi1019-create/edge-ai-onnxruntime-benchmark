import time
import torch
import torchvision.models as models


def main():
    # 1. 选择设备：有 GPU 就用 GPU，没有就用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 2. 加载 ResNet18 模型
    # weights=None 表示不下载预训练权重，先验证推理流程能跑通
    model = models.resnet18(weights=None)
    model = model.to(device)

    # 3. 设置为推理模式
    model.eval()

    # 4. 构造一张假的图片输入
    # ResNet18 标准输入格式：[batch_size, channels, height, width]
    # 这里表示 1 张 RGB 图片，大小是 224x224
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    # 5. 关闭梯度计算，进行推理
    with torch.no_grad():
        start_time = time.time()
        output = model(dummy_input)
        end_time = time.time()

    # 6. 输出结果形状
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)

    # 7. 获取预测类别
    pred_class = torch.argmax(output, dim=1).item()
    print("Predicted class index:", pred_class)

    # 8. 计算推理耗时
    latency_ms = (end_time - start_time) * 1000
    print("Inference latency: {:.3f} ms".format(latency_ms))


if __name__ == "__main__":
    main()