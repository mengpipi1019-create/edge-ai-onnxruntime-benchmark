import torch
import torchvision.models as models
from PIL import Image

# 1. 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 2. 加载 ResNet18 预训练模型
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# 3. 把模型放到设备上，并设置为推理模式
model = model.to(device)
model.eval()

# 4. 获取预处理流程和类别名称
preprocess = weights.transforms()
categories = weights.meta["categories"]

# 5. 读取图片
image = Image.open("images/cat.jpg").convert("RGB")

# 6. 图片预处理
input_tensor = preprocess(image)

# 7. 增加 batch 维度，并放到设备上
input_batch = input_tensor.unsqueeze(0).to(device)

print("Input batch shape:", input_batch.shape)

# 8. 推理
with torch.no_grad():
    output = model(input_batch)

# 9. logits 转概率
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# 10. 取 Top-3
top3_prob, top3_id = torch.topk(probabilities, 3)

# 11. 打印结果
print("\nTop-3 predictions:")
for rank, (prob, class_id) in enumerate(zip(top3_prob, top3_id), start=1):
    class_name = categories[class_id.item()]
    print(f"{rank}. {class_name} | prob: {prob.item():.4f}")