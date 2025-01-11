import torch
import torchvision.models as models
import time

# 加载模型并切换到 eval 模式
model = models.resnet50(pretrained=True).cuda().eval()

# 创建输入数据 (224x224 RGB 图片)
input_tensor = torch.randn(1, 3, 224, 224).cuda()  # Batch Size = 1

# 预热 GPU
for _ in range(10):
    _ = model(input_tensor)

# 测量推理时间
torch.cuda.synchronize()
start_time = time.time()
with torch.no_grad():
    for _ in range(100):  # 测量 100 次推理
        _ = model(input_tensor)
torch.cuda.synchronize()
end_time = time.time()

avg_time = (end_time - start_time) / 100
print(f"平均推理时间: {avg_time * 1000:.2f} ms")
