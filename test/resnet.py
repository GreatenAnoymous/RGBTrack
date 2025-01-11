import onnx
import torch
import torchvision


# 1. 定义模型
model = torchvision.models.resnet50(pretrained=True).cuda()

# 2.定义输入&输出
input_names = ['input']
output_names = ['output']
image = torch.randn(1, 3, 224, 224).cuda()

# 3.pt转onnx
onnx_file = "./resnet50.onnx"
torch.onnx.export(model, image, onnx_file, verbose=False,
                  input_names=input_names, output_names=output_names,
                  opset_version=11,
                  dynamic_axes={"input":{0: "batch_size"}, "output":{0: "batch_size"},})

# 4.检查onnx计算图
net = onnx.load("./resnet50.onnx")
onnx.checker.check_model(net)           # 检查文件模型是否正确

# 5.优化前后对比&验证
# 优化前
model.eval()
with torch.no_grad():
    output1 = model(image)

# 优化后
import onnxruntime

image = torch.randn(4, 3, 224, 224).cuda()
session = onnxruntime.InferenceSession("./resnet50.onnx")
session.get_modelmeta()
output2 = session.run(['output'], {"input": image.cpu().numpy()})
print("{}vs{}".format(output1.mean(), output2[0].mean()))
