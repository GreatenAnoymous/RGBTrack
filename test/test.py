import torch
import argparse
import time
import numpy as np
import torch_tensorrt

# Define a simple PyTorch model
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(64 * 16 * 16, 512)
        self.relu3 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = x.view(-1, 64 * 16 * 16)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

def compute(use_tensorrt=False):
    force_cpu = False
    useCuda = torch.cuda.is_available() and not force_cpu
    if useCuda:
        print('Using CUDA.')
        dtype = torch.cuda.FloatTensor
        ltype = torch.cuda.LongTensor
        device = torch.device("cuda:0")
    else:
        print('No CUDA available.')
        dtype = torch.FloatTensor
        ltype = torch.LongTensor
        device = torch.device("cpu")

    model = MyModel()

    input_shape = (8192, 3, 32, 32)

    if use_tensorrt:
        model = torch.compile(
            model,
            backend="torch_tensorrt",
            options={
                "truncate_long_and_double": True,
                "precision": dtype,
                "workspace_size" : 20 << 30
            },
            dynamic=False,
        )

    model = model.to(device)
    model.eval()

    num_iterations = 100
    total_time = 0.0
    with torch.no_grad():
        input_data = torch.randn(input_shape).to(device).type(dtype)
        #warmup
        for i in range(100):
            output_data = model(input_data)

        for i in range(num_iterations):
            start_time = time.time()
            output_data = model(input_data)
            end_time = time.time()
            total_time += end_time - start_time
    pytorch_fps = num_iterations / total_time
    print(f"PyTorch FPS: {pytorch_fps:.2f}")

if __name__ == "__main__":
    print("Without TensorRT")
    compute()
    print("With TensorRT")
    compute(use_tensorrt=True)