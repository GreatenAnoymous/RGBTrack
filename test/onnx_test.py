# coding utf-8
import os
import time
import torch
import torchvision
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

print(f"TensorRT version: {trt.__version__}")
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """
        host_mem: cpu memory
        device_mem: gpu memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def build_engine(onnx_file_path, engine_file_path, max_batch_size=1, fp16_mode=True, save_engine=True):
    """
    Args:
      max_batch_size: predefined size for memory allocation
      fp16_mode: whether to use FP16
      save_engine: whether to save the engine
    return:
      ICudaEngine
    """
    if os.path.exists(engine_file_path):
        print("Reading engine from file: {}".format(engine_file_path))
        with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    # Create builder and config
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    
    # Enable FP16 mode if requested
    if fp16_mode:
        config.set_flag(trt.BuilderFlag.FP16)

    # Create network
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    
    # Create ONNX parser
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX file
    if not os.path.exists(onnx_file_path):
        raise FileNotFoundError(f"ONNX file {onnx_file_path} not found!")
    
    print('Loading ONNX file from path {}...'.format(onnx_file_path))
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # Create optimization profile for dynamic shapes
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(
        input_name,  # Use actual input tensor name
        min=(1, 3, 224, 224),
        opt=(128, 3, 224, 224),
        max=(128, 3, 224, 224)
    )
    config.add_optimization_profile(profile)

    # Mark output if not already marked
    if network.num_outputs == 0:
        last_layer = network.get_layer(network.num_layers - 1)
        network.mark_output(last_layer.get_output(0))

    print("Building TensorRT engine. This may take a few minutes...")
    # Build and serialize network
    serialized_engine = builder.build_serialized_network(network, config)
    
    # Create runtime and deserialize engine
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    print("Completed creating engine")

    if save_engine and engine is not None:
        with open(engine_file_path, 'wb') as f:
            f.write(serialized_engine)
            print(f"Engine saved to {engine_file_path}")

    return engine
def allocate_buffers(engine, batch_size):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    
    for idx in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(idx)
        tensor_shape = engine.get_tensor_shape(tensor_name)
        tensor_dtype = engine.get_tensor_dtype(tensor_name)
        
        # Handle dynamic batch size
        if tensor_shape[0] == -1:
            print(f"Dynamic shape found for tensor '{tensor_name}'")
            tensor_shape = (batch_size,) + tuple(tensor_shape[1:])
            
        # Calculate size including batch dimension
        size = abs(trt.volume(tensor_shape))
            
        # Get numpy dtype
        dtype = trt.nptype(tensor_dtype)
        
        # Allocate host and device memory
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        
        bindings.append(int(device_mem))
        
        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
            
    return inputs, outputs, bindings, stream

def inference(context, bindings, inputs, outputs, stream, batch_size):
    # Copy input data to device
    for inp in inputs:
        cuda.memcpy_htod_async(inp.device, inp.host, stream)
    
    # Set tensor addresses for inputs and outputs
    for idx in range(context.engine.num_io_tensors):
        tensor_name = context.engine.get_tensor_name(idx)
        if context.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            context.set_tensor_address(tensor_name, inputs[0].device)
        else:
            context.set_tensor_address(tensor_name, outputs[0].device)
    
    # Execute inference
    context.execute_async_v3(stream_handle=stream.handle)
    
    # Transfer predictions back to host
    for out in outputs:
        cuda.memcpy_dtoh_async(out.host, out.device, stream)
    
    # Synchronize the stream
    stream.synchronize()
    
    # Return only the host outputs
    return [out.host for out in outputs]

def postprocess_the_outputs(h_outputs, shape_of_output):
    h_outputs = h_outputs.reshape(*shape_of_output)
    return h_outputs

if __name__ == '__main__':
    print(f"TensorRT version: {trt.__version__}")
    onnx_file_path = "resnet50.onnx"
    fp16_mode = False
    max_batch_size = 1
    trt_engine_path = "resnet50.trt"

    # 1. Create CUDA engine
    engine = build_engine(onnx_file_path, trt_engine_path, max_batch_size, fp16_mode)

    # 2. Create execution context
    
    
    context = engine.create_execution_context()
    batch_size = 32
    inputs, outputs, bindings, stream = allocate_buffers(engine, 252)

    # 3. Run inference
    output_shape = (batch_size, 1000)
    dummy_input = np.ones([batch_size, 3, 224, 224], dtype=np.float32)
    inputs[0].host = dummy_input.reshape(-1)
    input_name = engine.get_tensor_name(0)  # Get the name of the first input tensor
    context.set_input_shape(input_name, dummy_input.shape)

    for _ in range(10):
        trt_outputs = inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=batch_size)
    # Set input shape for dynamic input - using the new API
    torch.cuda.synchronize()
    t1 = time.time()

    trt_outputs = inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=batch_size)
    feat = postprocess_the_outputs(trt_outputs[0], output_shape)

    torch.cuda.synchronize()
    t2 = time.time()
    
    # 4. Compare with PyTorch

    model = torchvision.models.resnet50(pretrained=True).cuda()
    model = model.eval()
    dummy_input = torch.ones((batch_size, 3, 224, 224), dtype=torch.float32).cuda()
    for _ in range(10):
        _ = model(dummy_input)
    
    torch.cuda.synchronize()
    t3 = time.time()
    feat_2 = model(dummy_input)
    torch.cuda.synchronize()
    t4 = time.time()
    feat_2 = feat_2.cpu().data.numpy()

    mse = np.mean((feat - feat_2) ** 2)

    print("TensorRT engine time cost: {:.4f}s".format(t2 - t1))
    print("PyTorch model time cost: {:.4f}s".format(t4 - t3))
    print('MSE Error = {:.6f}'.format(mse))

