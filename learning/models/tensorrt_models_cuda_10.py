import numpy as np
import tensorrt as trt
import torch
import pycuda.driver as cuda
import pycuda.autoinit
import time
from score_network import ScoreNetMultiPair
from refine_network import RefineNet
# print("tensorrt version:", trt.__version__)

torch.cuda.set_device(0)
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class RefineNetTrt(object):
    def __init__(self, trt_path, batch_size=252):
        self.engine = self.load_tensorrt_model(trt_path)
        self.current_batch_size = batch_size
        self.context = self.engine.create_execution_context()
        self.context.set_input_shape("A", (self.current_batch_size, 6, 160, 160))
        self.context.set_input_shape("B", (self.current_batch_size, 6, 160, 160))
        self.output_shape= (self.current_batch_size, 3)
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine, self.current_batch_size)

    def load_tensorrt_model(self, trt_path):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(trt_path, 'rb') as f:
            engine_data = f.read()
        runtime= trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(engine_data)
        return engine

    def change_batch_size(self, batch_size):
        self.current_batch_size = batch_size
        self.context.set_input_shape("A", (self.current_batch_size, 6, 160, 160))
        self.context.set_input_shape("B", (self.current_batch_size, 6, 160, 160))
        self.output_shape= (self.current_batch_size, 3)
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine, self.current_batch_size)
        # A= torch.randn(self.current_batch_size, 6, 160, 160).cuda()
        # B= torch.randn(self.current_batch_size, 6, 160, 160).cuda()
        # for i in range(10):
        #     out= self.run_inference(A,B)


    
    def allocate_buffers(self,engine, batch_size=1):
        '''
        Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
        '''
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            tensor_shape = engine.get_tensor_shape(tensor_name)
            
            # Handle dynamic batch size
            if tensor_shape[0] == -1:
                tensor_shape = (batch_size,) + tuple(tensor_shape[1:])
            size = trt.volume(tensor_shape)
            dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))

            # Allocate host and device buffers

            device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)

            # Append the device buffer address to device bindings. 
            # When cast to int, it's a linear index into the context's memory (like memory address). 
            bindings.append(int(device_mem))

        # Append to the appropriate input/output list.
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append(device_mem)
            else:
                outputs.append(device_mem)

        return inputs, outputs, bindings, stream


    
    def inference(self,context, bindings, inputs, outputs, stream, batch_size):
        context.set_tensor_address("A", inputs[0])
        context.set_tensor_address("B", inputs[1])
        context.set_tensor_address("trans", outputs[0])
        context.set_tensor_address("rot", outputs[1])
        
        # Execute inference
        context.execute_async_v3(stream_handle=stream.handle)
        # Transfer predictions back to host
        
        # Synchronize the stream
        stream.synchronize()
        
        # Return only the host outputs
        return outputs

    def postprocess_the_outputs(self,h_outputs, shape_of_output):
        h_outputs = h_outputs.reshape(*shape_of_output)
        return h_outputs

    def run_inference(self, A, B):
        batch_size=A.shape[0]
        if batch_size != self.current_batch_size:
            self.change_batch_size(batch_size)
        # Transfer tensors to TensorRT buffers
        cuda.memcpy_dtod_async(self.inputs[0], A.data_ptr(), A.numel() * A.element_size(), self.stream)
        cuda.memcpy_dtod_async(self.inputs[1], B.data_ptr(), B.numel() * B.element_size(), self.stream)
        h_outputs = self.inference(self.context, self.bindings, self.inputs, self.outputs, self.stream, batch_size)
        output_ptr1 = torch.cuda.FloatTensor(batch_size, 3).cuda()
        output_ptr2 = torch.cuda.FloatTensor(batch_size, 3).cuda()
        cuda.memcpy_dtod_async(output_ptr1.data_ptr(), h_outputs[0], output_ptr1.numel() * output_ptr1.element_size(), self.stream)
        cuda.memcpy_dtod_async(output_ptr2.data_ptr(), h_outputs[1], output_ptr2.numel() * output_ptr2.element_size(), self.stream)
        self.stream.synchronize()
        # h_outputs = self.inference(self.context, bindings, inputs, outputs, stream, 1)
        
        return {"trans": output_ptr1, "rot": output_ptr2}
    
class ScoreNetTrt(object):
    def __init__(self, trt_path, batch_size=252):
        self.engine = self.load_tensorrt_model(trt_path)
        self.current_batch_size = batch_size
        self.context = self.engine.create_execution_context()
        self.context.set_input_shape("A", (self.current_batch_size, 6, 160, 160))
        self.context.set_input_shape("B", (self.current_batch_size, 6, 160, 160))
        self.output_shape = (self.current_batch_size, 3)
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine, self.current_batch_size)
    
    def load_tensorrt_model(self, trt_path):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(trt_path, 'rb') as f:
            engine_data = f.read()
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(engine_data)
        return engine

    def allocate_buffers(self, engine, batch_size=1):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            tensor_shape = engine.get_tensor_shape(tensor_name)

            if tensor_shape[0] == -1:
                tensor_shape = (batch_size,) + tuple(tensor_shape[1:])
            if tensor_shape[1] == -1:
                tensor_shape = (tensor_shape[0], batch_size) + tuple(tensor_shape[2:])
            
            size = trt.volume(tensor_shape)
            dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
            print(f"tensor_shape: {tensor_shape}, tensor_name: {tensor_name}")
            print(f"Size: {size}, dtype: {dtype}")

            # Allocate device memory only
            device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
            bindings.append(int(device_mem))

            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append(device_mem)
            else:
                outputs.append(device_mem)

        return inputs, outputs, bindings, stream

    def inference(self, context, bindings, inputs, outputs, stream):
        # Set tensor addresses for inputs and outputs
        context.set_tensor_address("A", inputs[0])
        context.set_tensor_address("B", inputs[1])
        context.set_tensor_address("score_logit", outputs[0])

        # Execute inference
        context.execute_async_v3(stream_handle=stream.handle)
        
        # Synchronize the stream
        stream.synchronize()

        # Return output buffers (device pointers)
        return outputs

    def run_inference(self, A, B):
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"

        # Transfer tensors to TensorRT buffers
        cuda.memcpy_dtod_async(self.inputs[0], A.data_ptr(), A.numel() * A.element_size(), self.stream)
        cuda.memcpy_dtod_async(self.inputs[1], B.data_ptr(), B.numel() * B.element_size(), self.stream)
        
        # Perform inference
        h_outputs = self.inference(self.context, self.bindings, self.inputs, self.outputs, self.stream)
        
        # Create an output CUDA tensor
        output_ptr = torch.cuda.FloatTensor(1, self.current_batch_size).cuda()
        
        # Ensure you're copying device-to-device (GPU memory)
        cuda.memcpy_dtod_async(output_ptr.data_ptr(), h_outputs[0], output_ptr.numel() * output_ptr.element_size(), self.stream)
        
        # Synchronize the stream to ensure the copy completes before using the data
        self.stream.synchronize()

        return {"score_logit": output_ptr}



if __name__ == "__main__":
    batch_size= 252
    model= ScoreNetTrt("scorenet.trt",batch_size)
    
    # model= RefineNetTrt("refinenet.trt", batch_size)
    A = torch.randn(batch_size, 6, 160, 160).cuda()
    B = torch.randn(batch_size, 6, 160, 160).cuda()
    for i in range(10):
        out= model.run_inference(A,B)
    torch.cuda.synchronize()
    t0=time.time()
    for i in range(100):
        out= model.run_inference(A,B)
    # print(out["score_logit"].is_cuda)
    torch.cuda.synchronize()
    t1=time.time()
    print("Time:", t1-t0)
            
            
    
    # ckpt="/mnt/ssd_990/teng/ycb/fp/FoundationPose/weights/2023-10-28-18-33-37/model_best.pth"
    # net= RefineNet(c_in=6).cuda()
    # ckpt= torch.load(ckpt)
    # if 'model' in ckpt:
    #     ckpt = ckpt['model']
    # net.load_state_dict(ckpt)
    
    # net.cuda().eval()
    
    
    
    # print("A shape:", A.shape)
    # print("B shape:", B.shape)
    # for _ in range(10):
    #     with torch.no_grad():
    #         output = net(A,B)
    
    # torch.cuda.synchronize()
    # t0=time.time()
    # with torch.no_grad():
    #     for i in range(100):
        
    #         output = net(A,B)
    
    
    # torch.cuda.synchronize()
    # t1=time.time()
    # print(f"Time taken: {t1-t0}")

    # print(np.mean(np.abs(output['trans'].cpu().detach().numpy()-out["trans"].cpu().detach().numpy()))) 
    # print(np.mean(np.abs(output['rot'].cpu().detach().numpy()-out["rot"].cpu().detach().numpy())))  
    
    
    
    
    ckpt="/mnt/ssd_990/teng/ycb/fp/FoundationPose/weights/2024-01-11-20-02-45/model_best.pth"
    net= ScoreNetMultiPair(c_in=6).cuda()
    ckpt= torch.load(ckpt)
    if 'model' in ckpt:
        ckpt = ckpt['model']
    net.load_state_dict(ckpt)
    net.cuda().eval()
    for _ in range(10):
        with torch.no_grad():
            output = net(A,B)
    
    torch.cuda.synchronize()
    t0=time.time()
    with torch.no_grad():
        for i in range(100):
        
            output = net(A,B)
    
    
    torch.cuda.synchronize()
    t1=time.time()
    print(f"Time taken: {t1-t0}")
    print(np.mean(np.abs(output['score_logit'].cpu().detach().numpy()-out["score_logit"].cpu().detach().numpy()))) 
    # print(np.mean(np.abs(output['rot'].cpu().detach().numpy()-out["rot"].cpu().detach().numpy())))  
    
    