import numpy as np
import tensorrt as trt
import torch
import pycuda.driver as cuda
import pycuda.autoinit
import time
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

            host_mem = cuda.pagelocked_empty(size, dtype) # page-locked memory buffer (won't swapped to disk)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Append the device buffer address to device bindings. 
            # When cast to int, it's a linear index into the context's memory (like memory address). 
            bindings.append(int(device_mem))

        # Append to the appropriate input/output list.
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream


    
    def inference(self,context, bindings, inputs, outputs, stream, batch_size):
        # Copy input data to device
        for inp in inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, stream)
        
        # Set tensor addresses for inputs and outputs
        for idx in range(context.engine.num_io_tensors):
            tensor_name = context.engine.get_tensor_name(idx)
            if tensor_name =="A":
                context.set_tensor_address(tensor_name, inputs[0].device)
            elif tensor_name =="B":
                context.set_tensor_address(tensor_name, inputs[1].device)
            elif tensor_name =="trans":
                context.set_tensor_address(tensor_name, outputs[0].device)
            elif tensor_name =="rot":
                context.set_tensor_address(tensor_name, outputs[1].device)

        
        # Execute inference
        context.execute_async_v3(stream_handle=stream.handle)
        # Transfer predictions back to host
        for out in outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, stream)
        
        # Synchronize the stream
        stream.synchronize()
        
        # Return only the host outputs
        return [out.host for out in outputs]

    def postprocess_the_outputs(self,h_outputs, shape_of_output):
        h_outputs = h_outputs.reshape(*shape_of_output)
        return h_outputs

    def run_inference(self, A, B):
        batch_size=A.shape[0]
        if batch_size != self.current_batch_size:
            self.change_batch_size(batch_size)
        self.inputs[0].host= A.reshape(-1)
        self.inputs[1].host= B.reshape(-1)
        h_outputs = self.inference(self.context, self.bindings, self.inputs, self.outputs, self.stream, batch_size)
        # h_outputs = self.inference(self.context, bindings, inputs, outputs, stream, 1)
        feat1= self.postprocess_the_outputs(h_outputs[0], (batch_size,3))
        feat2= self.postprocess_the_outputs(h_outputs[1], (batch_size,3))
        return {"trans": torch.from_numpy(feat1).cuda(), "rot": torch.from_numpy(feat2).cuda()}

class ScoreNetTrt(object):
    def __init__(self, trt_path, batch_size=252):
        self.engine = self.load_tensorrt_model(trt_path)
        self.current_batch_size = batch_size
        self.context = self.engine.create_execution_context()
        self.context.set_input_shape("A", (self.current_batch_size, 6, 160, 160))
        self.context.set_input_shape("B", (self.current_batch_size, 6, 160, 160))
        self.output_shape= (self.current_batch_size, 3)
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine, self.current_batch_size)
        # self.warm_up()
    def load_tensorrt_model(self, trt_path):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(trt_path, 'rb') as f:
            engine_data = f.read()
        runtime= trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(engine_data)
        return engine
    
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
            if tensor_shape[1] == -1:
                tensor_shape = (tensor_shape[0], batch_size) + tuple(tensor_shape[2:])
        

            
            size = trt.volume(tensor_shape)
            dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
            print("tensor_shape:", tensor_shape, "tensor_name:", tensor_name)
            # Allocate host and device buffers
            print("Size:", size, "dtype:", dtype)
            host_mem = cuda.pagelocked_empty(size, dtype) # page-locked memory buffer (won't swapped to disk)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Append the device buffer address to device bindings. 
            # When cast to int, it's a linear index into the context's memory (like memory address). 
            bindings.append(int(device_mem))

        # Append to the appropriate input/output list.
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream
    
    def inference(self,context, bindings, inputs, outputs, stream, batch_size):
        # Copy input data to device
        for inp in inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, stream)
        
        # Set tensor addresses for inputs and outputs
        for idx in range(context.engine.num_io_tensors):
            tensor_name = context.engine.get_tensor_name(idx)
            if tensor_name =="A":
                context.set_tensor_address(tensor_name, inputs[0].device)
            elif tensor_name =="B":
                context.set_tensor_address(tensor_name, inputs[1].device)
            elif tensor_name =="score_logit":
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
    
    def postprocess_the_outputs(self,h_outputs, shape_of_output):
        h_outputs = h_outputs.reshape(*shape_of_output)
        return h_outputs
    
    def run_inference(self, A, B):
        batch_size=A.shape[0]
        
        if batch_size != self.current_batch_size:
            self.change_batch_size(batch_size)
        self.inputs[0].host= A.numpy().reshape(-1)
        self.inputs[1].host= B.numpy().reshape(-1)
        h_outputs = self.inference(self.context, self.bindings, self.inputs, self.outputs, self.stream, batch_size)
        # h_outputs = self.inference(self.context, bindings, inputs, outputs, stream, 1)
        feat1= self.postprocess_the_outputs(h_outputs[0], (1, batch_size))
        return {"score_logit": torch.from_numpy(feat1).cuda()}



if __name__ == "__main__":
    B=252
    model= ScoreNetTrt("scorenet.trt",B)
    A = torch.randn(B, 6, 160, 160).cuda()
    B = torch.randn(B, 6, 160, 160).cuda()
    # A= np.random.randn(B, 6, 160, 160).astype(np.float16)
    # B= np.random.randn(B, 6, 160, 160).astype(np.float16)
    for i in range(10):
        out= model.run_inference(A,B)
    torch.cuda.synchronize()
    t0=time.time()
    out= model.run_inference(A,B)
    torch.cuda.synchronize()
    t1=time.time()
    print("Time:", t1-t0)
            