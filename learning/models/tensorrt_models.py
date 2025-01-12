import torch
import tensorrt as trt
from score_network import ScoreNetMultiPair
print(trt.__version__)


class RefineNetTrt(object):
    def __init__(self, trt_path, device="cuda"):
        """Initialize TensorRT plugins, engine and context."""
        self.trt_path = trt_path
        self.device = device
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.engine = self._load_engine()

        try:
            self.context = self.engine.create_execution_context()
            # Set optimization profile if dynamic shapes are used
            # self.context.active_optimization_profile = 0
            self.stream = torch.cuda.current_stream(device=self.device)
        except Exception as e:
            raise RuntimeError('fail to allocate CUDA resources') from e

        # Store shapes for later use
        self.input_shapes = []
        for binding in range(self.engine.num_bindings):
            if self.engine.binding_is_input(binding):
                shape = self.engine.get_binding_shape(binding)
                self.input_shapes.append(shape)

    def _load_engine(self):
        """Load TensorRT engine."""
        TRTbin = self.trt_path
        with open(TRTbin, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
        
    def infer(self, A, B, trans, rot):
        """Run inference on TensorRT engine."""
        # Set input shapes for dynamic shapes
        if -1 in self.input_shapes[0]:  # Check if dynamic shapes are used
            
            self.context.set_binding_shape(0, A.shape)
            self.input_shapes[0]=A.shape
            self.context.set_binding_shape(1, B.shape)
            self.input_shapes[1]=B.shape

        if self.input_shapes[0][0] != A.shape[0]:
            print("debug",self.input_shapes[0], A.shape)
            self.context.set_binding_shape(0, A.shape)
            self.context.set_binding_shape(1, B.shape)
            
        if not self.context.all_binding_shapes_specified:
            raise RuntimeError("Not all input shapes specified")
            
        if not self.context.all_shape_inputs_specified:
            raise RuntimeError("Not all shape inputs specified")

        self.bindings = [A.data_ptr(), B.data_ptr(), trans.data_ptr(),rot.data_ptr()]
        
        success = self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.cuda_stream
        )
        
        if not success:
            raise RuntimeError("Inference failed")
            
        self.stream.synchronize()
        return trans, rot
    
    def run_inference(self,A,B):
        if A.is_contiguous()==False:
            A= A.contiguous()
        if B.is_contiguous()==False:
            B= B.contiguous()
        trans = torch.zeros( A.shape[0], 3, device=self.device, dtype=A.dtype)
        rot = torch.zeros( A.shape[0], 3, device=self.device, dtype=A.dtype)
        trans, rot = self.infer(A, B, trans, rot)
        return {"trans": trans, "rot": rot}

class ScoreNetTrt(object):
    def __init__(self, trt_path, device="cuda"):
        """Initialize TensorRT plugins, engine and context."""
        self.trt_path = trt_path
        self.device = device
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.engine = self._load_engine()

        try:
            self.context = self.engine.create_execution_context()
            # Set optimization profile if dynamic shapes are used
            # self.context.active_optimization_profile = 0
            self.stream = torch.cuda.current_stream(device=self.device)
        except Exception as e:
            raise RuntimeError('fail to allocate CUDA resources') from e

        # Store shapes for later use
        self.input_shapes = []
        for binding in range(self.engine.num_bindings):
            if self.engine.binding_is_input(binding):
                shape = self.engine.get_binding_shape(binding)
                self.input_shapes.append(shape)

    def _load_engine(self):
        """Load TensorRT engine."""
        TRTbin = self.trt_path
        with open(TRTbin, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
        
    def infer(self, A, B, output):
        """Run inference on TensorRT engine."""
        # Set input shapes for dynamic shapes
        if -1 in self.input_shapes[0]:  # Check if dynamic shapes are used
            self.context.set_binding_shape(0, A.shape)
            self.context.set_binding_shape(1, B.shape)
        if not self.context.all_binding_shapes_specified:
            raise RuntimeError("Not all input shapes specified")
            
        if not self.context.all_shape_inputs_specified:
            raise RuntimeError("Not all shape inputs specified")

        self.bindings = [A.data_ptr(), B.data_ptr(), output.data_ptr()]
        
        success = self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.cuda_stream
        )
        
        if not success:
            raise RuntimeError("Inference failed")
            
        self.stream.synchronize()
        return output
    
    def run_inference(self,A,B):
        if A.is_contiguous()==False:
            A= A.contiguous()
        if B.is_contiguous()==False:
            B= B.contiguous()
        output = torch.zeros(1, A.shape[0], device=self.device, dtype=A.dtype)
        output = self.infer(A, B, output)
        return {"score_logit": output}

@torch.inference_mode()
def test_score_net():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    trt_path = 'scorenet.trt'
    trt_model = ScoreNetTrt(trt_path, device)
    batch_size = 252
    
    # Input datapoints
    A = torch.randn(batch_size, 6, 160, 160, device=device, dtype=torch.float32)
    B = torch.randn(batch_size, 6, 160, 160, device=device, dtype=torch.float32)
    
    # Pre-allocate output tensor
    output = torch.zeros(1, batch_size, device=device, dtype=torch.float32)
    
    # Run inference
    output = trt_model.infer(A, B, output)
    print(output)

    ckpt="/mnt/ssd_990/teng/ycb/fp/FoundationPose/weights/2024-01-11-20-02-45/model_best.pth"
    net= ScoreNetMultiPair(c_in=6).cuda()
    ckpt= torch.load(ckpt)
    if 'model' in ckpt:
        ckpt = ckpt['model']
    net.load_state_dict(ckpt)
    net.cuda().eval()
    output = net(A,B)
    print(output["score_logit"])


@torch.inference_mode()
def test_refine_net():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    trt_path = 'refinenet.trt'
    trt_model = RefineNetTrt(trt_path, device)
    batch_size = 1
    
    # Input datapoints
    A = torch.randn(batch_size, 6, 160, 160, device=device, dtype=torch.float32)
    B = torch.randn(batch_size, 6, 160, 160, device=device, dtype=torch.float32)
    
    # Pre-allocate output tensor
    trans = torch.zeros( batch_size,3, device=device, dtype=torch.float32)
    rot= torch.zeros(batch_size, 3, device=device, dtype=torch.float32)
    # Run inference
    trans, rot = trt_model.infer(A, B, trans, rot)
    

    # ckpt="/mnt/ssd_990/teng/ycb/fp/FoundationPose/weights/2024-01-11-20-02-45/model_best.pth"
    # net= ScoreNetMultiPair(c_in=6).cuda()
    # ckpt= torch.load(ckpt)
    # if 'model' in ckpt:
    #     ckpt = ckpt['model']
    # net.load_state_dict(ckpt)
    # net.cuda().eval()
    # output = net(A,B)
    # print(output["score_logit"])

if __name__ == '__main__':
    test_refine_net()