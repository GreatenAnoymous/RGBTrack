trtexec --onnx=refinenet.onnx \
        --minShapes=A:1x6x160x160,B:1x6x160x160 \
        --optShapes=A:256x6x160x160,B:256x6x160x160 \
        --maxShapes=A:256x6x160x160,B:256x6x160x160 \
        --saveEngine=refinenet.trt \
        --fp16 \
        # --memPoolSize=workspace:4096




# trtexec --onnx=score_network.onnx \
#         --saveEngine=scorenet.trt \
#         --fp16 \
        # --minShapes=A:1x6x160x160,B:1x6x160x160 \
        # --optShapes=A:256x6x160x160,B:256x6x160x160 \
        # --maxShapes=A:256x6x160x160,B:256x6x160x160 \
        
        # --fp16 \
        # --verbose \