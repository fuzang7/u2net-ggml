import torch
import gguf
import numpy as np
import os

def convert(model_path, output_path):
    print(f"Loading model from {model_path}...")
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    
    # 1. 初始化: 这里第二个参数 "u2net" 就是架构名
    print("Initializing GGUF Writer...")
    gguf_writer = gguf.GGUFWriter(output_path, "u2net")
    
    # 2. 写入一些基础元数据 (可选，不写也不影响读取)
    # gguf_writer.add_name("U-2-Net") 
    # gguf_writer.add_description("U-2-Net model converted to GGUF")

    # 3. 遍历权重
    print("Converting tensors...")
    for name, tensor in state_dict.items():
        if "num_batches_tracked" in name:
            continue
            
        # 转换为 F32
        data = tensor.numpy().astype(np.float32)
        
        # 写入 Tensor
        # print(f"  Writing: {name} | Shape: {data.shape}")
        gguf_writer.add_tensor(name, data)
    
    # 4. 保存
    print(f"Saving to {output_path}...")
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    print("Done!")

if __name__ == "__main__":
    # 注意：你说你的文件名是 u2net_portrait.pth，请确认路径
    MODEL_PATH = "models/u2net_portrait.pth" 
    OUTPUT_PATH = "models/u2net_f32.gguf"
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.")
    else:
        convert(MODEL_PATH, OUTPUT_PATH)