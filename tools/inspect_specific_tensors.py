import gguf
import numpy as np

reader = gguf.GGUFReader("tinyllama-q4.gguf")
for tensor in reader.tensors:
    if tensor.name in ["token_embd.weight", "blk.0.attn_q.weight"]:
        print(f"Tensor Name: {tensor.name}")
        print(f"  Shape: {tensor.shape}")
        print(f"  Dtype: {tensor.tensor_type}")
        
        raw_bytes = bytes(tensor.data)
        print(f"  Byte length: {len(raw_bytes)}")
        print(f"  First 32 bytes: {[f'0x{b:02X}' for b in raw_bytes[:32]]}")
