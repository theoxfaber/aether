import gguf
import numpy as np

reader = gguf.GGUFReader("tinyllama-q4.gguf")
for tensor in reader.tensors:
    if tensor.name in ["blk.0.attn_norm.weight", "blk.0.ffn_norm.weight"]:
        print(f"Tensor Name: {tensor.name}")
        print(f"  Shape: {tensor.shape}")
        print(f"  Dtype: {tensor.tensor_type}")
        floats = np.frombuffer(tensor.data, dtype=np.float32)
        print(f"  First 16 floats: {floats[:16].tolist()}")
        print(f"  Float range: {floats.min()} to {floats.max()}")
