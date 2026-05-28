import gguf
import numpy as np

reader = gguf.GGUFReader("tinyllama-q4.gguf")
for tensor in reader.tensors:
    # Only check F32 tensors for now as they are easy to parse
    if tensor.tensor_type == gguf.GGMLQuantizationType.F32:
        floats = np.frombuffer(tensor.data, dtype=np.float32)
        nan_count = np.isnan(floats).sum()
        if nan_count > 0:
            print(f"Tensor {tensor.name} contains {nan_count} NaNs in python reader!")
        else:
            # Also print min/max/mean to check if values are reasonable
            print(f"Tensor {tensor.name}: no NaNs. Range: {floats.min()} to {floats.max()}, mean: {floats.mean()}")
