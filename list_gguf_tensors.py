import gguf

reader = gguf.GGUFReader("tinyllama-q4.gguf")
print(f"Total tensors: {len(reader.tensors)}")
for idx, tensor in enumerate(reader.tensors):
    if idx < 30 or "output" in tensor.name or "token_embd" in tensor.name:
        print(f"{tensor.name}: shape={tensor.shape}, dtype={tensor.tensor_type}")
