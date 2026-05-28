import gguf
import numpy as np

reader = gguf.GGUFReader("tinyllama-q4.gguf")
tensors = {t.name: t for t in reader.tensors}

for name in ["blk.0.attn_v.weight", "blk.0.attn_q.weight", "blk.0.ffn_down.weight", "output.weight"]:
    t = tensors.get(name)
    if not t:
        print(f"\n{name}: NOT FOUND")
        continue
    data = t.data
    print(f"\n{name}: shape={t.shape}, dtype_code={t.tensor_type}, data_type={type(data)}, data_len={len(data)}")
    if hasattr(data, 'shape'):
        print(f"  data numpy shape: {data.shape}, dtype: {data.dtype}")
    if len(data) >= 4:
        raw = np.frombuffer(data[0:4], dtype=np.uint8)
        print(f"  First 4 raw bytes: {raw.tolist()}")
        # Try as f16
        d_f16 = np.frombuffer(data[0:2], dtype=np.float16)[0]
        print(f"  As f16: {d_f16}")
        # Try as f32
        d_f32 = np.frombuffer(data[0:4], dtype=np.float32)[0]
        print(f"  As f32: {d_f32}")
    if len(data) > 200:
        sc0 = np.frombuffer(data[194:195], dtype=np.int8)[0]
        print(f"  Byte 194 as int8: {sc0}")
