"""Verify Q6_K block layout by comparing Python dequant with both layouts."""
import gguf
import numpy as np
from gguf.quants import Q6_K

reader = gguf.GGUFReader("tinyllama-q4.gguf")
tensors = {t.name: t for t in reader.tensors}

# Get a Q6_K weight tensor
v = tensors["blk.0.attn_v.weight"]
flat = v.data.reshape(-1).astype(np.uint8)
raw_block = flat[:210]

print("=== Raw first block bytes ===")
print(f"d at start [0:2] as f16: {np.frombuffer(raw_block[0:2], dtype=np.float16)[0]}")
print(f"d at end [208:210] as f16: {np.frombuffer(raw_block[208:210], dtype=np.float16)[0]}")
print(f"scales[0:4] at [194:198] as int8: {raw_block[194:198].view(np.int8)}")
print(f"scales[0:4] at [192:196] as int8: {raw_block[192:196].view(np.int8)}")

# Use Python library's dequantize
blocks = raw_block.reshape(1, 210)
dequantized = Q6_K.dequantize_blocks(blocks)[0]
print(f"\nPython Q6_K.dequantize_blocks:")
print(f"  mean={dequantized.mean():.6f} std={dequantized.std():.6f}")
print(f"  range=[{dequantized.min():.6f}, {dequantized.max():.6f}]")
print(f"  first 8 values: {dequantized[:8].tolist()}")

# Now try with ggml layout (d at start, same shuffle)
def dequant_ggml(raw_bytes):
    """Use Python's dequantize_blocks logic but with d at start (ggml layout)."""
    d = np.frombuffer(raw_bytes[0:2], dtype=np.float16)[0]
    ql = raw_bytes[2:130]    # bytes 2..129
    qh = raw_bytes[130:194]  # bytes 130..193
    sc = raw_bytes[194:210].view(np.int8).astype(np.float32)

    u = np.zeros(256, dtype=np.float32)
    for half in range(2):
        qlo = half * 64; qho = half * 32; sco = half * 8
        for l in range(32):
            is_ = l // 16
            ql_l = int(ql[qlo + l]); ql_l32 = int(ql[qlo + l + 32])
            qh_l = int(qh[qho + l])
            q1 = ((ql_l & 0x0F) | ((qh_l & 0x03) << 4)) - 32
            q2 = ((ql_l32 & 0x0F) | ((qh_l & 0x0C) << 2)) - 32
            q3 = ((ql_l >> 4) | ((qh_l & 0x30))) - 32
            q4 = ((ql_l32 >> 4) | ((qh_l & 0xC0) >> 2)) - 32
            s1 = float(np.int8(sc[sco + is_ + 0]))
            s2 = float(np.int8(sc[sco + is_ + 2]))
            s3 = float(np.int8(sc[sco + is_ + 4]))
            s4 = float(np.int8(sc[sco + is_ + 6]))
            hb = half * 128
            u[hb + l] = float(d) * s1 * q1
            u[hb + l + 32] = float(d) * s2 * q2
            u[hb + l + 64] = float(d) * s3 * q3
            u[hb + l + 96] = float(d) * s4 * q4
    return u

# ggml layout: d at bytes 0..1
out_ggml = dequant_ggml(raw_block)
print(f"\nGGML layout (d at start):")
print(f"  mean={out_ggml.mean():.6f} std={out_ggml.std():.6f}")
print(f"  range=[{out_ggml.min():.6f}, {out_ggml.max():.6f}]")
print(f"  first 8 values: {out_ggml[:8].tolist()}")

# Python layout: d at bytes 208..209, swap ql/qh/sc positions
def dequant_py(raw_bytes):
    """Use Python's block layout (d at end) with correct shuffle."""
    d = np.frombuffer(raw_bytes[208:210], dtype=np.float16)[0]
    ql = raw_bytes[0:128]
    qh = raw_bytes[128:192]
    sc = raw_bytes[192:208].view(np.int8).astype(np.float32)

    u = np.zeros(256, dtype=np.float32)
    for half in range(2):
        qlo = half * 64; qho = half * 32; sco = half * 8
        for l in range(32):
            is_ = l // 16
            ql_l = int(ql[qlo + l]); ql_l32 = int(ql[qlo + l + 32])
            qh_l = int(qh[qho + l])
            q1 = ((ql_l & 0x0F) | ((qh_l & 0x03) << 4)) - 32
            q2 = ((ql_l32 & 0x0F) | ((qh_l & 0x0C) << 2)) - 32
            q3 = ((ql_l >> 4) | ((qh_l & 0x30))) - 32
            q4 = ((ql_l32 >> 4) | ((qh_l & 0xC0) >> 2)) - 32
            s1 = float(np.int8(sc[sco + is_ + 0]))
            s2 = float(np.int8(sc[sco + is_ + 2]))
            s3 = float(np.int8(sc[sco + is_ + 4]))
            s4 = float(np.int8(sc[sco + is_ + 6]))
            hb = half * 128
            u[hb + l] = float(d) * s1 * q1
            u[hb + l + 32] = float(d) * s2 * q2
            u[hb + l + 64] = float(d) * s3 * q3
            u[hb + l + 96] = float(d) * s4 * q4
    return u

out_py = dequant_py(raw_block)
print(f"\nPython layout (d at end):")
print(f"  mean={out_py.mean():.6f} std={out_py.std():.6f}")
print(f"  range=[{out_py.min():.6f}, {out_py.max():.6f}]")
print(f"  first 8 values: {out_py[:8].tolist()}")

# Compare with Python library output - should match
diff = np.abs(dequantized - out_py).max()
print(f"\nMax diff Python library vs our Python-layout: {diff:.10f}")
diff2 = np.abs(dequantized - out_ggml).max()
print(f"Max diff Python library vs our GGML-layout: {diff2:.2f}")

# Also check ffn_down
print("\n=== ffn_down.weight ===")
fn = tensors["blk.0.ffn_down.weight"]
flat_fn = fn.data.reshape(-1).astype(np.uint8)
raw_fn = flat_fn[:210]
d_start_fn = np.frombuffer(raw_fn[0:2], dtype=np.float16)[0]
d_end_fn = np.frombuffer(raw_fn[208:210], dtype=np.float16)[0]
print(f"d at start: {d_start_fn}")
print(f"d at end: {d_end_fn}")
