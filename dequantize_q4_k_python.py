import gguf
import numpy as np

def decode_d(val):
    # Convert uint16 bits to float16 then float32
    # In python, we can use np.frombuffer to decode f16
    f = np.array([val], dtype=np.uint16).view(np.float16)[0]
    if np.isnan(f) or np.isinf(f):
        return float(val) / 65536.0
    return float(f)

def get_scale_min_k4(j, scales):
    if j < 4:
        return scales[j] & 63, scales[j + 4] & 63
    else:
        sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4)
        mm = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)
        return sc, mm

def dequant_q4_k(data, shape):
    cols, rows = shape[0], shape[1]
    n = cols * rows
    num_blocks = n // 256
    out = np.zeros(n, dtype=np.float32)
    
    for bi in range(num_blocks):
        bo = bi * 144
        d = decode_d(np.frombuffer(data[bo:bo+2], dtype=np.uint16)[0])
        dmin = decode_d(np.frombuffer(data[bo+2:bo+4], dtype=np.uint16)[0])
        
        scales = data[bo+4:bo+16]
        qs = data[bo+16:bo+144]
        
        is_val = 0
        for j in range(0, 256, 64):
            sc0, mm0 = get_scale_min_k4(is_val, scales)
            d1 = d * sc0
            m1 = dmin * mm0
            sc1, mm1 = get_scale_min_k4(is_val + 1, scales)
            d2 = d * sc1
            m2 = dmin * mm1
            
            for l in range(32):
                out[bi * 256 + is_val * 32 + l] = d1 * (qs[l] & 0x0F) - m1
                out[bi * 256 + (is_val + 1) * 32 + l] = d2 * ((qs[l] >> 4) & 0x0F) - m2
            
            qs = qs[32:]
            is_val += 2
            
    return out

reader = gguf.GGUFReader("tinyllama-q4.gguf")
for tensor in reader.tensors:
    if tensor.name == "token_embd.weight":
        raw_bytes = bytes(tensor.data)
        deq = dequant_q4_k(raw_bytes, tensor.shape)
        # Reshape to [32000, 2048]
        deq_matrix = deq.reshape((tensor.shape[1], tensor.shape[0]))
        print("token_embd.weight shape:", deq_matrix.shape)
        print("Row 0 first 5:", deq_matrix[0, :5].tolist())
        print("Row 1 first 5:", deq_matrix[1, :5].tolist())
        print("Row 2 first 5:", deq_matrix[2, :5].tolist())
        
    if tensor.name == "blk.0.attn_q.weight":
        raw_bytes = bytes(tensor.data)
        deq = dequant_q4_k(raw_bytes, tensor.shape)
        deq_matrix = deq.reshape((tensor.shape[1], tensor.shape[0]))
        print("blk.0.attn_q.weight shape:", deq_matrix.shape)
        print("Row 0 first 5:", deq_matrix[0, :5].tolist())
