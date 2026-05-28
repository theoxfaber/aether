#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]
use aether::loader::dequant::dequantize;
/// Debug: prefill all 22 layers, quantized vs f32 reference.
use aether::loader::gguf::GGUFLoader;
use aether::quant::quantized_matmul_impl;

const D: usize = 2048;
const NH: usize = 32;
const NKV: usize = 4;
const HD: usize = 64;
const DFF: usize = 5632;
const NL: usize = 22;
const EPS: f32 = 1e-5;

fn main() {
    let gguf = GGUFLoader::load("tinyllama-q4.gguf").unwrap();
    let tensors = &gguf.tensors;
    let tokens = [1u32, 450, 7483, 310, 3444, 338];
    let _n = tokens.len();

    let w = load_all_f32(tensors);

    let (logits_q, _) = prefill_quant(tensors, &tokens);
    let (logits_r, _) = prefill_f32(&w, &tokens);

    eprintln!("\n=== Final logits comparison ===");
    eprintln!("quant top-5: {:?}", top_k(&logits_q, 5));
    eprintln!("ref   top-5: {:?}", top_k(&logits_r, 5));
    eprintln!("max_diff: {:.10}", max_diff(&logits_q, &logits_r));
}

fn prefill_quant(
    tensors: &std::collections::HashMap<String, aether::loader::gguf::GGUFTensor>,
    tokens: &[u32],
) -> (Vec<f32>, Vec<f32>) {
    let n = tokens.len();
    let max_seq = 512usize;
    let mut kv = vec![0.0f32; NL * 2 * NKV * max_seq * HD];
    let kv_ls = 2 * NKV * max_seq * HD;
    let kv_hs = max_seq * HD;

    let emb_t = tensors.get("token_embd.weight").unwrap();
    let emb = dequantize(&emb_t.data, emb_t.dtype, &emb_t.shape);
    let mut bx = vec![0.0f32; n * D];
    for (i, &t) in tokens.iter().enumerate() {
        bx[i * D..(i + 1) * D].copy_from_slice(&emb[(t as usize) * D..(t as usize) * D + D]);
    }

    let (rsin, rcos) = precomp_rope(max_seq, HD, 10000.0);
    let mut bn = vec![0.0f32; n * D];
    let mut bq = vec![0.0f32; n * NH * HD];
    let mut bk = vec![0.0f32; n * NKV * HD];
    let mut bv = vec![0.0f32; n * NKV * HD];
    let mut ba = vec![0.0f32; n * NH * HD];
    let mut sc = vec![0.0f32; n * n];
    let mut bp = vec![0.0f32; n * D];
    let mut bg = vec![0.0f32; n * DFF];
    let mut bu = vec![0.0f32; n * DFF];
    let mut bm = vec![0.0f32; n * D];

    for layer in 0..NL {
        let tn = |s: &str| -> String { format!("blk.{}.{}", layer, s) };

        let nw = deq(tensors, &tn("attn_norm.weight"));
        let fw = deq(tensors, &tn("ffn_norm.weight"));

        let (tq, tk, tv, to, tg, tu, td) = (
            tensors.get(&tn("attn_q.weight")).unwrap(),
            tensors.get(&tn("attn_k.weight")).unwrap(),
            tensors.get(&tn("attn_v.weight")).unwrap(),
            tensors.get(&tn("attn_output.weight")).unwrap(),
            tensors.get(&tn("ffn_gate.weight")).unwrap(),
            tensors.get(&tn("ffn_up.weight")).unwrap(),
            tensors.get(&tn("ffn_down.weight")).unwrap(),
        );

        // RMSNorm
        for i in 0..n {
            rms(
                &bx[i * D..(i + 1) * D],
                &nw,
                EPS,
                &mut bn[i * D..(i + 1) * D],
            );
        }

        // QKV
        for i in 0..n {
            let xi = &bn[i * D..(i + 1) * D];
            qm(xi, tq, &mut bq[i * NH * HD..(i + 1) * NH * HD]);
            qm(xi, tk, &mut bk[i * NKV * HD..(i + 1) * NKV * HD]);
            qm(xi, tv, &mut bv[i * NKV * HD..(i + 1) * NKV * HD]);
        }

        // RoPE
        for i in 0..n {
            rope(
                &mut bq[i * NH * HD..(i + 1) * NH * HD],
                NH,
                HD,
                i,
                &rsin,
                &rcos,
            );
            rope(
                &mut bk[i * NKV * HD..(i + 1) * NKV * HD],
                NKV,
                HD,
                i,
                &rsin,
                &rcos,
            );
        }

        // KV cache
        let lb = layer * kv_ls;
        for i in 0..n {
            for h in 0..NKV {
                let ks = &bk[(i * NKV + h) * HD..(i * NKV + h + 1) * HD];
                kv[lb + h * kv_hs + i * HD..lb + h * kv_hs + (i + 1) * HD].copy_from_slice(ks);
                let vs = &bv[(i * NKV + h) * HD..(i * NKV + h + 1) * HD];
                kv[lb + NKV * kv_hs + h * kv_hs + i * HD
                    ..lb + NKV * kv_hs + h * kv_hs + (i + 1) * HD]
                    .copy_from_slice(vs);
            }
        }

        // Attention
        attn_batch(&bq, &bk, &bv, n, NH, NKV, HD, &mut ba, &mut sc, None);

        // Output
        for i in 0..n {
            qm(
                &ba[i * NH * HD..(i + 1) * NH * HD],
                to,
                &mut bp[i * D..(i + 1) * D],
            );
        }
        for i in 0..n {
            for j in 0..D {
                bx[i * D + j] += bp[i * D + j];
            }
        }

        // FFN
        for i in 0..n {
            rms(
                &bx[i * D..(i + 1) * D],
                &fw,
                EPS,
                &mut bn[i * D..(i + 1) * D],
            );
        }
        for i in 0..n {
            qm(&bn[i * D..(i + 1) * D], tg, &mut bg[i * DFF..(i + 1) * DFF]);
            qm(&bn[i * D..(i + 1) * D], tu, &mut bu[i * DFF..(i + 1) * DFF]);
            for j in 0..DFF {
                bg[i * DFF + j] = bg[i * DFF + j] * sig(bg[i * DFF + j]) * bu[i * DFF + j];
            }
            qm(&bg[i * DFF..(i + 1) * DFF], td, &mut bm[i * D..(i + 1) * D]);
        }
        for i in 0..n {
            for j in 0..D {
                bx[i * D + j] += bm[i * D + j];
            }
        }
    }

    // Final norm + LM head
    let onw = deq(tensors, "output_norm.weight");
    let mut fnm = vec![0.0f32; D];
    rms(&bx[(n - 1) * D..n * D], &onw, EPS, &mut fnm);

    let lh = tensors.get("output.weight").unwrap();
    let mut logits = vec![0.0f32; 32000];
    qm(&fnm, lh, &mut logits);
    (logits, bx)
}

fn prefill_f32(
    w: &std::collections::HashMap<String, Vec<f32>>,
    tokens: &[u32],
) -> (Vec<f32>, Vec<f32>) {
    let n = tokens.len();
    let max_seq = 512usize;
    let mut kv = vec![0.0f32; NL * 2 * NKV * max_seq * HD];
    let kv_ls = 2 * NKV * max_seq * HD;
    let kv_hs = max_seq * HD;

    let emb = w.get("token_embd.weight").unwrap();
    let mut bx = vec![0.0f32; n * D];
    for (i, &t) in tokens.iter().enumerate() {
        bx[i * D..(i + 1) * D].copy_from_slice(&emb[(t as usize) * D..(t as usize) * D + D]);
    }

    let (rsin, rcos) = precomp_rope(max_seq, HD, 10000.0);
    let mut bn = vec![0.0f32; n * D];
    let mut bq = vec![0.0f32; n * NH * HD];
    let mut bk = vec![0.0f32; n * NKV * HD];
    let mut bv = vec![0.0f32; n * NKV * HD];
    let mut ba = vec![0.0f32; n * NH * HD];
    let mut sc = vec![0.0f32; n * n];
    let mut bp = vec![0.0f32; n * D];
    let mut bg = vec![0.0f32; n * DFF];
    let mut bu = vec![0.0f32; n * DFF];
    let mut bm = vec![0.0f32; n * D];

    for layer in 0..NL {
        let tn = |s: &str| -> String { format!("blk.{}.{}", layer, s) };

        let nw = w.get(&tn("attn_norm.weight")).unwrap();
        let fw = w.get(&tn("ffn_norm.weight")).unwrap();
        let bq_w = w.get(&tn("attn_q.weight")).unwrap();
        let bk_w = w.get(&tn("attn_k.weight")).unwrap();
        let bv_w = w.get(&tn("attn_v.weight")).unwrap();
        let bo_w = w.get(&tn("attn_output.weight")).unwrap();
        let bg_w = w.get(&tn("ffn_gate.weight")).unwrap();
        let bu_w = w.get(&tn("ffn_up.weight")).unwrap();
        let bd_w = w.get(&tn("ffn_down.weight")).unwrap();

        for i in 0..n {
            rms(
                &bx[i * D..(i + 1) * D],
                nw,
                EPS,
                &mut bn[i * D..(i + 1) * D],
            );
        }

        for i in 0..n {
            let xi = &bn[i * D..(i + 1) * D];
            mf32(
                xi,
                bq_w,
                1,
                NH * HD,
                D,
                &mut bq[i * NH * HD..(i + 1) * NH * HD],
            );
            mf32(
                xi,
                bk_w,
                1,
                NKV * HD,
                D,
                &mut bk[i * NKV * HD..(i + 1) * NKV * HD],
            );
            mf32(
                xi,
                bv_w,
                1,
                NKV * HD,
                D,
                &mut bv[i * NKV * HD..(i + 1) * NKV * HD],
            );
        }

        for i in 0..n {
            rope(
                &mut bq[i * NH * HD..(i + 1) * NH * HD],
                NH,
                HD,
                i,
                &rsin,
                &rcos,
            );
            rope(
                &mut bk[i * NKV * HD..(i + 1) * NKV * HD],
                NKV,
                HD,
                i,
                &rsin,
                &rcos,
            );
        }

        let lb = layer * kv_ls;
        for i in 0..n {
            for h in 0..NKV {
                let ks = &bk[(i * NKV + h) * HD..(i * NKV + h + 1) * HD];
                kv[lb + h * kv_hs + i * HD..lb + h * kv_hs + (i + 1) * HD].copy_from_slice(ks);
                let vs = &bv[(i * NKV + h) * HD..(i * NKV + h + 1) * HD];
                kv[lb + NKV * kv_hs + h * kv_hs + i * HD
                    ..lb + NKV * kv_hs + h * kv_hs + (i + 1) * HD]
                    .copy_from_slice(vs);
            }
        }

        attn_batch(&bq, &bk, &bv, n, NH, NKV, HD, &mut ba, &mut sc, None);

        for i in 0..n {
            mf32(
                &ba[i * NH * HD..(i + 1) * NH * HD],
                bo_w,
                1,
                D,
                D,
                &mut bp[i * D..(i + 1) * D],
            );
        }
        for i in 0..n {
            for j in 0..D {
                bx[i * D + j] += bp[i * D + j];
            }
        }

        for i in 0..n {
            rms(
                &bx[i * D..(i + 1) * D],
                fw,
                EPS,
                &mut bn[i * D..(i + 1) * D],
            );
        }
        for i in 0..n {
            mf32(
                &bn[i * D..(i + 1) * D],
                bg_w,
                1,
                DFF,
                D,
                &mut bg[i * DFF..(i + 1) * DFF],
            );
            mf32(
                &bn[i * D..(i + 1) * D],
                bu_w,
                1,
                DFF,
                D,
                &mut bu[i * DFF..(i + 1) * DFF],
            );
            for j in 0..DFF {
                bg[i * DFF + j] = bg[i * DFF + j] * sig(bg[i * DFF + j]) * bu[i * DFF + j];
            }
            mf32(
                &bg[i * DFF..(i + 1) * DFF],
                bd_w,
                1,
                D,
                DFF,
                &mut bm[i * D..(i + 1) * D],
            );
        }
        for i in 0..n {
            for j in 0..D {
                bx[i * D + j] += bm[i * D + j];
            }
        }
    }

    let onw = w.get("output_norm.weight").unwrap();
    let mut fnm = vec![0.0f32; D];
    rms(&bx[(n - 1) * D..n * D], onw, EPS, &mut fnm);

    let lh_w = w.get("output.weight").unwrap();
    let mut logits = vec![0.0f32; 32000];
    mf32(&fnm, lh_w, 1, 32000, D, &mut logits);
    (logits, bx)
}

// ─── Helpers ───

fn deq(
    t: &std::collections::HashMap<String, aether::loader::gguf::GGUFTensor>,
    n: &str,
) -> Vec<f32> {
    let tn = t.get(n).unwrap();
    dequantize(&tn.data, tn.dtype, &tn.shape)
}

fn qm(a: &[f32], t: &aether::loader::gguf::GGUFTensor, c: &mut [f32]) {
    quantized_matmul_impl(
        a,
        1,
        &t.data,
        &[t.shape[1], t.shape[0]],
        t.dtype,
        c,
        None,
    )
}

fn mf32(a: &[f32], b: &[f32], _m: usize, n: usize, k: usize, c: &mut [f32]) {
    c.fill(0.0);
    for i in 0..k {
        let s = a[i];
        if s != 0.0 {
            for j in 0..n {
                c[j] = s.mul_add(b[i * n + j], c[j]);
            }
        }
    }
}

fn rms(x: &[f32], w: &[f32], eps: f32, o: &mut [f32]) {
    let n = x.len() as f32;
    let ss: f32 = x.iter().map(|&v| v * v).sum();
    let r = (ss / n + eps).sqrt();
    for (i, (&xi, &wi)) in x.iter().zip(w.iter()).enumerate() {
        o[i] = xi / r * wi;
    }
}

fn precomp_rope(ms: usize, hd: usize, base: f32) -> (Vec<f32>, Vec<f32>) {
    let h = hd / 2;
    let mut s = vec![0.0; ms * h];
    let mut c = vec![0.0; ms * h];
    for p in 0..ms {
        for i in 0..h {
            let t = (p as f32) * base.powf(-2.0 * i as f32 / hd as f32);
            s[p * h + i] = t.sin();
            c[p * h + i] = t.cos();
        }
    }
    (s, c)
}

fn rope(x: &mut [f32], nh: usize, hd: usize, pos: usize, sin: &[f32], cos: &[f32]) {
    let h = hd / 2;
    let ms = sin.len() / h;
    let cp = pos.min(ms - 1);
    let sr = &sin[cp * h..(cp + 1) * h];
    let cr = &cos[cp * h..(cp + 1) * h];
    for hh in 0..nh {
        let head = &mut x[hh * hd..(hh + 1) * hd];
        for i in 0..h {
            let x0 = head[i];
            let x1 = head[i + h];
            head[i] = x0 * cr[i] - x1 * sr[i];
            head[i + h] = x0 * sr[i] + x1 * cr[i];
        }
    }
}

fn attn_batch(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    out: &mut [f32],
    scores: &mut [f32],
    _sw: Option<usize>,
) {
    let scale = 1.0 / (hd as f32).sqrt();
    let kg = nh / nkv;
    let qs = nh * hd;
    let ks = nkv * hd;
    for h in 0..nh {
        let kv_h = h / kg;
        for i in 0..n {
            let qr = &q[i * qs + h * hd..][..hd];
            let sr = &mut scores[i * n..(i + 1) * n];
            for j in 0..=i {
                let kr = &k[j * ks + kv_h * hd..][..hd];
                sr[j] = qr.iter().zip(kr.iter()).map(|(&a, &b)| a * b).sum::<f32>() * scale;
            }
            for j in i + 1..n {
                sr[j] = f32::NEG_INFINITY;
            }
        }
        for i in 0..n {
            let sr = &mut scores[i * n..(i + 1) * n];
            let mv = sr.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for s in sr.iter_mut() {
                if *s > f32::NEG_INFINITY / 2.0 {
                    *s = (*s - mv).exp();
                    sum += *s;
                }
            }
            for s in sr.iter_mut() {
                if *s > 0.0 {
                    *s /= sum;
                }
            }
        }
        for i in 0..n {
            let sr = &scores[i * n..(i + 1) * n];
            let or = &mut out[i * qs + h * hd..][..hd];
            or.fill(0.0);
            for j in 0..=i {
                let w = sr[j];
                if w <= 0.0 {
                    continue;
                }
                let vr = &v[j * ks + kv_h * hd..][..hd];
                for d in 0..hd {
                    or[d] += w * vr[d];
                }
            }
        }
    }
}

fn sig(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}
fn top_k(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut v: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, v)| (i, *v)).collect();
    v.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    v.iter().take(k).map(|(i, v)| (*i as u32, *v)).collect()
}

fn load_all_f32(
    tensors: &std::collections::HashMap<String, aether::loader::gguf::GGUFTensor>,
) -> std::collections::HashMap<String, Vec<f32>> {
    let mut w = std::collections::HashMap::new();
    for (name, tensor) in tensors {
        w.insert(
            name.clone(),
            dequantize(&tensor.data, tensor.dtype, &tensor.shape),
        );
    }
    w
}
