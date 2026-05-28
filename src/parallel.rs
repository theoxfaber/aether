use rayon::prelude::*;

pub fn parallel_map(data: &[f32], f: impl Fn(f32) -> f32 + Send + Sync) -> Vec<f32> {
    if data.len() < 4096 {
        data.iter().copied().map(f).collect()
    } else {
        data.par_iter().copied().map(f).collect()
    }
}

pub fn parallel_map2(a: &[f32], b: &[f32], f: impl Fn(f32, f32) -> f32 + Send + Sync) -> Vec<f32> {
    if a.len() < 4096 {
        a.iter()
            .copied()
            .zip(b.iter().copied())
            .map(|(x, y)| f(x, y))
            .collect()
    } else {
        a.par_iter()
            .copied()
            .zip(b.par_iter().copied())
            .map(|(x, y)| f(x, y))
            .collect()
    }
}

pub fn parallel_map_inplace(data: &[f32], out: &mut [f32], f: impl Fn(f32) -> f32 + Send + Sync) {
    assert_eq!(data.len(), out.len());
    if data.len() < 4096 {
        for (i, &x) in data.iter().enumerate() {
            out[i] = f(x);
        }
    } else {
        out.par_iter_mut().enumerate().for_each(|(i, val)| {
            *val = f(data[i]);
        });
    }
}

pub fn parallel_map2_inplace(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    f: impl Fn(f32, f32) -> f32 + Send + Sync,
) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    if a.len() < 4096 {
        for i in 0..a.len() {
            out[i] = f(a[i], b[i]);
        }
    } else {
        out.par_iter_mut().enumerate().for_each(|(i, val)| {
            *val = f(a[i], b[i]);
        });
    }
}
