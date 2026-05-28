pub(crate) const Q8_BLOCK_SIZE: usize = 32;
pub(crate) const Q8_BLOCK_BYTES: usize = 34;

pub(crate) const Q4K_BLOCK_SIZE: usize = 256;
pub(crate) const Q4K_BLOCK_BYTES: usize = 144;

pub(crate) const Q6K_BLOCK_SIZE: usize = 256;
pub(crate) const Q6K_BLOCK_BYTES: usize = 210;

pub(crate) const Q5K_BLOCK_SIZE: usize = 256;
pub(crate) const Q5K_BLOCK_BYTES: usize = 176;

pub(crate) const Q2K_BLOCK_SIZE: usize = 256;
pub(crate) const Q2K_BLOCK_BYTES: usize = 84;

pub(crate) const Q3K_BLOCK_SIZE: usize = 256;
pub(crate) const Q3K_BLOCK_BYTES: usize = 110;

pub(crate) fn decode_f16_scale(lo: u8, hi: u8) -> f32 {
    let bits = u16::from_le_bytes([lo, hi]);
    let f = half::f16::from_bits(bits).to_f32();
    if f.is_nan() || f.is_infinite() {
        bits as f32 / 65536.0
    } else {
        f
    }
}

#[inline(always)]
pub(crate) fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let mm = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (sc, mm)
    }
}
