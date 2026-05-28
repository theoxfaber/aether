use std::alloc::{dealloc, Layout};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::sync::Mutex;
use tracing::error;

use crate::inference::runner::{sample, LlamaRunner};
use crate::inference::telemetry::LayerTelemetry;
use crate::Error;

pub const AETHER_OK: i32 = 0;
pub const AETHER_ERR: i32 = -1;
pub const AETHER_TIMEOUT: i32 = -2;

/// Opaque handle to a loaded model, wrapped in a Mutex for thread safety.
pub struct AetherModel {
    inner: Mutex<AetherModelInner>,
}

struct AetherModelInner {
    runner: LlamaRunner,
    frame_budget_ms: f32,
}

// The Mutex provides thread safety; raw pointer access in C is serialized.
unsafe impl Send for AetherModel {}
unsafe impl Sync for AetherModel {}

fn with_model<F, R>(model: *const AetherModel, f: F) -> R
where
    F: FnOnce(&AetherModelInner) -> R,
{
    // SAFETY: caller guarantees model is a valid, non-null pointer.
    let m = unsafe { &*model };
    // Recover from a poisoned mutex rather than panicking across FFI:
    // a previous operation may have panicked while holding the lock.
    let inner = m.inner.lock().unwrap_or_else(|e| e.into_inner());
    f(&inner)
}

fn with_model_mut<F, R>(model: *mut AetherModel, f: F) -> R
where
    F: FnOnce(&mut AetherModelInner) -> R,
{
    // SAFETY: caller guarantees model is a valid, non-null pointer.
    let m = unsafe { &mut *model };
    let mut inner = m.inner.lock().unwrap_or_else(|e| e.into_inner());
    f(&mut inner)
}

/// Load a GGUF model from disk.
#[no_mangle]
pub extern "C" fn aether_load(path: *const c_char) -> *mut AetherModel {
    let path = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    match LlamaRunner::from_gguf(path) {
        Ok(runner) => {
            let inner = AetherModelInner {
                runner,
                frame_budget_ms: 0.0,
            };
            let model = AetherModel {
                inner: Mutex::new(inner),
            };
            Box::into_raw(Box::new(model))
        }
        Err(e) => {
            error!("Load failed: {}", e);
            std::ptr::null_mut()
        }
    }
}

/// Load a GGUF model with streaming / LRU caching.
/// `max_hot` controls how many layers are kept in the LRU cache at once.
/// Pass 0 to let the runner auto-detect.
#[no_mangle]
pub extern "C" fn aether_load_streaming(path: *const c_char, max_hot: i32) -> *mut AetherModel {
    let path = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    let max_hot = if max_hot <= 0 { 32 } else { max_hot as usize };
    match LlamaRunner::from_gguf_streaming(path, max_hot) {
        Ok(runner) => {
            let inner = AetherModelInner {
                runner,
                frame_budget_ms: 0.0,
            };
            let model = AetherModel {
                inner: Mutex::new(inner),
            };
            Box::into_raw(Box::new(model))
        }
        Err(e) => {
            error!("Load streaming failed: {}", e);
            std::ptr::null_mut()
        }
    }
}

/// Free a model loaded by `aether_load`.
/// Safe to call with NULL pointer.
#[no_mangle]
pub extern "C" fn aether_free(model: *mut AetherModel) {
    if !model.is_null() {
        unsafe { drop(Box::from_raw(model)) };
    }
}

// ── Config queries ──────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn aether_vocab_size(model: *const AetherModel) -> i32 {
    with_model(model, |inner| inner.runner.model.config.vocab_size as i32)
}

#[no_mangle]
pub extern "C" fn aether_context_len(model: *const AetherModel) -> i32 {
    with_model(model, |inner| {
        inner.runner.model.config.max_seq_len.min(4096) as i32
    })
}

#[no_mangle]
pub extern "C" fn aether_num_layers(model: *const AetherModel) -> i32 {
    with_model(model, |inner| inner.runner.model.config.num_layers as i32)
}

#[no_mangle]
pub extern "C" fn aether_d_model(model: *const AetherModel) -> i32 {
    with_model(model, |inner| inner.runner.model.config.d_model as i32)
}

#[no_mangle]
pub extern "C" fn aether_eos_id(model: *const AetherModel) -> i32 {
    with_model(model, |inner| inner.runner.tokenizer.eos_id as i32)
}

#[no_mangle]
pub extern "C" fn aether_bos_id(model: *const AetherModel) -> i32 {
    with_model(model, |inner| inner.runner.tokenizer.bos_id as i32)
}

#[no_mangle]
pub extern "C" fn aether_num_gpu_layers(model: *const AetherModel) -> i32 {
    with_model(model, |inner| {
        inner.runner.layer_assignment.gpu_layers as i32
    })
}

#[no_mangle]
pub extern "C" fn aether_num_cpu_layers(model: *const AetherModel) -> i32 {
    with_model(model, |inner| {
        inner.runner.layer_assignment.cpu_layers as i32
    })
}

// ── Frame budget ────────────────────────────────────────────────────────

/// Set a per-decode-step time budget in milliseconds.
/// A value of 0 means no budget (run to completion).
/// When a budget is set, `aether_decode_budgeted` will return `AETHER_TIMEOUT`
/// if the step takes longer than the budget.
#[no_mangle]
pub extern "C" fn aether_set_frame_budget(model: *mut AetherModel, max_ms: f32) {
    with_model_mut(model, |inner| {
        inner.frame_budget_ms = max_ms;
        inner.runner.set_frame_budget(max_ms);
    });
}

#[no_mangle]
pub extern "C" fn aether_frame_budget(model: *const AetherModel) -> f32 {
    with_model(model, |inner| inner.frame_budget_ms)
}

// ── Tokenization ────────────────────────────────────────────────────────

/// Encode text to token ids.
/// Returns a heap-allocated array of `i32` token ids. The caller must free
/// with `aether_free_tokens`. `*out_len` is set to the number of tokens.
/// Returns NULL on failure.
#[no_mangle]
pub extern "C" fn aether_encode(
    model: *const AetherModel,
    text: *const c_char,
    out_len: *mut i32,
) -> *mut i32 {
    let text = match unsafe { CStr::from_ptr(text) }.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    let tokens = with_model(model, |inner| inner.runner.tokenizer.encode(text, true));
    let n = tokens.len();
    unsafe { *out_len = n as i32 };
    let layout = match Layout::array::<i32>(n + 1) {
        Ok(l) => l,
        Err(_) => return std::ptr::null_mut(),
    };
    let ptr = unsafe { std::alloc::alloc(layout) as *mut i32 };
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        *ptr = n as i32;
        for (i, &t) in tokens.iter().enumerate() {
            *ptr.add(1 + i) = t as i32;
        }
        ptr.add(1)
    }
}

/// Decode a single token id to a string.
/// Returns a heap-allocated C string. The caller must free with `aether_free_string`.
#[no_mangle]
pub extern "C" fn aether_decode_token(model: *const AetherModel, token: i32) -> *mut c_char {
    let s = with_model(model, |inner| {
        inner.runner.tokenizer.decode_one(token as u32)
    });
    CString::new(s).unwrap_or_default().into_raw()
}

/// Free a string returned by the API.
#[no_mangle]
pub extern "C" fn aether_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe { drop(CString::from_raw(s)) };
    }
}

/// Free a token array returned by `aether_encode`.
#[no_mangle]
pub extern "C" fn aether_free_tokens(tokens: *mut i32) {
    if !tokens.is_null() {
        unsafe {
            let header = tokens.sub(1);
            let n = *header as usize;
            let layout = Layout::array::<i32>(n + 1).unwrap();
            dealloc(header as *mut u8, layout);
        }
    }
}

// ── Inference ───────────────────────────────────────────────────────────

/// Run prefill on the given token ids.
/// `logits_out` must point to a buffer of at least `aether_vocab_size()` floats.
/// Returns `AETHER_OK` on success, `AETHER_ERR` on failure.
#[no_mangle]
pub extern "C" fn aether_prefill(
    model: *mut AetherModel,
    tokens: *const i32,
    n_tokens: i32,
    logits_out: *mut f32,
) -> i32 {
    with_model_mut(model, |inner| {
        let n = n_tokens as usize;
        let slice = unsafe { std::slice::from_raw_parts(tokens as *const u32, n) };
        match inner.runner.prefill(slice) {
            Ok(logits) => {
                let out = unsafe { std::slice::from_raw_parts_mut(logits_out, logits.len()) };
                out.copy_from_slice(&logits);
                AETHER_OK
            }
            Err(e) => {
                error!("Prefill failed: {}", e);
                AETHER_ERR
            }
        }
    })
}

/// Decode one token.
/// `logits_out` must point to a buffer of at least `aether_vocab_size()` floats.
/// Returns `AETHER_OK` on success, `AETHER_ERR` on failure.
///
/// The caller is responsible for tracking position: `pos` should be
/// `prefill_tokens + decode_step_index`.
#[no_mangle]
pub extern "C" fn aether_decode(
    model: *mut AetherModel,
    token: i32,
    pos: i32,
    logits_out: *mut f32,
) -> i32 {
    with_model_mut(model, |inner| {
        let n_layers = inner.runner.model.config.num_layers;
        let mut tel = vec![LayerTelemetry::default(); n_layers];
        match inner
            .runner
            .decode_step(token as u32, pos as usize, &mut tel)
        {
            Ok(logits) => {
                let out = unsafe { std::slice::from_raw_parts_mut(logits_out, logits.len()) };
                out.copy_from_slice(&logits);
                AETHER_OK
            }
            Err(e) => {
                error!("Decode failed: {}", e);
                AETHER_ERR
            }
        }
    })
}

/// Decode one token with frame budget.
///
/// If the decode step exceeds `model.frame_budget_ms`, the partial state is
/// saved and `AETHER_TIMEOUT` is returned WITHOUT writing logits. The caller
/// should retry with the same `token` and `pos` to resume from where it left
/// off. When the decode eventually completes, `AETHER_OK` is returned and
/// `logits_out` contains the full logits.
///
/// If no budget was set (or budget ≤ 0), this behaves exactly like
/// `aether_decode`.
#[no_mangle]
pub extern "C" fn aether_decode_budgeted(
    model: *mut AetherModel,
    token: i32,
    pos: i32,
    logits_out: *mut f32,
) -> i32 {
    with_model_mut(model, |inner| {
        // Sync the runner's frame budget from the C API field
        inner.runner.set_frame_budget(inner.frame_budget_ms);

        let n_layers = inner.runner.model.config.num_layers;
        let mut tel = vec![LayerTelemetry::default(); n_layers];
        match inner
            .runner
            .decode_step_budgeted(token as u32, pos as usize, &mut tel)
        {
            Ok(logits) => {
                let out = unsafe { std::slice::from_raw_parts_mut(logits_out, logits.len()) };
                out.copy_from_slice(&logits);
                AETHER_OK
            }
            Err(Error::BudgetExceeded(_n)) => {
                // Partial decode saved in runner; caller must retry
                AETHER_TIMEOUT
            }
            Err(e) => {
                error!("Decode failed: {}", e);
                AETHER_ERR
            }
        }
    })
}

// ── Sampling ────────────────────────────────────────────────────────────

/// Sample a token from logits using temperature.
/// - `temperature` = 0.0 → greedy (argmax)
/// - `temperature` > 0.0 → softmax sampling
/// - `top_p` = 1.0 → no nucleus filtering
///
/// Returns the sampled token id, or -1 on error.
#[no_mangle]
pub extern "C" fn aether_sample(
    model: *const AetherModel,
    logits: *const f32,
    temperature: f32,
    top_p: f32,
) -> i32 {
    with_model(model, |inner| {
        let vocab = inner.runner.model.config.vocab_size;
        let logits = unsafe { std::slice::from_raw_parts(logits, vocab) };
        let token = sample(logits, temperature, top_p, &[], 1.0);
        token as i32
    })
}

/// Greedy sample (argmax) from logits.
/// Returns the token with the highest probability.
#[no_mangle]
pub extern "C" fn aether_argmax(model: *const AetherModel, logits: *const f32) -> i32 {
    with_model(model, |inner| {
        let vocab = inner.runner.model.config.vocab_size;
        let logits = unsafe { std::slice::from_raw_parts(logits, vocab) };
        let token = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        token as i32
    })
}

// ── Error message ───────────────────────────────────────────────────────

/// Get the last error message. Returns a heap-allocated C string.
/// The caller must free with `aether_free_string`.
#[no_mangle]
pub extern "C" fn aether_last_error() -> *mut c_char {
    CString::new("See stderr for error details")
        .unwrap_or_default()
        .into_raw()
}
