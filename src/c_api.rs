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

/// Upper bound for token counts accepted across the FFI boundary.
/// A negative `i32` would otherwise wrap to a huge `usize` and create an
/// out-of-bounds slice (undefined behavior). Counts above this are rejected.
const MAX_FFI_TOKENS: usize = 1 << 20; // ~1M tokens

/// Upper bound trusted for the header length in `aether_free_tokens`.
const MAX_FFI_FREE_LEN: usize = 1 << 28;

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

/// Run `f`, converting any Rust panic into `default` instead of unwinding
/// across the FFI boundary (which is undefined behavior for `extern "C"`).
/// Every public entry point below is wrapped in this guard.
fn ffi_catch<R>(default: R, f: impl FnOnce() -> R) -> R {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(v) => v,
        Err(_) => {
            error!("Aether FFI call panicked; returning error instead of unwinding");
            default
        }
    }
}

/// Validate a token count coming from C: must be positive and bounded.
/// Rejects negatives (which would wrap with `as usize`) and absurd sizes.
fn check_count(n: i32) -> Option<usize> {
    if n <= 0 {
        return None;
    }
    let n = n as usize;
    if n > MAX_FFI_TOKENS {
        return None;
    }
    Some(n)
}

fn with_model<F, R>(model: *const AetherModel, f: F) -> R
where
    F: FnOnce(&AetherModelInner) -> R,
{
    // SAFETY: every `extern "C"` entry point null-checks `model` before
    // calling this helper, so `model` is non-null and points to a live
    // `Box<AetherModel>` (freed only via `aether_free`).
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
    // SAFETY: see `with_model` — entry points null-check first.
    let m = unsafe { &mut *model };
    let mut inner = m.inner.lock().unwrap_or_else(|e| e.into_inner());
    f(&mut inner)
}

/// Load a GGUF model from disk.
/// Returns NULL on failure (including NULL `path`). Never panics.
#[no_mangle]
pub extern "C" fn aether_load(path: *const c_char) -> *mut AetherModel {
    ffi_catch(std::ptr::null_mut(), || {
        if path.is_null() {
            return std::ptr::null_mut();
        }
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
    })
}

/// Load a GGUF model with streaming / LRU caching.
/// `max_hot` controls how many layers are kept in the LRU cache at once.
/// Pass 0 to let the runner auto-detect.
/// Returns NULL on failure (including NULL `path`). Never panics.
#[no_mangle]
pub extern "C" fn aether_load_streaming(path: *const c_char, max_hot: i32) -> *mut AetherModel {
    ffi_catch(std::ptr::null_mut(), || {
        if path.is_null() {
            return std::ptr::null_mut();
        }
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
    })
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
// All getters return -1 on NULL model instead of dereferencing it.

#[no_mangle]
pub extern "C" fn aether_vocab_size(model: *const AetherModel) -> i32 {
    if model.is_null() {
        return -1;
    }
    ffi_catch(-1, || {
        with_model(model, |inner| {
            inner.runner.ctx.model.config.vocab_size as i32
        })
    })
}

#[no_mangle]
pub extern "C" fn aether_context_len(model: *const AetherModel) -> i32 {
    if model.is_null() {
        return -1;
    }
    ffi_catch(-1, || {
        with_model(model, |inner| {
            inner.runner.ctx.model.config.max_seq_len.min(4096) as i32
        })
    })
}

#[no_mangle]
pub extern "C" fn aether_num_layers(model: *const AetherModel) -> i32 {
    if model.is_null() {
        return -1;
    }
    ffi_catch(-1, || {
        with_model(model, |inner| {
            inner.runner.ctx.model.config.num_layers as i32
        })
    })
}

#[no_mangle]
pub extern "C" fn aether_d_model(model: *const AetherModel) -> i32 {
    if model.is_null() {
        return -1;
    }
    ffi_catch(-1, || {
        with_model(model, |inner| inner.runner.ctx.model.config.d_model as i32)
    })
}

#[no_mangle]
pub extern "C" fn aether_eos_id(model: *const AetherModel) -> i32 {
    if model.is_null() {
        return -1;
    }
    ffi_catch(-1, || {
        with_model(model, |inner| inner.runner.tokenizer.eos_id as i32)
    })
}

#[no_mangle]
pub extern "C" fn aether_bos_id(model: *const AetherModel) -> i32 {
    if model.is_null() {
        return -1;
    }
    ffi_catch(-1, || {
        with_model(model, |inner| inner.runner.tokenizer.bos_id as i32)
    })
}

#[no_mangle]
pub extern "C" fn aether_num_gpu_layers(model: *const AetherModel) -> i32 {
    if model.is_null() {
        return -1;
    }
    ffi_catch(-1, || {
        with_model(model, |inner| {
            inner.runner.layer_assignment.gpu_layers as i32
        })
    })
}

#[no_mangle]
pub extern "C" fn aether_num_cpu_layers(model: *const AetherModel) -> i32 {
    if model.is_null() {
        return -1;
    }
    ffi_catch(-1, || {
        with_model(model, |inner| {
            inner.runner.layer_assignment.cpu_layers as i32
        })
    })
}

// ── Frame budget ────────────────────────────────────────────────────────

/// Set a per-decode-step time budget in milliseconds.
/// A value of 0 means no budget (run to completion).
/// When a budget is set, `aether_decode_budgeted` will return `AETHER_TIMEOUT`
/// if the step takes longer than the budget.
#[no_mangle]
pub extern "C" fn aether_set_frame_budget(model: *mut AetherModel, max_ms: f32) {
    if model.is_null() {
        return;
    }
    ffi_catch((), || {
        with_model_mut(model, |inner| {
            inner.frame_budget_ms = max_ms;
            inner.runner.set_frame_budget(max_ms);
        })
    });
}

#[no_mangle]
pub extern "C" fn aether_frame_budget(model: *const AetherModel) -> f32 {
    if model.is_null() {
        return 0.0;
    }
    ffi_catch(0.0, || with_model(model, |inner| inner.frame_budget_ms))
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
    ffi_catch(std::ptr::null_mut(), || {
        if model.is_null() || text.is_null() || out_len.is_null() {
            return std::ptr::null_mut();
        }
        let text = match unsafe { CStr::from_ptr(text) }.to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        };
        let tokens = with_model(model, |inner| inner.runner.tokenizer.encode(text, true));
        let n = tokens.len();
        // Header layout stores `n` as i32; refuse counts that don't fit.
        let Ok(n_i32) = i32::try_from(n) else {
            return std::ptr::null_mut();
        };
        let layout = match Layout::array::<i32>(n.saturating_add(1)) {
            Ok(l) => l,
            Err(_) => return std::ptr::null_mut(),
        };
        let ptr = unsafe { std::alloc::alloc(layout) as *mut i32 };
        if ptr.is_null() {
            return std::ptr::null_mut();
        }
        unsafe {
            *ptr = n_i32;
            for (i, &t) in tokens.iter().enumerate() {
                *ptr.add(1 + i) = t as i32;
            }
            // Only publish the length after the buffer is fully written.
            *out_len = n_i32;
            ptr.add(1)
        }
    })
}

/// Decode a single token id to a string.
/// Returns a heap-allocated C string. The caller must free with `aether_free_string`.
#[no_mangle]
pub extern "C" fn aether_decode_token(model: *const AetherModel, token: i32) -> *mut c_char {
    ffi_catch(std::ptr::null_mut(), || {
        if model.is_null() {
            return std::ptr::null_mut();
        }
        let Ok(id) = u32::try_from(token) else {
            return std::ptr::null_mut();
        };
        let s = with_model(model, |inner| inner.runner.tokenizer.decode_one(id));
        CString::new(s).unwrap_or_default().into_raw()
    })
}

/// Free a string returned by the API.
#[no_mangle]
pub extern "C" fn aether_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe { drop(CString::from_raw(s)) };
    }
}

/// Free a token array returned by `aether_encode`.
///
/// The length header is trusted but sanity-checked: a bogus length fails
/// the bound check and is ignored (leaking rather than corrupting the heap).
/// Passing a pointer that did not come from `aether_encode` is still a
/// caller bug — free with the matching allocator only.
#[no_mangle]
pub extern "C" fn aether_free_tokens(tokens: *mut i32) {
    if tokens.is_null() {
        return;
    }
    ffi_catch((), || {
        unsafe {
            let header = tokens.sub(1);
            let n = *header as usize;
            if n > MAX_FFI_FREE_LEN {
                error!("aether_free_tokens: implausible length header, ignoring");
                return;
            }
            let Ok(layout) = Layout::array::<i32>(n.saturating_add(1)) else {
                return;
            };
            // Zero the header so a double-free is caught by the bound check.
            *header = 0;
            dealloc(header as *mut u8, layout);
        }
    });
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
    ffi_catch(AETHER_ERR, || {
        let Some(n) = check_count(n_tokens) else {
            return AETHER_ERR;
        };
        if model.is_null() || tokens.is_null() || logits_out.is_null() {
            return AETHER_ERR;
        }
        // Token ids are non-negative; the bit pattern is reinterpreted, so
        // validate the range explicitly instead of `as u32` wrapping.
        let slice = unsafe { std::slice::from_raw_parts(tokens, n) };
        if slice.iter().any(|&t| t < 0) {
            return AETHER_ERR;
        }
        let ids: Vec<u32> = slice.iter().map(|&t| t as u32).collect();
        with_model_mut(model, |inner| {
            // Reject token ids outside the vocabulary: `forward_batch`
            // indexes `token_embeddings` directly with the id.
            let vocab = inner.runner.ctx.model.config.vocab_size as u32;
            if ids.iter().any(|&t| t >= vocab) {
                error!("Prefill failed: token id out of vocabulary range");
                return AETHER_ERR;
            }
            match inner.runner.prefill(&ids) {
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
    ffi_catch(AETHER_ERR, || {
        let (Some(id), Some(position)) = (u32::try_from(token).ok(), usize::try_from(pos).ok())
        else {
            return AETHER_ERR;
        };
        if model.is_null() || logits_out.is_null() {
            return AETHER_ERR;
        }
        with_model_mut(model, |inner| {
            if id >= inner.runner.ctx.model.config.vocab_size as u32 {
                error!("Decode failed: token id out of vocabulary range");
                return AETHER_ERR;
            }
            let n_layers = inner.runner.ctx.model.config.num_layers;
            let mut tel = vec![LayerTelemetry::default(); n_layers];
            match inner.runner.decode_step(id, position, &mut tel) {
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
    ffi_catch(AETHER_ERR, || {
        let (Some(id), Some(position)) = (u32::try_from(token).ok(), usize::try_from(pos).ok())
        else {
            return AETHER_ERR;
        };
        if model.is_null() || logits_out.is_null() {
            return AETHER_ERR;
        }
        with_model_mut(model, |inner| {
            if id >= inner.runner.ctx.model.config.vocab_size as u32 {
                error!("Decode failed: token id out of vocabulary range");
                return AETHER_ERR;
            }
            // Sync the runner's frame budget from the C API field
            inner.runner.set_frame_budget(inner.frame_budget_ms);

            let n_layers = inner.runner.ctx.model.config.num_layers;
            let mut tel = vec![LayerTelemetry::default(); n_layers];
            match inner.runner.decode_step_budgeted(id, position, &mut tel) {
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
    ffi_catch(-1, || {
        if model.is_null() || logits.is_null() {
            return -1;
        }
        with_model(model, |inner| {
            let vocab = inner.runner.ctx.model.config.vocab_size;
            let logits = unsafe { std::slice::from_raw_parts(logits, vocab) };
            let prev_set = std::collections::HashSet::new();
            let token = sample(logits, temperature, top_p, &prev_set, 1.0);
            token as i32
        })
    })
}

/// Greedy sample (argmax) from logits.
/// Returns the token with the highest probability.
#[no_mangle]
pub extern "C" fn aether_argmax(model: *const AetherModel, logits: *const f32) -> i32 {
    ffi_catch(-1, || {
        if model.is_null() || logits.is_null() {
            return -1;
        }
        with_model(model, |inner| {
            let vocab = inner.runner.ctx.model.config.vocab_size;
            let logits = unsafe { std::slice::from_raw_parts(logits, vocab) };
            let token = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            token as i32
        })
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
