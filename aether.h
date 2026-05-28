#ifndef AETHER_H
#define AETHER_H

#ifdef _WIN32
#define AETHER_API __declspec(dllexport)
#else
#define AETHER_API __attribute__((visibility("default")))
#endif

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Opaque handle ───────────────────────────────────────────────────── */

typedef struct aether_model aether_model;

/* ── Error codes ──────────────────────────────────────────────────────── */

#define AETHER_OK       0
#define AETHER_ERR     -1
#define AETHER_TIMEOUT -2

/* ── Model lifecycle ──────────────────────────────────────────────────── */

AETHER_API aether_model* aether_load(const char* path);
AETHER_API aether_model* aether_load_streaming(const char* path, int max_hot);
AETHER_API void aether_free(aether_model* model);

/* ── Config queries ───────────────────────────────────────────────────── */

AETHER_API int aether_vocab_size(const aether_model* model);
AETHER_API int aether_context_len(const aether_model* model);
AETHER_API int aether_num_layers(const aether_model* model);
AETHER_API int aether_d_model(const aether_model* model);
AETHER_API int aether_eos_id(const aether_model* model);
AETHER_API int aether_bos_id(const aether_model* model);
AETHER_API int aether_num_gpu_layers(const aether_model* model);
AETHER_API int aether_num_cpu_layers(const aether_model* model);

/* ── Frame budget ────────────────────────────────────────────────────────
 *
 * Set a per-decode-step time budget in milliseconds.
 * Once set, aether_decode_budgeted returns AETHER_TIMEOUT if the decode
 * step exceeds this budget. Useful for real-time rendering loops (UE5,
 * Unity) where a frame must not exceed X ms.
 */

AETHER_API void aether_set_frame_budget(aether_model* model, float max_ms);
AETHER_API float aether_frame_budget(const aether_model* model);

/* ── Tokenization ─────────────────────────────────────────────────────── */

AETHER_API int* aether_encode(const aether_model* model, const char* text, int* out_len);
AETHER_API char* aether_decode_token(const aether_model* model, int token);
AETHER_API void aether_free_string(char* s);
AETHER_API void aether_free_tokens(int* tokens);

/* ── Inference ───────────────────────────────────────────────────────────
 *
 * All inference functions write logits to the provided buffer, which must
 * be at least aether_vocab_size() * sizeof(float) bytes.
 *
 * Position tracking: after aether_prefill with N tokens, the first decode
 * step has pos = N. Increment pos by 1 for each subsequent decode step.
 */

AETHER_API int aether_prefill(aether_model* model, const int* tokens, int n_tokens, float* logits_out);
AETHER_API int aether_decode(aether_model* model, int token, int pos, float* logits_out);
AETHER_API int aether_decode_budgeted(aether_model* model, int token, int pos, float* logits_out);

/* ── Sampling ─────────────────────────────────────────────────────────── */

AETHER_API int aether_sample(const aether_model* model, const float* logits, float temperature, float top_p);
AETHER_API int aether_argmax(const aether_model* model, const float* logits);

/* ── Error ────────────────────────────────────────────────────────────── */

AETHER_API char* aether_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* AETHER_H */
