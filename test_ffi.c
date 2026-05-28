// Minimal C test for Aether FFI.
// Build: gcc -o test_ffi test_ffi.c -L target/release -laether -Wl,-rpath,target/release
#include <stdio.h>
#include <stdlib.h>
#include "aether.h"

int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1] : "tinyllama-q4.gguf";

    printf("[test] Loading model: %s\n", model_path);
    aether_model* m = aether_load(model_path);
    if (!m) {
        fprintf(stderr, "[test] FAIL: aether_load returned NULL\n");
        return 1;
    }

    int vocab = aether_vocab_size(m);
    int ctx   = aether_context_len(m);
    int nlay  = aether_num_layers(m);
    int dmod  = aether_d_model(m);
    printf("[test] Model: vocab=%d ctx=%d layers=%d d_model=%d\n", vocab, ctx, nlay, dmod);

    if (vocab <= 0) {
        fprintf(stderr, "[test] FAIL: vocab_size=%d\n", vocab);
        aether_free(m);
        return 1;
    }

    // Encode prompt
    int n_tokens;
    int* tokens = aether_encode(m, "The future of AI is", &n_tokens);
    if (!tokens) {
        fprintf(stderr, "[test] FAIL: aether_encode returned NULL\n");
        aether_free(m);
        return 1;
    }
    printf("[test] Prompt encoded: %d tokens\n", n_tokens);
    for (int i = 0; i < n_tokens && i < 5; i++) {
        printf("  token[%d] = %d\n", i, tokens[i]);
    }

    // Prefill
    float* logits = (float*)malloc(vocab * sizeof(float));
    int ret = aether_prefill(m, tokens, n_tokens, logits);
    if (ret != AETHER_OK) {
        fprintf(stderr, "[test] FAIL: aether_prefill returned %d\n", ret);
        free(logits);
        aether_free_tokens(tokens);
        aether_free(m);
        return 1;
    }
    printf("[test] Prefill OK, logits[0]=%.4f max=%.4f\n",
           logits[0], logits[aether_argmax(m, logits)]);

    // Sample
    int sampled = aether_sample(m, logits, 0.7f, 0.9f);
    char* token_str = aether_decode_token(m, sampled);
    printf("[test] First token: %d -> '%s'\n", sampled, token_str);
    aether_free_string(token_str);

    // Decode a few steps
    aether_set_frame_budget(m, 50.0f);
    for (int i = 0; i < 3; i++) {
        int pos = n_tokens + i;

        // Non-budgeted decode
        ret = aether_decode(m, sampled, pos, logits);
        printf("[test] Decode step %d: ret=%d logits[0]=%.4f\n", i, ret, logits[0]);

        sampled = aether_argmax(m, logits);
        token_str = aether_decode_token(m, sampled);
        printf("  -> token %d: '%s'\n", sampled, token_str);
        aether_free_string(token_str);
    }

    free(logits);
    aether_free_tokens(tokens);
    aether_free(m);
    printf("[test] PASS: All FFI tests passed\n");
    return 0;
}
