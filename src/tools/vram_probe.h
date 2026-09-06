#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 权重 / KV 估算用的元素宽度。BF16 与 FP16 同为 2 字节。
typedef enum VramDtype {
    VRAM_DTYPE_F32 = 0,
    VRAM_DTYPE_F16 = 1,
    VRAM_DTYPE_BF16 = 2,
    VRAM_DTYPE_I8 = 3,
    VRAM_DTYPE_I4 = 4,
} VramDtype;

// 返回每元素字节数；非法 dtype 返回 -1。
// 例：VRAM_DTYPE_F32 -> 4；VRAM_DTYPE_F16/BF16 -> 2；I8 -> 1；I4 -> 0（半字节，见 estimate
// 内特殊处理）
int vram_probe_bytes_per_elem(VramDtype dtype);

// 运行时：cudaMemGetInfo。成功返回 0，失败返回 -1。
int vram_probe_query(size_t *free_bytes, size_t *total_bytes);

// 多行打印（对齐可读）：
// -------- [vram] tag --------
//   free  ... MiB
//   used≈ ... MiB
//   total ... MiB
// --------------------------------
// tag 可为 NULL（打印为 "-"）。成功 0，失败 -1。
int vram_probe_print(const char *tag);

// 权重字节：num_params * bytes_per_elem。
// I4：按 ceil(num_params/2) 字节（未含 scale）。
// 例：num_params=1.1e9, F16 -> 2200000000
int64_t vram_probe_estimate_weight_bytes(int64_t num_params, VramDtype dtype);

// KV：layers * 2(K+V) * num_kv_heads * head_dim * max_seq * elem_bytes
// 例：22 * 2 * 4 * 64 * 256 * 4 = 11534336（约 11.0 MiB）
int64_t vram_probe_estimate_kv_bytes(int num_layers, int num_kv_heads, int head_dim, int max_seq,
                                     VramDtype dtype);

// 当前 Engine session 池（F32）：token_ids + logits[vocab,max_seq] + hidden[hidden,max_seq] +
// out_token 例：vocab=32000 hidden=2048 max_seq=256 -> logits≈32.8MiB + hidden≈2.0MiB + 少量 int
int64_t vram_probe_estimate_session_pool_bytes(int vocab_size, int hidden_size, int max_seq);

// 按 Llama/TinyLlama dense 布局估算参数量（与 ModelConfig 字段一致）。
// 含 embed、final_norm、各层 QKVO+SwiGLU+2*RMSNorm；untied 时再加 lm_head。
// 例：TinyLlama 22 层典型约 1.1e9 量级。
int64_t vram_probe_estimate_llama_num_params(int hidden_size, int intermediate_size, int num_layers,
                                             int num_q_heads, int num_kv_heads, int head_dim,
                                             int vocab_size, int tie_word_embeddings);

#ifdef __cplusplus
}
#endif
