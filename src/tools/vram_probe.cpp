#include "vram_probe.h"

#include <cuda_runtime.h>
#include <cstdio>

int vram_probe_bytes_per_elem(VramDtype dtype) {
    switch (dtype) {
    case VRAM_DTYPE_F32:
        return 4;
    case VRAM_DTYPE_F16:
    case VRAM_DTYPE_BF16:
        return 2;
    case VRAM_DTYPE_I8:
        return 1;
    case VRAM_DTYPE_I4:
        return 0; // 半字节，estimate_weight_bytes 单独处理
    default:
        return -1;
    }
}

int vram_probe_query(size_t *free_bytes, size_t *total_bytes) {
    if (free_bytes == nullptr || total_bytes == nullptr) {
        return -1;
    }
    const cudaError_t err = cudaMemGetInfo(free_bytes, total_bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[vram] cudaMemGetInfo failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

int vram_probe_print(const char *tag) {
    size_t free_b = 0;
    size_t total_b = 0;
    if (vram_probe_query(&free_b, &total_b) != 0) {
        return -1;
    }
    const size_t used_b = total_b - free_b;
    const char *label = (tag != nullptr && tag[0] != '\0') ? tag : "-";
    // MiB = 1024*1024；多行对齐，例：
    // -------- [vram] test_query --------
    //   free   1532.92 MiB
    //   used≈  2069.00 MiB
    //   total  3601.92 MiB
    // ------------------------------------
    const double free_mib = free_b / (1024.0 * 1024.0);
    const double used_mib = used_b / (1024.0 * 1024.0);
    const double total_mib = total_b / (1024.0 * 1024.0);
    std::printf("-------- [vram] %s --------\n", label);
    std::printf("  free  %8.2f MiB\n", free_mib);
    std::printf("  used≈ %8.2f MiB\n", used_mib);
    std::printf("  total %8.2f MiB\n", total_mib);
    std::printf("--------------------------------\n");
    std::fflush(stdout);
    return 0;
}

int64_t vram_probe_estimate_weight_bytes(int64_t num_params, VramDtype dtype) {
    if (num_params < 0) {
        return -1;
    }
    if (dtype == VRAM_DTYPE_I4) {
        // 两枚 INT4 挤进 1 字节；奇数参数向上取整。未计入 scale/zero-point。
        // 例：num_params=8 -> 4 字节；num_params=7 -> 4 字节
        return (num_params + 1) / 2;
    }
    const int bpe = vram_probe_bytes_per_elem(dtype);
    if (bpe <= 0) {
        return -1;
    }
    return num_params * static_cast<int64_t>(bpe);
}

int64_t vram_probe_estimate_kv_bytes(int num_layers, int num_kv_heads, int head_dim, int max_seq,
                                     VramDtype dtype) {
    if (num_layers <= 0 || num_kv_heads <= 0 || head_dim <= 0 || max_seq <= 0) {
        return -1;
    }
    const int bpe = vram_probe_bytes_per_elem(dtype);
    if (bpe <= 0) {
        // KV 暂不估 INT4 packed；需要时再扩
        return -1;
    }
    // layers * (K+V) * kv_heads * head_dim * max_seq * bytes
    const int64_t elems = static_cast<int64_t>(num_layers) * 2 *
                          static_cast<int64_t>(num_kv_heads) * static_cast<int64_t>(head_dim) *
                          static_cast<int64_t>(max_seq);
    return elems * static_cast<int64_t>(bpe);
}

int64_t vram_probe_estimate_session_pool_bytes(int vocab_size, int hidden_size, int max_seq) {
    if (vocab_size <= 0 || hidden_size <= 0 || max_seq <= 0) {
        return -1;
    }
    // 与 ie_create 池对齐（当前全 F32 / int）：
    //   d_token_ids: int[max_seq]
    //   d_logits:    float[vocab, max_seq]
    //   d_hidden_out:float[hidden, max_seq]
    //   d_out_token: int[1]
    const int64_t token_ids = static_cast<int64_t>(max_seq) * static_cast<int64_t>(sizeof(int));
    const int64_t logits = static_cast<int64_t>(vocab_size) * static_cast<int64_t>(max_seq) *
                           static_cast<int64_t>(sizeof(float));
    const int64_t hidden = static_cast<int64_t>(hidden_size) * static_cast<int64_t>(max_seq) *
                           static_cast<int64_t>(sizeof(float));
    const int64_t out_token = static_cast<int64_t>(sizeof(int));
    return token_ids + logits + hidden + out_token;
}

int64_t vram_probe_estimate_llama_num_params(int hidden_size, int intermediate_size, int num_layers,
                                             int num_q_heads, int num_kv_heads, int head_dim,
                                             int vocab_size, int tie_word_embeddings) {
    if (hidden_size <= 0 || intermediate_size <= 0 || num_layers <= 0 || num_q_heads <= 0 ||
        num_kv_heads <= 0 || head_dim <= 0 || vocab_size <= 0) {
        return -1;
    }
    const int64_t H = hidden_size;
    const int64_t I = intermediate_size;
    const int64_t q_dim = static_cast<int64_t>(num_q_heads) * head_dim;
    const int64_t kv_dim = static_cast<int64_t>(num_kv_heads) * head_dim;
    const int64_t V = vocab_size;

    // 单层：Q/K/V/O + gate/up/down + 2 * RMSNorm(H)
    // 例：H=2048,I=5632,q=2048,kv=256
    //   q:2048*2048  k:256*2048  v:256*2048  o:2048*2048
    //   gate/up:5632*2048  down:2048*5632  norms:2*2048
    const int64_t per_layer =
        q_dim * H + kv_dim * H + kv_dim * H + H * q_dim + I * H + I * H + H * I + H + H;

    int64_t total = V * H; // embed
    total += H;            // final_norm
    total += static_cast<int64_t>(num_layers) * per_layer;
    if (!tie_word_embeddings) {
        total += H * V; // untied lm_head
    }
    return total;
}
