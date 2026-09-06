#include "vram_probe.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <unordered_map>

namespace py = pybind11;

static VramDtype parse_dtype(const std::string &name) {
    static const std::unordered_map<std::string, VramDtype> kMap = {
        {"f32", VRAM_DTYPE_F32},   {"fp32", VRAM_DTYPE_F32},      {"float32", VRAM_DTYPE_F32},
        {"f16", VRAM_DTYPE_F16},   {"fp16", VRAM_DTYPE_F16},      {"float16", VRAM_DTYPE_F16},
        {"bf16", VRAM_DTYPE_BF16}, {"bfloat16", VRAM_DTYPE_BF16}, {"i8", VRAM_DTYPE_I8},
        {"int8", VRAM_DTYPE_I8},   {"i4", VRAM_DTYPE_I4},         {"int4", VRAM_DTYPE_I4},
    };
    const auto it = kMap.find(name);
    if (it == kMap.end()) {
        throw std::invalid_argument("unknown dtype: " + name);
    }
    return it->second;
}

static double to_mib(int64_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

PYBIND11_MODULE(vram_probe_me, m) {
    m.doc() = "VRAM probe: cudaMemGetInfo + static TinyLlama/Orin budget estimates";

    m.def(
        "query",
        []() {
            size_t free_b = 0;
            size_t total_b = 0;
            if (vram_probe_query(&free_b, &total_b) != 0) {
                throw std::runtime_error("vram_probe_query failed");
            }
            return py::make_tuple(free_b, total_b);
        },
        "Return (free_bytes, total_bytes) from cudaMemGetInfo");

    m.def(
        "print_mem",
        [](const std::string &tag) {
            if (vram_probe_print(tag.c_str()) != 0) {
                throw std::runtime_error("vram_probe_print failed");
            }
        },
        py::arg("tag") = "", "Print multi-line free / used≈ / total (MiB) via cudaMemGetInfo");

    m.def(
        "bytes_per_elem",
        [](const std::string &dtype) {
            const VramDtype dt = parse_dtype(dtype);
            const int b = vram_probe_bytes_per_elem(dt);
            if (b < 0 && dt != VRAM_DTYPE_I4) {
                throw std::invalid_argument("bad dtype");
            }
            return b;
        },
        py::arg("dtype"),
        "Element size in bytes; int4 returns 0 (packed, see estimate_weight_bytes)");

    m.def(
        "estimate_weight_bytes",
        [](int64_t num_params, const std::string &dtype) {
            const int64_t n = vram_probe_estimate_weight_bytes(num_params, parse_dtype(dtype));
            if (n < 0) {
                throw std::invalid_argument("estimate_weight_bytes failed");
            }
            return n;
        },
        py::arg("num_params"), py::arg("dtype") = "f16");

    m.def(
        "estimate_kv_bytes",
        [](int num_layers, int num_kv_heads, int head_dim, int max_seq, const std::string &dtype) {
            const int64_t n = vram_probe_estimate_kv_bytes(num_layers, num_kv_heads, head_dim,
                                                           max_seq, parse_dtype(dtype));
            if (n < 0) {
                throw std::invalid_argument("estimate_kv_bytes failed");
            }
            return n;
        },
        py::arg("num_layers"), py::arg("num_kv_heads"), py::arg("head_dim"), py::arg("max_seq"),
        py::arg("dtype") = "f32");

    m.def(
        "estimate_session_pool_bytes",
        [](int vocab_size, int hidden_size, int max_seq) {
            const int64_t n =
                vram_probe_estimate_session_pool_bytes(vocab_size, hidden_size, max_seq);
            if (n < 0) {
                throw std::invalid_argument("estimate_session_pool_bytes failed");
            }
            return n;
        },
        py::arg("vocab_size"), py::arg("hidden_size"), py::arg("max_seq"));

    m.def(
        "estimate_llama_num_params",
        [](int hidden_size, int intermediate_size, int num_layers, int num_q_heads,
           int num_kv_heads, int head_dim, int vocab_size, int tie_word_embeddings) {
            const int64_t n = vram_probe_estimate_llama_num_params(
                hidden_size, intermediate_size, num_layers, num_q_heads, num_kv_heads, head_dim,
                vocab_size, tie_word_embeddings);
            if (n < 0) {
                throw std::invalid_argument("estimate_llama_num_params failed");
            }
            return n;
        },
        py::arg("hidden_size"), py::arg("intermediate_size"), py::arg("num_layers"),
        py::arg("num_q_heads"), py::arg("num_kv_heads"), py::arg("head_dim"), py::arg("vocab_size"),
        py::arg("tie_word_embeddings") = 0);

    m.def(
        "estimate_report",
        [](int hidden_size, int intermediate_size, int num_layers, int num_q_heads,
           int num_kv_heads, int head_dim, int vocab_size, int max_seq, int tie_word_embeddings,
           const std::string &weight_dtype, const std::string &kv_dtype) {
            const int64_t nparams = vram_probe_estimate_llama_num_params(
                hidden_size, intermediate_size, num_layers, num_q_heads, num_kv_heads, head_dim,
                vocab_size, tie_word_embeddings);
            if (nparams < 0) {
                throw std::invalid_argument("bad model shape");
            }
            const VramDtype wdt = parse_dtype(weight_dtype);
            const VramDtype kdt = parse_dtype(kv_dtype);
            const int64_t w_bytes = vram_probe_estimate_weight_bytes(nparams, wdt);
            const int64_t kv_bytes =
                vram_probe_estimate_kv_bytes(num_layers, num_kv_heads, head_dim, max_seq, kdt);
            const int64_t pool_bytes =
                vram_probe_estimate_session_pool_bytes(vocab_size, hidden_size, max_seq);
            if (w_bytes < 0 || kv_bytes < 0 || pool_bytes < 0) {
                throw std::invalid_argument("estimate failed");
            }
            const int64_t subtotal = w_bytes + kv_bytes + pool_bytes;

            size_t free_b = 0;
            size_t total_b = 0;
            const bool have_runtime = (vram_probe_query(&free_b, &total_b) == 0);

            py::dict out;
            out["num_params"] = nparams;
            out["weight_bytes"] = w_bytes;
            out["kv_bytes"] = kv_bytes;
            out["session_pool_bytes"] = pool_bytes;
            out["subtotal_bytes"] = subtotal;
            out["weight_mib"] = to_mib(w_bytes);
            out["kv_mib"] = to_mib(kv_bytes);
            out["session_pool_mib"] = to_mib(pool_bytes);
            out["subtotal_mib"] = to_mib(subtotal);
            out["weight_dtype"] = weight_dtype;
            out["kv_dtype"] = kv_dtype;
            if (have_runtime) {
                out["cuda_free_bytes"] = free_b;
                out["cuda_total_bytes"] = total_b;
                out["cuda_used_bytes"] = total_b - free_b;
                out["cuda_free_mib"] = to_mib(static_cast<int64_t>(free_b));
                out["cuda_total_mib"] = to_mib(static_cast<int64_t>(total_b));
                out["fits_weights_only"] = static_cast<int64_t>(total_b) > w_bytes;
                out["fits_subtotal"] = static_cast<int64_t>(total_b) > subtotal;
            }
            return out;
        },
        py::arg("hidden_size"), py::arg("intermediate_size"), py::arg("num_layers"),
        py::arg("num_q_heads"), py::arg("num_kv_heads"), py::arg("head_dim"), py::arg("vocab_size"),
        py::arg("max_seq"), py::arg("tie_word_embeddings") = 0, py::arg("weight_dtype") = "f16",
        py::arg("kv_dtype") = "f32",
        "Static budget dict (+ runtime cuda free/total when available)");
}
