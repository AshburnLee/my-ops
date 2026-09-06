"""VRAM 探针：静态估算 + cudaMemGetInfo；可选在 2 层切片 load/create 前后打印。"""

from __future__ import annotations

import os

import vram_probe_me

# TinyLlama-1.1B-Chat 与本机切片 config 对齐的字段
_TINYLLAMA = dict(
    hidden_size=2048,
    intermediate_size=5632,
    num_q_heads=32,
    num_kv_heads=4,
    head_dim=64,
    vocab_size=32000,
    max_seq=256,
    tie_word_embeddings=0,
)


def test_vram_probe_static_tinyllama_budget() -> None:
    # 22 层 F32 权重应大于 Orin ~3.78GB；F16 应落在 ~2GB 量级
    rep22_f32 = vram_probe_me.estimate_report(
        **{**_TINYLLAMA, "num_layers": 22}, weight_dtype="f32", kv_dtype="f32"
    )
    rep22_f16 = vram_probe_me.estimate_report(
        **{**_TINYLLAMA, "num_layers": 22}, weight_dtype="f16", kv_dtype="f32"
    )
    rep2_f32 = vram_probe_me.estimate_report(
        **{**_TINYLLAMA, "num_layers": 2}, weight_dtype="f32", kv_dtype="f32"
    )

    assert rep22_f32["num_params"] > 1_000_000_000
    assert rep22_f32["num_params"] < 1_300_000_000
    # F16 权重约为 F32 一半
    assert abs(rep22_f16["weight_bytes"] * 2 - rep22_f32["weight_bytes"]) < 8

    # KV@256 F32 22 层：22*2*4*64*256*4 = 11534336
    assert rep22_f32["kv_bytes"] == 22 * 2 * 4 * 64 * 256 * 4

    total = int(rep22_f16.get("cuda_total_bytes", 0))
    if total > 0:
        # Orin 上：F32 整模权重不应 fits；F16 权重应 fits_weights_only
        assert rep22_f32["fits_weights_only"] is False
        assert rep22_f16["fits_weights_only"] is True
        assert rep2_f32["weight_bytes"] < total

    print(
        "Passed test_vram_probe_static_tinyllama_budget",
        f"nparams22={rep22_f16['num_params']}",
        f"w_f32_mib={rep22_f32['weight_mib']:.1f}",
        f"w_f16_mib={rep22_f16['weight_mib']:.1f}",
        f"kv_mib={rep22_f16['kv_mib']:.2f}",
        f"pool_mib={rep22_f16['session_pool_mib']:.2f}",
        f"cuda_total_mib={rep22_f16.get('cuda_total_mib', 'n/a')}",
    )


def test_vram_probe_print_and_query() -> None:
    free_b, total_b = vram_probe_me.query()
    assert total_b > 0
    assert free_b <= total_b
    vram_probe_me.print_mem("test_query")
    print(
        "Passed test_vram_probe_print_and_query",
        f"free_mib={free_b / (1024**2):.2f}",
        f"total_mib={total_b / (1024**2):.2f}",
    )


def test_vram_probe_on_two_layer_slice_optional() -> None:
    """可选：2 层 F32 load/create 前后打探针。无切片则 skip。"""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    slice_dir = os.environ.get("JJ_HF_LLAMA_SLICE_DIR", "").strip() or os.path.join(
        root, "models", "tinyllama_2layer"
    )
    st_path = os.path.join(slice_dir, "model.safetensors")
    cfg_path = os.path.join(slice_dir, "config.txt")
    if not (os.path.isfile(st_path) and os.path.isfile(cfg_path)):
        print("Passed test_vram_probe_on_two_layer_slice_optional skipped (no slice)")
        return

    import inference_engine_me
    import transformer_model_me

    from test_hf_llama_real_slice_smoke import _parse_config_txt

    cfg = _parse_config_txt(cfg_path)
    vram_probe_me.print_mem("before_create_model")
    model = transformer_model_me.create_model(**cfg)
    engine = None
    try:
        vram_probe_me.print_mem("after_create_model_empty")
        transformer_model_me.load_weights_from_safetensors_hf_llama(model, st_path)
        vram_probe_me.print_mem("after_load_weights_2layer_f32")
        engine = inference_engine_me.create_engine(model)
        vram_probe_me.print_mem("after_ie_create")
        print("Passed test_vram_probe_on_two_layer_slice_optional")
    finally:
        if engine is not None:
            inference_engine_me.destroy_engine(engine)
        transformer_model_me.destroy_model(model)
        vram_probe_me.print_mem("after_destroy")


if __name__ == "__main__":
    test_vram_probe_static_tinyllama_budget()
    test_vram_probe_print_and_query()
    test_vram_probe_on_two_layer_slice_optional()
