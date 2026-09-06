# Orin 显存账（TinyLlama 整模落地）

目标：把 **TinyLlama-1.1B-Chat 22 层** 塞进本机 Orin，而不是继续只跑 2 层 F32 切片。
量化管的是 **放得下**；2 层 F32 vs HF logits 回归必须继续绿（另一条路径）。

硬件实测（`./build/show_device_info`）：

~~~
device                 Orin
total_global_mem       ~3.78 GB
~~~

---

## 1. 权重显存（主导项）

TinyLlama ~1.1B 参数。按「整模参数量 × 每元素字节」粗算：

~~~
dtype        每参数    权重约      相对 F32    本机对照
──────────────────────────────────────────────────────────
F32          4 B       ~4.40 GB    100%        放不进 3.78GB
FP16 / BF16  2 B       ~2.20 GB     50%        与 HF 源仓
                                              model.safetensors
                                              (torch_dtype=bfloat16,
                                               文件 ~2.20 GB) 一致
INT8         1 B       ~1.10 GB     25%
INT4         0.5 B     ~0.55 GB     12%        + scale 开销另计
~~~

**结论（权重 alone）**：

- 全量 **F32**：不行
- **FP16（或磁盘 BF16 转成 GPU FP16）**：权重约 2.2GB，**理论上能进** 3.78GB，但仍要留 KV / BufferPool / 驱动与系统余量
- 若 FP16 整模 create / 短 generate 仍 OOM 或 free 过紧 -> 再 **INT8 / INT4**

Orin 生产路径优先 **FP16**（2 字节，算子生态成熟）。HF 仓库常是 **BF16 文件**；加载时可 BF16->FP16，不必先上 BF16 Tensor Core 路径。

---

## 2. 其它显存（相对权重小，但不能当 0）

以 TinyLlama config：`hidden=2048`，`vocab=32000`，`num_layers=22`，`num_kv_heads=4`，`head_dim=64`，`max_seq_len=256`（与现切片开发配置一致）粗算：

~~~
项                         估算（约）
────────────────────────────────────────
KV cache F32               22 * 2 * 4 * 64 * 256 * 4 B  ≈ 11.5 MB
session d_logits F32       32000 * 256 * 4 B            ≈ 32.8 MB
session d_hidden F32       2048 * 256 * 4 B             ≈  2.1 MB
layer workspace（按 T）    视 T；decode T=1 远小于权重
CUDA 上下文 / 碎片 / OS    数百 MB 量级（实测为准）
~~~

**经验式总账（落地验收时用探针打印，不靠拍脑袋）**：

~~~
peak ≈ weights(dtype) + KV + BufferPool + layer_workspace + cuda_overhead
要求：peak < total_global_mem，且 create 后 free 仍留余量（建议至少几百 MB）
~~~

---

## 3. 静态估算 + 运行时探针（已实现）

C：`src/tools/vram_probe.h` + `src/tools/vram_probe.cpp`（诊断工具；与实现同目录）
Python：`vram_probe_me`（`tests/test_vram_probe.py`）

~~~
vram_probe_me.query()                    -> (free_bytes, total_bytes)
vram_probe_me.print_mem(tag)             -> 多行打印 free / used≈ / total（MiB）
vram_probe_me.estimate_llama_num_params(...)
vram_probe_me.estimate_weight_bytes(n, dtype)
vram_probe_me.estimate_kv_bytes(...)
vram_probe_me.estimate_session_pool_bytes(...)
vram_probe_me.estimate_report(..., weight_dtype="f16")  -> dict
~~~

本机一次实测（`test_vram_probe.py`，Orin）：

~~~
cuda total                 ~3602 MiB   (~3.52 GiB 进程可见；含驱动占用后小于标称 3.78GB)
TinyLlama 22L 估算参数     1,100,048,384
权重 F32                   ~4196 MiB   fits_weights_only=False
权重 F16                   ~2098 MiB   fits_weights_only=True
KV F32 @256                ~11 MiB
session pool F32           ~33 MiB
~~~

建议在 `load_weights` 前后、`ie_create` 后、第一次 prefill 后调用 `print_mem`，判断 FP16 够不够。
频率探针（CPU/GPU cur vs max）见 [`freq_probe.md`](freq_probe.md)，可与 `print_mem` 同一边界一起打点。

---

## 4. 与 2 层 F32 ref 的关系

~~~
路径 A  2 层 F32 切片  <->  ref_prefill_logits_*.npy     必须继续绿
路径 B  22 层 FP16/更深量化  <->  另定 tol / 能否 load+generate
~~~

FP16 整模 **不会** 与 F32 ref 逐点对齐；验收先是 **不 OOM + 短 generate 有限输出**，数值 tol 另开。

---

## 5. 推荐推进顺序

1. 写本账 + roadmap 步骤（本文）
2. 落地 `print_cuda_mem` / 静态估算，先在 **现有 2 层 F32** 上把探针跑通
3. Loader/Model/Linear 支持 **权重 FP16**（激活/KV 可仍 F32）
4. 加载 **22 层 FP16**（可由 HF BF16 转），打印显存，短 generate
5. 若仍放不下 -> INT8/INT4，再测

清单勾选见 **jjjetson-ops-roadmap.mdc** 模块 6。
