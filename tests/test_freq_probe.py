"""CPU/GPU 频率探针：sysfs 实时 cur vs max；可与 vram_probe_me 一起打点。"""

from __future__ import annotations

import freq_probe_me
import vram_probe_me


def test_freq_probe_query_cpu_gpu() -> None:
    cpu = freq_probe_me.query_cpu()
    gpu = freq_probe_me.query_gpu()

    assert cpu["num_cores"] >= 1
    assert cpu["cur_khz_min"] > 0
    assert cpu["cur_khz_max"] >= cpu["cur_khz_min"]
    assert cpu["cur_khz_avg"] >= cpu["cur_khz_min"]
    assert cpu["max_khz"] >= cpu["cur_khz_max"]

    assert gpu["cur_khz"] > 0
    assert gpu["max_khz"] >= gpu["cur_khz"]
    assert "gpu" in gpu["path"]

    both = freq_probe_me.query()
    assert both["cpu"] is not None
    assert both["gpu"] is not None

    freq_probe_me.print_freq("test_query")
    print(
        "Passed test_freq_probe_query_cpu_gpu",
        f"cpu_avg={cpu['cur_mhz_avg']:.1f}/{cpu['max_mhz']:.1f}MHz",
        f"gpu={gpu['cur_mhz']:.1f}/{gpu['max_mhz']:.1f}MHz",
        f"path={gpu['path']}",
    )


def test_freq_and_vram_probe_together() -> None:
    """推理边界推荐打点：频率 + 显存一起打（不进 decode 热路径）。"""
    freq_probe_me.print_freq("together_before")
    vram_probe_me.print_mem("together_before")
    _ = freq_probe_me.query()
    free_b, total_b = vram_probe_me.query()
    assert total_b > 0
    assert free_b <= total_b
    freq_probe_me.print_freq("together_after")
    vram_probe_me.print_mem("together_after")
    print("Passed test_freq_and_vram_probe_together")


if __name__ == "__main__":
    test_freq_probe_query_cpu_gpu()
    test_freq_and_vram_probe_together()
