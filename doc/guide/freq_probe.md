# CPU / GPU 频率探针（与 vram_probe 配套）

Python：`freq_probe_me`。C：`src/tools/freq_probe.{h,cpp}`（诊断工具；与实现同目录）。

和 `vram_probe_me` 一样是 **诊断工具**：在 load / create / 首步 prefill 边界手动打点，**不要**塞进每步 decode。

---

## 原理

不读 CUDA「当前 kernel 频率」API（`cudaDeviceProp.clockRate` 基本是标称峰值，不是实时）。

本工具读 Linux sysfs：

~~~
CPU  /sys/devices/system/cpu/cpuN/cpufreq/scaling_cur_freq   # 实时，kHz
     /sys/devices/system/cpu/cpuN/cpufreq/cpuinfo_max_freq   # 硬件/策略上限，kHz

GPU  /sys/class/devfreq/17000000.gpu/cur_freq                # 实时，Hz（内部转 kHz）
     /sys/class/devfreq/17000000.gpu/max_freq                # 上限，Hz
~~~

Orin 上 GPU 节点名通常含 `gpu`；找不到固定路径时会扫 `/sys/class/devfreq/*/name`。

与温控/降频的关系：若 `cur` 长期远低于 `max`，可能是 governor、nvpmodel 上限或 thermal throttle（见仓库根 `CPU-throttling.md`）。探针只报数，不解释原因。

---

## API

~~~python
import freq_probe_me
import vram_probe_me

freq_probe_me.print_freq("before_load")
vram_probe_me.print_mem("before_load")

cpu = freq_probe_me.query_cpu()
# cpu: num_cores, cur_khz_{min,max,avg}, max_khz, cur_mhz_avg, max_mhz

gpu = freq_probe_me.query_gpu()
# gpu: cur_khz, max_khz, cur_mhz, max_mhz, path

both = freq_probe_me.query()  # {cpu: {...}, gpu: {...}}
~~~

打印示例：

~~~
-------- [freq] before_load --------
  CPU  cores=6
       cur   min= 729.6  avg=1400.0  max=1728.0  MHz
       hw_max=1728.0 MHz
  GPU  cur= 306.0 / max=1020.0 MHz
--------------------------------
~~~

---

## 测试

~~~bash
./run_tests.sh --suite test_freq_probe.py
~~~

`test_freq_and_vram_probe_together` 演示与显存探针一起用。
