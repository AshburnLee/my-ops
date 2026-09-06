#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// CPU 快照：各 online 核 scaling_cur_freq 的 min/max/avg，以及 cpuinfo_max_freq（理论/硬件上限）。
// 频率单位一律 kHz（与 Linux cpufreq sysfs 一致）。
//
// 例（Orin 某时刻 6 核）：
//   cur 可能是 729600 / 1267200 / 1728000 混杂
//   max_khz == 1728000  （cpuinfo_max_freq）
typedef struct FreqProbeCpuSnapshot {
    int num_cores; // 成功读到 scaling_cur_freq 的核数
    int cur_khz_min;
    int cur_khz_max;
    int cur_khz_avg; // 整数平均
    int max_khz;     // 各核 cpuinfo_max_freq 的最大值；读失败为 0
} FreqProbeCpuSnapshot;

// GPU 快照：devfreq 的 cur_freq / max_freq（sysfs 为 Hz，这里换成 kHz）。
// 例（Orin 17000000.gpu）：
//   cur_khz == 306000   （306 MHz，空闲常见）
//   max_khz == 1020000  （1020 MHz，与 cudaDeviceProp.clockRate 量级一致，但是实时读 sysfs）
typedef struct FreqProbeGpuSnapshot {
    int cur_khz;
    int max_khz;
    char path[256]; // 实际读取的 devfreq 目录，便于核对
} FreqProbeGpuSnapshot;

// 读 CPU sysfs。成功 0；无 cpufreq 或全失败 -1。
int freq_probe_cpu_query(FreqProbeCpuSnapshot *out);

// 读 GPU devfreq（优先 name 含 "gpu" 的节点）。成功 0；找不到 -1。
int freq_probe_gpu_query(FreqProbeGpuSnapshot *out);

// 多行打印 CPU + GPU（对齐可读）：
// -------- [freq] tag --------
//   CPU  cores=N
//        cur   min=...  avg=...  max=...  MHz
//        hw_max=... MHz
//   GPU  cur=... / max=... MHz
// --------------------------------
// tag 可为 NULL。任一侧失败仍打印另一侧；两侧都失败返回 -1。
int freq_probe_print(const char *tag);

#ifdef __cplusplus
}
#endif
