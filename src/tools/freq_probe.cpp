#include "freq_probe.h"

#include <cstdio>
#include <cstring>
#include <dirent.h>

namespace {

// 读单个 sysfs 文件里的 long long；失败返回 false。
bool read_ll_file(const char *path, long long *out) {
    if (path == nullptr || out == nullptr) {
        return false;
    }
    FILE *fp = std::fopen(path, "r");
    if (fp == nullptr) {
        return false;
    }
    long long v = 0;
    const int n = std::fscanf(fp, "%lld", &v);
    std::fclose(fp);
    if (n != 1) {
        return false;
    }
    *out = v;
    return true;
}

bool read_trimmed_line(const char *path, char *buf, size_t buflen) {
    if (path == nullptr || buf == nullptr || buflen == 0) {
        return false;
    }
    FILE *fp = std::fopen(path, "r");
    if (fp == nullptr) {
        return false;
    }
    if (std::fgets(buf, static_cast<int>(buflen), fp) == nullptr) {
        std::fclose(fp);
        return false;
    }
    std::fclose(fp);
    // 去掉尾部空白
    size_t n = std::strlen(buf);
    while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r' || buf[n - 1] == ' ')) {
        buf[--n] = '\0';
    }
    return n > 0;
}

double khz_to_mhz(int khz) {
    return static_cast<double>(khz) / 1000.0;
}

} // namespace

int freq_probe_cpu_query(FreqProbeCpuSnapshot *out) {
    if (out == nullptr) {
        return -1;
    }
    std::memset(out, 0, sizeof(*out));

    int n = 0;
    long long sum = 0;
    int cur_min = 0;
    int cur_max = 0;
    int hw_max = 0;

    // 扫描 cpu0..cpu255；缺目录或无 scaling_cur_freq 则跳过该核。
    // 例：Orin 常见 6 核 online，各核 cur 可能不同（729600 vs 1728000）。
    for (int i = 0; i < 256; ++i) {
        char cur_path[128];
        char max_path[128];
        std::snprintf(cur_path, sizeof(cur_path),
                      "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", i);
        std::snprintf(max_path, sizeof(max_path),
                      "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", i);

        long long cur = 0;
        if (!read_ll_file(cur_path, &cur) || cur <= 0) {
            continue;
        }
        const int cur_i = static_cast<int>(cur);
        if (n == 0) {
            cur_min = cur_i;
            cur_max = cur_i;
        } else {
            if (cur_i < cur_min) {
                cur_min = cur_i;
            }
            if (cur_i > cur_max) {
                cur_max = cur_i;
            }
        }
        sum += cur;
        ++n;

        long long mx = 0;
        if (read_ll_file(max_path, &mx) && mx > hw_max) {
            hw_max = static_cast<int>(mx);
        }
    }

    if (n <= 0) {
        return -1;
    }
    out->num_cores = n;
    out->cur_khz_min = cur_min;
    out->cur_khz_max = cur_max;
    out->cur_khz_avg = static_cast<int>(sum / n);
    out->max_khz = hw_max;
    return 0;
}

int freq_probe_gpu_query(FreqProbeGpuSnapshot *out) {
    if (out == nullptr) {
        return -1;
    }
    std::memset(out, 0, sizeof(*out));

    // 优先固定 Orin 路径；找不到再扫 /sys/class/devfreq 里 name 含 gpu 的节点。
    static const char *kPrefer[] = {
        "/sys/class/devfreq/17000000.gpu",
        "/sys/devices/platform/gpu.0/devfreq/17000000.gpu",
        nullptr,
    };

    char chosen[256] = {0};
    for (int i = 0; kPrefer[i] != nullptr; ++i) {
        char cur_path[300];
        std::snprintf(cur_path, sizeof(cur_path), "%s/cur_freq", kPrefer[i]);
        long long cur_hz = 0;
        if (read_ll_file(cur_path, &cur_hz) && cur_hz > 0) {
            std::snprintf(chosen, sizeof(chosen), "%s", kPrefer[i]);
            break;
        }
    }

    if (chosen[0] == '\0') {
        DIR *dir = opendir("/sys/class/devfreq");
        if (dir == nullptr) {
            return -1;
        }
        while (const dirent *ent = readdir(dir)) {
            if (ent->d_name[0] == '.') {
                continue;
            }
            char name_path[320];
            char name_buf[64];
            std::snprintf(name_path, sizeof(name_path), "/sys/class/devfreq/%s/name", ent->d_name);
            if (!read_trimmed_line(name_path, name_buf, sizeof(name_buf))) {
                continue;
            }
            // name 常为 "17000000.gpu" 或 "gpu"
            if (std::strstr(name_buf, "gpu") == nullptr) {
                continue;
            }
            char candidate[320];
            std::snprintf(candidate, sizeof(candidate), "/sys/class/devfreq/%s", ent->d_name);
            if (std::strlen(candidate) >= sizeof(chosen)) {
                continue;
            }
            std::snprintf(chosen, sizeof(chosen), "%s", candidate);
            break;
        }
        closedir(dir);
    }

    if (chosen[0] == '\0') {
        return -1;
    }

    char cur_path[300];
    char max_path[300];
    std::snprintf(cur_path, sizeof(cur_path), "%s/cur_freq", chosen);
    std::snprintf(max_path, sizeof(max_path), "%s/max_freq", chosen);

    long long cur_hz = 0;
    long long max_hz = 0;
    if (!read_ll_file(cur_path, &cur_hz) || cur_hz <= 0) {
        return -1;
    }
    // max 读失败时仍返回 cur；max_khz=0 表示未知
    (void)read_ll_file(max_path, &max_hz);

    // sysfs GPU 是 Hz；统一成 kHz，方便和 CPU / DeviceHwInfo.sm_clock_khz 对照
    out->cur_khz = static_cast<int>(cur_hz / 1000);
    out->max_khz = max_hz > 0 ? static_cast<int>(max_hz / 1000) : 0;
    std::snprintf(out->path, sizeof(out->path), "%s", chosen);
    return 0;
}

int freq_probe_print(const char *tag) {
    FreqProbeCpuSnapshot cpu{};
    FreqProbeGpuSnapshot gpu{};
    const int cpu_ok = freq_probe_cpu_query(&cpu);
    const int gpu_ok = freq_probe_gpu_query(&gpu);
    if (cpu_ok != 0 && gpu_ok != 0) {
        std::fprintf(stderr, "[freq] query failed (cpu and gpu)\n");
        return -1;
    }

    // 多行对齐，避免一行堆字段；例：
    // -------- [freq] test_query --------
    //   CPU  cores=6
    //        cur   min= 729.6  avg= 985.6  max=1113.6  MHz
    //        hw_max=1728.0 MHz
    //   GPU  cur= 306.0 / max=1020.0 MHz
    // ------------------------------------
    const char *label = (tag != nullptr && tag[0] != '\0') ? tag : "-";
    std::printf("-------- [freq] %s --------\n", label);
    if (cpu_ok == 0) {
        std::printf("  CPU  cores=%d\n", cpu.num_cores);
        std::printf("       cur   min=%7.1f  avg=%7.1f  max=%7.1f  MHz\n",
                    khz_to_mhz(cpu.cur_khz_min), khz_to_mhz(cpu.cur_khz_avg),
                    khz_to_mhz(cpu.cur_khz_max));
        std::printf("       hw_max=%7.1f MHz\n", khz_to_mhz(cpu.max_khz));
    } else {
        std::printf("  CPU  NA\n");
    }
    if (gpu_ok == 0) {
        std::printf("  GPU  cur=%7.1f / max=%7.1f MHz\n", khz_to_mhz(gpu.cur_khz),
                    khz_to_mhz(gpu.max_khz));
    } else {
        std::printf("  GPU  NA\n");
    }
    std::printf("--------------------------------\n");
    std::fflush(stdout);
    return 0;
}
