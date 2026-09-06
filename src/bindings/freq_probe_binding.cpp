#include "freq_probe.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>

namespace py = pybind11;

PYBIND11_MODULE(freq_probe_me, m) {
    m.doc() = "CPU/GPU frequency probe: sysfs realtime cur vs max (Jetson Orin)";

    m.def(
        "query_cpu",
        []() {
            FreqProbeCpuSnapshot s{};
            if (freq_probe_cpu_query(&s) != 0) {
                throw std::runtime_error("freq_probe_cpu_query failed");
            }
            py::dict out;
            out["num_cores"] = s.num_cores;
            out["cur_khz_min"] = s.cur_khz_min;
            out["cur_khz_max"] = s.cur_khz_max;
            out["cur_khz_avg"] = s.cur_khz_avg;
            out["max_khz"] = s.max_khz;
            out["cur_mhz_avg"] = s.cur_khz_avg / 1000.0;
            out["max_mhz"] = s.max_khz / 1000.0;
            return out;
        },
        "Read cpufreq scaling_cur_freq / cpuinfo_max_freq for online cores");

    m.def(
        "query_gpu",
        []() {
            FreqProbeGpuSnapshot s{};
            if (freq_probe_gpu_query(&s) != 0) {
                throw std::runtime_error("freq_probe_gpu_query failed");
            }
            py::dict out;
            out["cur_khz"] = s.cur_khz;
            out["max_khz"] = s.max_khz;
            out["cur_mhz"] = s.cur_khz / 1000.0;
            out["max_mhz"] = s.max_khz / 1000.0;
            out["path"] = std::string(s.path);
            return out;
        },
        "Read GPU devfreq cur_freq / max_freq (Hz in sysfs, returned as kHz/MHz)");

    m.def(
        "query",
        []() {
            py::dict out;
            FreqProbeCpuSnapshot cpu{};
            FreqProbeGpuSnapshot gpu{};
            if (freq_probe_cpu_query(&cpu) == 0) {
                py::dict c;
                c["num_cores"] = cpu.num_cores;
                c["cur_khz_min"] = cpu.cur_khz_min;
                c["cur_khz_max"] = cpu.cur_khz_max;
                c["cur_khz_avg"] = cpu.cur_khz_avg;
                c["max_khz"] = cpu.max_khz;
                c["cur_mhz_avg"] = cpu.cur_khz_avg / 1000.0;
                c["max_mhz"] = cpu.max_khz / 1000.0;
                out["cpu"] = c;
            } else {
                out["cpu"] = py::none();
            }
            if (freq_probe_gpu_query(&gpu) == 0) {
                py::dict g;
                g["cur_khz"] = gpu.cur_khz;
                g["max_khz"] = gpu.max_khz;
                g["cur_mhz"] = gpu.cur_khz / 1000.0;
                g["max_mhz"] = gpu.max_khz / 1000.0;
                g["path"] = std::string(gpu.path);
                out["gpu"] = g;
            } else {
                out["gpu"] = py::none();
            }
            if (out["cpu"].is_none() && out["gpu"].is_none()) {
                throw std::runtime_error("freq_probe query: both cpu and gpu failed");
            }
            return out;
        },
        "Return {cpu: {...}|None, gpu: {...}|None}");

    m.def(
        "print_freq",
        [](const std::string &tag) {
            if (freq_probe_print(tag.c_str()) != 0) {
                throw std::runtime_error("freq_probe_print failed");
            }
        },
        py::arg("tag") = "", "Print multi-line CPU cur min/avg/max vs hw_max and GPU cur vs max");
}
