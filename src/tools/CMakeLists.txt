# 诊断工具 OBJECT：显存 / 频率探针（非 Model 生产路径）

add_library(vram_probe_model OBJECT
  vram_probe.cpp
)

add_library(freq_probe_model OBJECT
  freq_probe.cpp
)

target_link_libraries(vram_probe_model PRIVATE CUDA::cudart)
set_target_properties(vram_probe_model PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(vram_probe_model PRIVATE ${MY_OPS_INCLUDE_DIRS})

# freq_probe 只读 sysfs，不链 cudart
set_target_properties(freq_probe_model PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(freq_probe_model PRIVATE ${MY_OPS_INCLUDE_DIRS})

if(CMAKE_BUILD_TYPE MATCHES "Debug")
  target_compile_definitions(vram_probe_model PRIVATE MY_OPS_DEBUG=1)
  target_compile_definitions(freq_probe_model PRIVATE MY_OPS_DEBUG=1)
endif()
