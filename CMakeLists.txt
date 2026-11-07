cmake_minimum_required(VERSION 3.24)
project(CelV2 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
# Turing (75), Ampere (80, 86), Ada (89), Hopper (90): SASS only; Blackwell (120): SASS + PTX (forward-compatible).
set(CMAKE_CUDA_ARCHITECTURES 75-real 80-real 86-real 89-real 90-real 120)
add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>")
# 8-wide float loops in fwd_dct; MSVC does not contract to FMA under /fp:precise, so results match SSE2.
add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:/arch:AVX2>")
add_compile_definitions(NOMINMAX)  # AVM headers pull in windows.h

include(cmake/avm.cmake)

# Stage 3's replay of avmenc's decisions (tools/oracle, docs/superpowers/specs/2026-11-04-av1-gap-breakdown-design.md):
# measurement only, never in a release.
option(CELV2_ORACLE "Build celv2enc with the replay, and oracle_dump" OFF)
if(CELV2_ORACLE)
  add_compile_definitions(CELV2_ORACLE)
endif()

add_library(celv2_core STATIC
  src/io/y4m.cpp
  src/io/ivf.cpp
  src/io/stats.cpp
  src/io/memory.cpp
  src/core/size_control.cpp
  src/core/vmaf_probe.cpp
  src/core/film_grain.cpp
  src/core/perceptual_aq.cpp
  src/core/psy.cpp
  src/core/screen_gate.cpp
  src/core/intrabc_index.cpp
  src/core/profile.cpp
  src/verify/avm_decode.cpp
  src/bitstream/obu.cpp
  src/bitstream/seq_header.cpp
  src/bitstream/frame_header.cpp
  src/bitstream/block_writer.cpp
  src/state/codec_state.cpp
  src/core/encoder.cpp
  src/core/global_motion.cpp
  src/core/rd_pass.cpp
  src/core/quant.cpp
  src/core/ccso_search.cpp
  src/core/ccso_apply.cpp
  src/core/tpl.cpp
  src/core/temporal_filter.cpp
  src/core/threads.cpp
  src/core/scene_cut.cpp
  src/core/transform.cpp
  src/core/ist.cpp
  src/gpu/decision_map.cpp
  ${AVM_ROOT}/common/md5_utils.c)
target_include_directories(celv2_core PUBLIC src)
target_link_libraries(celv2_core PUBLIC avm)
if(CELV2_ORACLE)
  target_sources(celv2_core PRIVATE tools/oracle/oracle.cpp)
  target_include_directories(celv2_core PUBLIC tools/oracle)
  # The inspection build of the decoder (scripts/build_avm_lib.ps1's sources, CONFIG_INSPECTION and
  # CONFIG_ACCOUNTING on): its own process, its structs differ from avm.lib's.
  set(AVM_INSPECT_BUILD "${AVM_ROOT}/_build_inspect" CACHE PATH "AVM inspection build directory")
  # The inspection buffers leave out TCQ's levels: a copy of decodetxb.c that stores them, in the build directory (the
  # AVM tree stays as it is), linked ahead of avm.lib's.
  file(READ "${AVM_ROOT}/av2/decoder/decodetxb.c" txb)
  string(REPLACE "\r\n" "\n" txb "${txb}")
  string(REPLACE "        tcoeffs[pos] = sign ? -level : level;\n      } else {"
                 "        tcoeffs[pos] = sign ? -level : level;\n        quant_coeffs[pos] = tcoeffs[pos];\n      } else {" txb "${txb}")
  string(REPLACE "        tcoeffs[pos] = clamp(dq_coeff, min_value, max_value);\n      }\n      // state transition"
                 "        tcoeffs[pos] = clamp(dq_coeff, min_value, max_value);\n        tcoeffs_copy[pos] = tcoeffs[pos];\n      }\n      // state transition" txb "${txb}")
  file(WRITE "${CMAKE_BINARY_DIR}/decodetxb_levels.c" "${txb}")
  add_executable(oracle_dump tools/oracle/oracle_dump.c "${CMAKE_BINARY_DIR}/decodetxb_levels.c")
  target_include_directories(oracle_dump PRIVATE "${AVM_ROOT}" "${AVM_INSPECT_BUILD}" "${AVM_INSPECT_BUILD}/config" tools/oracle)
  target_link_libraries(oracle_dump PRIVATE "${AVM_INSPECT_BUILD}/avm.lib")
endif()

# GPU decision engine (CUDA). kernels_ref.cpp is the CPU reference used by tests.
add_library(celv2_gpu STATIC
  src/gpu/engine.cu
  src/gpu/kernels.cu
  src/gpu/kernels_ref.cpp
  src/gpu/cpu_engine.cpp
  src/pipeline.cpp
  src/chunk_coder.cpp)
target_link_libraries(celv2_gpu PUBLIC celv2_core)

# The Vulkan decision engine, the default GPU path.
include(cmake/vulkan.cmake)
if(CELV2_VULKAN)
  add_library(celv2_vk STATIC src/vk/vk_context.cpp src/vk/vk_kernels.cpp src/vk/vk_engine.cpp)
  target_include_directories(celv2_vk PUBLIC src "${CELV2_VK_INCLUDE}")
  target_link_libraries(celv2_vk PUBLIC celv2_core)
  celv2_vk_shaders(celv2_vk downsample subpel me satd cost finalize)
  target_link_libraries(celv2_gpu PUBLIC celv2_vk)
  target_compile_definitions(celv2_gpu PRIVATE CELV2_HAVE_VULKAN)
  add_executable(engine_bench tools/engine_bench.cpp)  # the decision engines alone, Vulkan against the CPU
  target_link_libraries(engine_bench PRIVATE celv2_gpu celv2_vk)
endif()

add_executable(celv2enc src/main.cpp src/utf8.manifest)
target_link_libraries(celv2enc PRIVATE celv2_core celv2_gpu)

enable_testing()
function(celv2_test name)
  add_executable(${name} tests/${name}.cpp)
  target_link_libraries(${name} PRIVATE celv2_core celv2_gpu)
  target_include_directories(${name} PRIVATE tests)
  target_compile_definitions(${name} PRIVATE CELV2_TEST_DATA="${CMAKE_SOURCE_DIR}/tests/data")
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES TIMEOUT 300)
endfunction()

celv2_test(test_avm_decode)
celv2_test(test_io)
celv2_test(test_seq_header)
celv2_test(test_keyframe_skip)
celv2_test(test_keyframe_residual)
celv2_test(test_gpu_intra)
celv2_test(test_keyframe_gpu)
celv2_test(test_inter_header)
celv2_test(test_gpu_me)
celv2_test(test_ippp)
celv2_test(test_pipeline_sync)
celv2_test(test_random_decisions)
celv2_test(test_transform)
celv2_test(test_ist)
celv2_test(test_cli)
celv2_test(test_vmaf_search)
celv2_test(test_perceptual_aq)
celv2_test(test_psy)
celv2_test(test_warp)
celv2_test(test_screen)
set_tests_properties(test_cli PROPERTIES TIMEOUT 1200)  # many encodes (--workers, --target-size): about 7 minutes on a busy machine
celv2_test(test_row_gate)
celv2_test(test_cpu_engine)
celv2_test(test_scene_cut)
celv2_test(test_ccso)
celv2_test(test_ccso_apply)
celv2_test(test_tpl)
celv2_test(test_temporal_filter)
celv2_test(test_rd_pass)
celv2_test(test_threads)
celv2_test(test_size_control)
if(CELV2_VULKAN)
  celv2_test(test_vk_kernels)
  target_link_libraries(test_vk_kernels PRIVATE celv2_vk)
endif()
target_compile_definitions(test_cli PRIVATE CELV2ENC_PATH="$<TARGET_FILE:celv2enc>")
add_dependencies(test_cli celv2enc)
if(CELV2_ORACLE)
  target_compile_definitions(test_cli PRIVATE ORACLE_DUMP_PATH="$<TARGET_FILE:oracle_dump>")
  add_dependencies(test_cli oracle_dump)
endif()
add_test(NAME no_encoder_includes
  COMMAND ${CMAKE_COMMAND} -DSRC_DIR=${CMAKE_SOURCE_DIR}/src -P ${CMAKE_SOURCE_DIR}/tests/no_encoder_includes.cmake)
add_test(NAME gui_tests
  COMMAND powershell -NoProfile -ExecutionPolicy Bypass -File ${CMAKE_SOURCE_DIR}/gui/test.ps1)
# RUN_SERIAL: runner_cancel_stops_quickly counts every celv2enc and ffmpeg on the machine, test_cli's included.
set_tests_properties(gui_tests PROPERTIES TIMEOUT 300 RUN_SERIAL TRUE)
