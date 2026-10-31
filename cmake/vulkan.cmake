# Vulkan decision engine: the Vulkan headers and slangc, from the LunarG SDK ($VULKAN_SDK), the PATH,
# or portable copies unpacked under %LOCALAPPDATA%\bin (Vulkan-Headers-<version>, slang-<version>).
# Without them, CELV2_VULKAN is OFF and nothing Vulkan is built. No vulkan-1.lib is needed:
# vk_context.cpp opens the system loader with LoadLibrary.
file(GLOB _vk_headers "$ENV{LOCALAPPDATA}/bin/Vulkan-Headers-*/include")
file(GLOB _vk_slang "$ENV{LOCALAPPDATA}/bin/slang-*/bin")
find_path(CELV2_VK_INCLUDE vulkan/vulkan.h HINTS "$ENV{VULKAN_SDK}/Include" ${_vk_headers})
find_program(CELV2_SLANGC slangc HINTS "$ENV{VULKAN_SDK}/Bin" ${_vk_slang})
if(CELV2_VK_INCLUDE AND CELV2_SLANGC)
  set(CELV2_VULKAN ON)
  message(STATUS "Vulkan: headers ${CELV2_VK_INCLUDE}, slangc ${CELV2_SLANGC}")
else()
  set(CELV2_VULKAN OFF)
  message(STATUS "Vulkan: off (no Vulkan headers or slangc)")
endif()

# celv2_vk_shaders(<target> <name>...): src/vk/shaders/<name>.slang -> SPIR-V -> a header
# vk_shaders/<name>.h holding `kSpv_<name>` (uint32_t words), on <target>'s include path.
function(celv2_vk_shaders target)
  set(out_dir "${CMAKE_BINARY_DIR}/vk_shaders")
  set(headers)
  foreach(name IN LISTS ARGN)
    set(src "${CMAKE_SOURCE_DIR}/src/vk/shaders/${name}.slang")
    set(spv "${out_dir}/${name}.spv")
    set(hdr "${out_dir}/${name}.h")
    add_custom_command(OUTPUT "${hdr}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${out_dir}"
      COMMAND "${CELV2_SLANGC}" "${src}" -target spirv -profile spirv_1_5 -I "${CMAKE_SOURCE_DIR}/src/vk/shaders" -entry main -stage compute -O3 -o "${spv}"
      COMMAND "${CMAKE_COMMAND}" -DSPV=${spv} -DHDR=${hdr} -DNAME=${name} -P "${CMAKE_SOURCE_DIR}/cmake/spv_header.cmake"
      DEPENDS "${src}" "${CMAKE_SOURCE_DIR}/src/vk/shaders/common.slang" "${CMAKE_SOURCE_DIR}/cmake/spv_header.cmake"
      COMMENT "slangc ${name}.slang")
    list(APPEND headers "${hdr}")
  endforeach()
  target_sources(${target} PRIVATE ${headers})
  target_include_directories(${target} PRIVATE "${CMAKE_BINARY_DIR}")
endfunction()
