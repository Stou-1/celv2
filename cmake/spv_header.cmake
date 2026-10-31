# cmake -DSPV=<in.spv> -DHDR=<out.h> -DNAME=<name> -P spv_header.cmake: SPIR-V words as a C++ array.
file(READ "${SPV}" hex HEX)
string(LENGTH "${hex}" n)
math(EXPR rem "${n} % 8")
if(n EQUAL 0 OR NOT rem EQUAL 0)
  message(FATAL_ERROR "${SPV}: not a SPIR-V module")
endif()
# Little-endian bytes b0 b1 b2 b3 -> 0xb3b2b1b0.
string(REGEX REPLACE "(..)(..)(..)(..)" "0x\\4\\3\\2\\1u," words "${hex}")
file(WRITE "${HDR}" "// Generated from ${NAME}.slang by cmake/spv_header.cmake.\n#pragma once\n#include <cstdint>\n"
  "inline constexpr uint32_t kSpv_${NAME}[] = {${words}};\n")
