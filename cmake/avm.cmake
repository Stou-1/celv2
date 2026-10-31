# Imports AVM v1.0.0 built by scripts/build_avm_lib.ps1 (clang-cl, no LTO) as target `avm`.
set(AVM_ROOT "$ENV{AVM_ROOT}" CACHE PATH "AVM source tree (tag v1.0.0); defaults to $AVM_ROOT")
if(NOT AVM_ROOT)
  message(FATAL_ERROR "set AVM_ROOT (environment or -DAVM_ROOT=...) to the AVM v1.0.0 source tree")
endif()
set(AVM_LIB_BUILD "${AVM_ROOT}/_build_lib" CACHE PATH "AVM library build directory")

if(NOT EXISTS "${AVM_LIB_BUILD}/avm.lib")
  message(FATAL_ERROR "avm.lib not found; run scripts/build_avm_lib.ps1 first")
endif()

# avm.lib's ML-based objects reference TFLite and its dependencies; link every library the AVM build made.
file(GLOB_RECURSE AVM_DEP_LIBS "${AVM_LIB_BUILD}/*.lib")
list(FILTER AVM_DEP_LIBS EXCLUDE REGEX "/avm\\.lib$")

add_library(avm STATIC IMPORTED GLOBAL)
set_target_properties(avm PROPERTIES
  IMPORTED_LOCATION "${AVM_LIB_BUILD}/avm.lib"
  INTERFACE_INCLUDE_DIRECTORIES "${AVM_ROOT};${AVM_LIB_BUILD}"
  INTERFACE_LINK_LIBRARIES "${AVM_DEP_LIBS}")
