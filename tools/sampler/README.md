# sampler

A sampling profiler that needs no administrator rights (nsys CPU sampling does): it runs a command
as a child process, reads its threads' instruction pointers about every millisecond, and after the
child exits resolves them with the modules' PDBs. It prints self time per function; samples in the
system's wait functions are reported as "waiting" and left out of the percentages.

Build (from a shell set up by `scripts\env.ps1`):

    cl /nologo /O2 /EHsc /std:c++17 tools\sampler\sampler.cpp

A celv2enc with symbols, optimized as the release build:

    cmake -S . -B build_prof -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl "-DCMAKE_CXX_FLAGS_RELEASE=/O2 /Ob2 /DNDEBUG /Zi" "-DCMAKE_C_FLAGS_RELEASE=/O2 /Ob2 /DNDEBUG /Zi" "-DCMAKE_EXE_LINKER_FLAGS_RELEASE=/DEBUG /OPT:REF /OPT:ICF"
    cmake --build build_prof --target celv2enc

Run:

    sampler.exe --top 60 -- build_prof\celv2enc.exe -i in.y4m -o out.ivf --qindex 130

Self time only (no call stacks); AVM functions show by their public names.
