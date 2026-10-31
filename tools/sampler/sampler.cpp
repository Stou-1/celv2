// Sampling profiler without administrator rights: runs a command as a child process, suspends its
// threads about every millisecond to read their instruction pointers, records where its modules are
// loaded, and after it exits resolves the samples with DbgHelp (the modules' PDBs, loaded offline at
// the recorded bases). Prints self time per function; samples in the system's wait functions are
// counted apart as "(waiting)" and left out of the percentages.
// usage: sampler.exe [--top N] -- command line...
#include <windows.h>
#include <dbghelp.h>
#include <psapi.h>
#include <tlhelp32.h>

#include <algorithm>
#include <cstdio>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#pragma comment(lib, "dbghelp.lib")
#pragma comment(lib, "psapi.lib")

struct Module {
  std::wstring path;
  uint64_t base, size;
};

int wmain(int argc, wchar_t** argv) {
  int top = 40, first = 1;
  if (argc > 2 && std::wstring(argv[1]) == L"--top") top = _wtoi(argv[2]), first = 3;
  if (first < argc && std::wstring(argv[first]) == L"--") first++;
  if (first >= argc) {
    std::fprintf(stderr, "usage: sampler.exe [--top N] -- command...\n");
    return 2;
  }
  std::wstring cmd;
  for (int i = first; i < argc; i++) cmd += (i > first ? L" " : L"") + std::wstring(L"\"") + argv[i] + L"\"";
  STARTUPINFOW si{sizeof(si)};
  PROCESS_INFORMATION pi{};
  if (!CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) {
    std::fprintf(stderr, "CreateProcess failed: %lu\n", GetLastError());
    return 1;
  }
  std::unordered_map<uint64_t, uint64_t> hits;
  std::map<DWORD, HANDLE> threads;
  std::map<uint64_t, Module> modules;  // by base
  auto refresh = [&] {
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    THREADENTRY32 te{sizeof(te)};
    for (BOOL ok = Thread32First(snap, &te); ok; ok = Thread32Next(snap, &te))
      if (te.th32OwnerProcessID == pi.dwProcessId && !threads.count(te.th32ThreadID))
        if (HANDLE h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT, FALSE, te.th32ThreadID))
          threads[te.th32ThreadID] = h;
    CloseHandle(snap);
    HMODULE mods[1024];
    DWORD bytes = 0;
    if (EnumProcessModulesEx(pi.hProcess, mods, sizeof(mods), &bytes, LIST_MODULES_ALL))
      for (DWORD i = 0; i < bytes / sizeof(HMODULE); i++) {
        const uint64_t base = reinterpret_cast<uint64_t>(mods[i]);
        if (modules.count(base)) continue;
        wchar_t path[MAX_PATH];
        MODULEINFO mi{};
        if (GetModuleFileNameExW(pi.hProcess, mods[i], path, MAX_PATH) &&
            GetModuleInformation(pi.hProcess, mods[i], &mi, sizeof(mi)))
          modules[base] = Module{path, base, mi.SizeOfImage};
      }
  };
  uint64_t samples = 0;
  for (int tick = 0; WaitForSingleObject(pi.hProcess, 1) == WAIT_TIMEOUT; tick++) {
    if (tick % 50 == 0) refresh();
    for (auto it = threads.begin(); it != threads.end();) {
      if (SuspendThread(it->second) == static_cast<DWORD>(-1)) {
        CloseHandle(it->second);
        it = threads.erase(it);
        continue;
      }
      CONTEXT ctx{};
      ctx.ContextFlags = CONTEXT_CONTROL;
      if (GetThreadContext(it->second, &ctx)) hits[ctx.Rip]++, samples++;
      ResumeThread(it->second);
      ++it;
    }
  }
  DWORD code = 0;
  GetExitCodeProcess(pi.hProcess, &code);

  // Offline symbols: a pseudo process handle with every module loaded at its recorded base.
  const HANDLE sym = reinterpret_cast<HANDLE>(static_cast<uintptr_t>(0x5a5a));
  SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_FAIL_CRITICAL_ERRORS);
  if (!SymInitializeW(sym, nullptr, FALSE)) {
    std::fprintf(stderr, "SymInitialize failed: %lu\n", GetLastError());
    return 1;
  }
  for (const auto& [base, m] : modules)
    SymLoadModuleExW(sym, nullptr, m.path.c_str(), nullptr, m.base, static_cast<DWORD>(m.size), nullptr, 0);
  std::map<std::string, uint64_t> by_func;
  uint64_t waiting = 0, counted = 0;
  std::vector<char> buf(sizeof(SYMBOL_INFO) + 512);
  for (const auto& [addr, n] : hits) {
    std::string mod = "?";
    auto it = modules.upper_bound(addr);
    if (it != modules.begin()) {
      --it;
      if (addr < it->second.base + it->second.size) {
        const std::wstring& p = it->second.path;
        const std::wstring file = p.substr(p.find_last_of(L"\\/") + 1);
        mod.assign(file.begin(), file.end());
      }
    }
    auto* si2 = reinterpret_cast<SYMBOL_INFO*>(buf.data());
    si2->SizeOfStruct = sizeof(SYMBOL_INFO);
    si2->MaxNameLen = 511;
    DWORD64 disp = 0;
    std::string name = SymFromAddr(sym, addr, &disp, si2) ? std::string(si2->Name) : "?";
    const bool sys = mod == "ntdll.dll" || mod == "KERNELBASE.dll" || mod == "kernel32.dll" || mod == "KERNEL32.DLL";
    if (sys && (name.find("Wait") != std::string::npos || name.find("Delay") != std::string::npos ||
                name.find("Remove") != std::string::npos || name.find("Sleep") != std::string::npos)) {
      waiting += n;
      continue;
    }
    by_func[mod + "!" + name] += n;
    counted += n;
  }
  std::vector<std::pair<uint64_t, std::string>> rows;
  for (const auto& [f, n] : by_func) rows.push_back({n, f});
  std::sort(rows.rbegin(), rows.rend());
  std::printf("child exit %lu, %llu samples, %.1f%% waiting\n", code, static_cast<unsigned long long>(samples),
              samples ? 100.0 * waiting / samples : 0.0);
  for (int i = 0; i < top && i < static_cast<int>(rows.size()); i++)
    std::printf("%6.2f%%  %s\n", 100.0 * rows[i].first / counted, rows[i].second.c_str());
  SymCleanup(sym);
  return static_cast<int>(code);
}
