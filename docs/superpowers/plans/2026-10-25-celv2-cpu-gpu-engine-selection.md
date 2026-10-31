# 判定エンジンの CPU / GPU 自動選択と複数世代ビルド Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** GPU が使えない・古い PC でも celv2enc が動くよう、起動時に GPU を調べて判定を GPU（`GpuEngine`）か CPU（新しい `CpuEngine`）で行い、Turing 以降の全世代向けの GPU コードを1つの実行ファイルに入れる。

**Architecture:** 共通の抽象クラス `DecisionEngine`（`launch` / `collect`）を `GpuEngine` と `CpuEngine` が実装し、`Pipeline` は `make_engine` が選んだものを使う。`CpuEngine` は参照実装 `kernels_ref.cpp` を行単位に分け直したものを複数スレッドで動かし、行を上から順に `RowGate` で公開する。判定は整数だけなので、出力はどちらでもバイト単位で同じ。

**Tech Stack:** C++17、CUDA 13.3、MSVC + Ninja、AVM v1.0.0、CTest。

**Spec:** `docs/superpowers/specs/2026-10-25-celv2-cpu-gpu-engine-selection-design.md`

## Global Constraints

- CPU 版と GPU 版の出力はバイト単位で同じ。GPU 版の出力は今の main と同じ（bench.y4m 720p と 1080p、30枚、q150、1タイルと2タイル）。
- GPU で判定する条件: `cudaGetDeviceCount` 成功かつ 1 以上、デバイス 0 の compute capability 7.5 以上、ビルドの GPU コードが実際に動く。使うのはデバイス 0 だけ。
- `CMAKE_CUDA_ARCHITECTURES` は `75-real;80-real;86-real;89-real;90-real;120`。CUDA ランタイムは静的リンクのまま。
- `--device auto|gpu|cpu`（既定 `auto`）。表示は `CelV2: decisions on GPU (<name>, compute <major>.<minor>)` / `CelV2: decisions on CPU (<reason>)`。`--device gpu` で使えなければ終了コード 1。
- 判定の作業スレッド数 = max(1, コア数 − タイル符号化のスレッド数)。コア数は `std::thread::hardware_concurrency()`（0 なら 1）。
- `DecisionEngine`、`ProbeResult`、`RowGate`、`DecisionMap` は CUDA にも AVM にも依存しない。
- av2/encoder を使わない（CTest `no_encoder_includes`）。コメントは英語。
- コミットはリポジトリの設定の作者で行う（`-c user.name/user.email` を付けない）。末尾に `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`。
- ビルド: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build.ps1`、テスト: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\test.ps1 [-R name]`（今は 15 本）。

## Review Focus

1. CPU 版で、行数よりスレッドがずっと多いフレーム（32x32 は1行、16x16 も1行で、作業スレッドは 46 本）: 余ったスレッドがすぐ終わり、止まらないこと → Task 3 の `cpu_engine_tiny_frames`。
2. 高さ・幅が 64 や 8 の倍数でないフレーム（202x122）で、CPU 版の動き探索の行分担の端が正しいこと → Task 1 の `me_rows_split_matches_whole` と Task 3 の判定一致。
3. 長い列（30 フレーム以上）で、フレームごとに作業スレッドが必ず join され、スレッドが残らないこと → Task 3 の `cpu_engine_many_frames`。
4. GPU がない環境で `--device gpu` を指定したとき、理由を表示して終了コード 1 で止まること（CPU に切り替えない） → Task 4 の `cli_device_gpu_without_gpu_fails`。
5. KEY と P が混ざる列（keyint）で、CPU 版の参照画像の有無が正しく切り替わること → Task 4 の keyint 付きのバイト一致。

---

### Task 1: 参照実装を行単位に分け直す

**Files:**
- Modify: `src/gpu/kernels_ref.h`、`src/gpu/kernels_ref.cpp`
- Modify: `tests/test_gpu_me.cpp`（テストを追加）

**Interfaces:**
- Produces（`kernels_ref.h`）:
  ```cpp
  struct RefPyramid {                       // luma pyramid, CelV2::kMeLevels levels
    std::vector<std::vector<uint8_t>> px;   // level l pixels
    std::vector<CelV2::Plane> plane;         // level l view (p points into px[l])
    RefPyramid() = default;
    RefPyramid(const RefPyramid&) = delete;  // a copy would keep pointing at the original's pixels
    RefPyramid& operator=(const RefPyramid&) = delete;
    RefPyramid(RefPyramid&&) = default;      // moving keeps the vectors' buffers, so the views stay valid
    RefPyramid& operator=(RefPyramid&&) = default;
  };
  RefPyramid build_pyramid_ref(const Frame420& f);
  struct RefMvLevel { std::vector<int> mvx, mvy; int cols = 0, rows = 0; };  // per-8x8 MVs of one level
  // Searches 8x8 block rows [row0, row1) of level l; coarser == nullptr at the top level.
  // out must be sized (cols x rows of level l) by the caller.
  void me_level_rows_ref(const RefPyramid& cur, const RefPyramid& ref, int l, const RefMvLevel* coarser,
                         int row0, int row1, RefMvLevel& out);
  // One superblock's decision; level0 == nullptr: intra only.
  SbDecision decide_sb_ref(const Frame420& src, const RefPyramid* ref, const RefMvLevel* level0,
                           int sb_col, int sb_row, int sb_size, int qindex);
  ```
  `kernels_ref.h` は `gpu/intra_common.h`（`CelV2::Plane`）を include する。`decide_intra_ref` / `decide_inter_ref` は残し、これらを呼ぶ形にする（結果は変わらない）。

- [ ] **Step 1: 失敗するテストを書く**（`tests/test_gpu_me.cpp`）
  - `me_rows_split_matches_whole`: 202x122 と 640x360 の `scene_frame` 2枚（dx=5, dy=3）で、各段を `me_level_rows_ref(..., 0, rows, out)` で1回に探索した結果と、`[0, 1)`, `[1, 4)`, `[4, rows)` の3回に分けた結果の `mvx` / `mvy` が完全に同じ。
  - `decide_sb_ref_matches_decide`: 同じ2枚で、全スーパーブロックの `decide_sb_ref` の `tree` と `blocks` が `decide_inter_ref` の結果と一致する。`ref == nullptr` で `decide_intra_ref` とも一致する。
- [ ] **Step 2: 実行して失敗を確認する** → `scripts\build.ps1` が新しい関数の未定義でエラー。
- [ ] **Step 3: 実装する**: 今の無名名前空間の `pyramid` / ME ループ / `decide` の中身を、上の関数に移す（ロジックは変えない）。`decide_inter_ref` は `build_pyramid_ref` ×2 → 上の段から `me_level_rows_ref(…, 0, rows, …)` → 全スーパーブロックの `decide_sb_ref`。
- [ ] **Step 4: テストを実行する** → `scripts\test.ps1` → 15/15 pass（`test_gpu_intra`、`test_random_decisions`、`test_gpu_me` の既存の比較も含む）。
- [ ] **Step 5: コミットする**: `refactor: CPU reference decision callable per ME row range and per superblock`

### Task 2: DecisionEngine、GPU の確認、タイルのスレッド数

**Files:**
- Create: `src/gpu/decision_engine.h`
- Modify: `src/gpu/engine.h`、`src/gpu/engine.cu`（`GpuEngine : public DecisionEngine`、`probe_gpu`）
- Modify: `src/core/encoder.h`、`src/core/encoder.cpp`（`tile_threads`）
- Modify: `tests/test_gpu_intra.cpp`、`tests/test_ippp.cpp`（テストを追加）

**Interfaces:**
- Consumes: 既存の `GpuEngine::launch` / `collect`、`RowGate`。
- Produces:
  ```cpp
  // decision_engine.h (no CUDA, no AVM)
  class DecisionEngine {
   public:
    virtual ~DecisionEngine() = default;
    virtual void launch(const Frame420& src, const Frame420* ref, int qindex, DecisionMap& dm) = 0;
    virtual void collect(DecisionMap& dm, RowGate& gate) = 0;
  };
  struct ProbeResult { bool ok = false; std::string name; int cc_major = 0, cc_minor = 0; std::string reason; };
  // engine.h
  ProbeResult probe_gpu();   // the spec's three checks, device 0
  // encoder.h
  int Encoder::tile_threads() const;   // threads encode() uses: min(tiles, cores, cfg.threads > 0 ? cfg.threads : tiles)
  ```
  コメントは spec の `DecisionEngine` の節のものを使う。`Encoder::encode` のスレッド数の計算は `tile_threads()` を呼ぶ形にする（同じ式を2か所に書かない）。

- [ ] **Step 1: 失敗するテストを書く**
  - `test_gpu_intra.cpp` の `probe_gpu_reports_this_device`: `probe_gpu()` が `ok == true`、`cc_major * 10 + cc_minor >= 75`、`name` が空でない（この PC は RTX 5090 で compute 12.0）。
  - `test_ippp.cpp` の `tile_threads_counts`: 640x360 で 2x1 タイル → `Encoder::tile_threads() == 2`（コア数が 2 以上の前提。`std::thread::hardware_concurrency() < 2` ならこの確認だけ飛ばす）、`cfg.threads = 1` → 1、1タイル → 1。
- [ ] **Step 2: 実行して失敗を確認する** → 未定義でビルドエラー。
- [ ] **Step 3: 実装する**: `probe_gpu` は spec の順で確かめ、3つ目の確認は `blocks_per_sb(1)` を `try` で呼ぶ（例外の文言を `reason` にする）。`GpuEngine` は `DecisionEngine` を継承し、`launch` / `collect` に `override` を付ける。
- [ ] **Step 4: テストを実行する** → 15/15 pass。`bench720.y4m`（`%TEMP%\celv2_bench\`）30枚 q150 `--tile-columns 1` の出力 MD5 が `596938CFC45108D927756465844D5CB8` のまま。
- [ ] **Step 5: コミットする**: `feat: DecisionEngine interface, probe_gpu, Encoder::tile_threads`

### Task 3: CpuEngine

**Files:**
- Create: `src/gpu/cpu_engine.h`、`src/gpu/cpu_engine.cpp`（`CMakeLists.txt` の `celv2_gpu` に追加）
- Create: `tests/test_cpu_engine.cpp`（`celv2_test(test_cpu_engine)`）
- Modify: `tests/test_gpu_intra.cpp`（判定一致のテストを追加）

**Interfaces:**
- Consumes: Task 1 の `build_pyramid_ref` / `me_level_rows_ref` / `decide_sb_ref`、Task 2 の `DecisionEngine`、既存の `RowGate`。
- Produces:
  ```cpp
  int decision_threads(int cores, int tile_threads);   // max(1, (cores > 0 ? cores : 1) - tile_threads)
  class CpuEngine : public DecisionEngine {
   public:
    // row_hook is called by a worker right before it decides a superblock row (tests only).
    CpuEngine(int width, int height, int sb_size, int threads,
              std::function<void(int row)> row_hook = {});
    void launch(const Frame420& src, const Frame420* ref, int qindex, DecisionMap& dm) override;
    void collect(DecisionMap& dm, RowGate& gate) override;
  };
  ```
  流れは spec の「CPU 版」の節のとおり: `launch` は src / ref を自分のバッファにコピーし `dm` を整え、`threads` 本の作業スレッドを起動して戻る。作業スレッドは（P なら）段ごとに 8x8 ブロックの行を共有カウンタで分担し、段の変わり目で全員がそろうのを待つ（`std::mutex` + `std::condition_variable` の小さなバリアでよい）。その後、共有カウンタで行を取り、`decide_sb_ref` で全スーパーブロックを `dm.sbs` に書き、完了を記録する。`collect` は上から順に完了を待って連続した行を `publish` し、最後に全スレッドを `join` する。作業スレッドの例外は最初の1つを記録して残りの作業を止め、`collect` が `gate.fail` → `join` → 投げ直し。

- [ ] **Step 1: 失敗するテストを書く**
  - `test_gpu_intra.cpp` の `cpu_engine_matches_gpu`: 202x122、640x360、32x32 の `scene_frame` 2枚で、`CpuEngine(w, h, 64, 8)` と `GpuEngine` の `launch` + `collect` の結果が、KEY（`ref == nullptr`）と P の両方で全スーパーブロックの `tree` と `blocks` まで一致する。
  - `test_cpu_engine.cpp`（各テストは待ちに 30 秒の上限、超えたら理由を表示して `std::abort`）:
    - `decision_threads_values`: `decision_threads(48, 2) == 46`、`decision_threads(4, 4) == 1`、`decision_threads(1, 1) == 1`、`decision_threads(0, 1) == 1`。
    - `cpu_engine_publishes_in_order`: 640x360 の P で、別スレッドの `collect` の間、本スレッドが `gate.ready_rows()` を繰り返し読み、値が一度も減らず、最後に `dm.sb_rows`（6）になる。
    - `cpu_engine_fails_cleanly`: `row_hook` が row 2 で `std::runtime_error("hook")` を投げる `CpuEngine` で、`collect` が `"hook"` の例外を投げ、その後 `gate.wait_row(5)` も例外を投げる（止まったままにならない）。
    - `cpu_engine_tiny_frames`: 32x32 と 16x16 を `threads = 46` で KEY と P それぞれ `launch` + `collect` し、終わること（`ready_rows() == 1`）。
    - `cpu_engine_many_frames`: 64x64 で 30 フレーム続けて `launch` + `collect`（KEY 1枚 + P 29枚）がすべて終わること。
- [ ] **Step 2: 実行して失敗を確認する** → `CpuEngine` 未定義でビルドエラー。
- [ ] **Step 3: `CpuEngine` と `decision_threads` を実装する**
- [ ] **Step 4: テストを実行する** → 16/16 pass。
- [ ] **Step 5: コミットする**: `feat: CpuEngine, a multithreaded CPU decision with row-by-row publication`

### Task 4: エンジンの選択と --device

**Files:**
- Modify: `src/pipeline.h`、`src/pipeline.cpp`
- Modify: `src/main.cpp`
- Modify: `tests/test_cpu_engine.cpp`、`tests/test_ippp.cpp`、`tests/test_cli.cpp`

**Interfaces:**
- Consumes: Task 2 の `DecisionEngine` / `ProbeResult` / `probe_gpu` / `Encoder::tile_threads`、Task 3 の `CpuEngine` / `decision_threads`。
- Produces（`pipeline.h`）:
  ```cpp
  enum class Device { Auto, Gpu, Cpu };
  // description receives the one-line choice ("decisions on GPU (...)" / "decisions on CPU (...)").
  std::unique_ptr<DecisionEngine> make_engine(Device device, int width, int height, int tile_threads,
                                              const std::function<ProbeResult()>& probe,
                                              std::string* description);
  class Pipeline {
   public:
    explicit Pipeline(const SeqConfig& c, Device device = Device::Auto);
    std::vector<uint8_t> encode(const Frame420& src, bool key);   // unchanged
    const Frame420& last_recon() const;
    const std::string& engine_description() const;
  };
  ```
  `make_engine` は `Auto` / `Gpu` / `Cpu` を spec の表どおりに扱う（`Gpu` で失敗なら `std::runtime_error("no usable GPU: " + reason)`）。`CpuEngine` の本数は `decision_threads(hardware_concurrency, tile_threads)`。`Pipeline` は `probe_gpu` を渡し、`std::unique_ptr<DecisionEngine> engine_` を持つ（`gpu_` を置き換える）。`celv2enc` は `--device auto|gpu|cpu` を受け（それ以外は使い方の誤りとして終了コード 1）、起動時に `"CelV2: " + engine_description()` を標準エラー出力に1行出す。使い方の文言（main.cpp 先頭のコメント）に `--device` を足す。

- [ ] **Step 1: 失敗するテストを書く**
  - `test_cpu_engine.cpp` の `make_engine_selection`: 偽の probe（`ok = false, reason = "fake"`）で `Auto` → 戻り値が `CpuEngine`（`dynamic_cast` で確かめる）かつ description に `"decisions on CPU"` と `"fake"` を含む。`Gpu` → `std::runtime_error`。`Cpu` → probe を呼ばずに `CpuEngine`（probe が呼ばれたら失敗にする）。
  - `test_ippp.cpp` の `cpu_pipeline_matches_gpu`: `Pipeline(c, Device::Cpu)` と `Pipeline(c, Device::Gpu)` の全フレームの TU が同じ。640x360（1 / 2x1 タイル、6枚）、202x122（2x1 タイル、4枚）、640x360 で keyint 3（2x1 タイル、7枚。KEY は `encode(src, i % 3 == 0)`）。CPU 版の列はデコード一致も確かめる。
  - `test_cli.cpp`:
    - `cli_device_cpu_matches_gpu`: 128x64 の3枚の y4m を `--device cpu` と `--device gpu` で符号化し、IVF のバイト列が同じ。stderr にそれぞれ `"decisions on CPU"` / `"decisions on GPU"`。
    - `cli_device_bogus_rejected`: `--device bogus` が 0 以外で終わる。
    - `cli_auto_falls_back_without_gpu`: `_putenv_s("CUDA_VISIBLE_DEVICES", "-1")` してから `--device auto` で符号化し（終わったら `_putenv_s("CUDA_VISIBLE_DEVICES", "")` で戻す）、終了コード 0、stderr に `"decisions on CPU"`、IVF が `cli_device_cpu_matches_gpu` の出力と同じ。
    - `cli_device_gpu_without_gpu_fails`: 同じ環境変数のもとで `--device gpu` が 0 以外で終わり、stderr が空でない。
- [ ] **Step 2: 実行して失敗を確認する** → `Device` / `make_engine` 未定義でビルドエラー。
- [ ] **Step 3: 実装する**
- [ ] **Step 4: テストを実行する** → 16/16 pass。`bench720.y4m` 30枚 q150 の既定（`auto`）の出力 MD5 が `--tile-columns 0` で `6D53C123849E0467E3901255C3337C04`、`--tile-columns 1` で `596938CFC45108D927756465844D5CB8` のまま。
- [ ] **Step 5: コミットする**: `feat: choose the CPU or GPU decision engine at startup (--device)`

### Task 5: 複数世代のビルドと計測

**Files:**
- Modify: `CMakeLists.txt`（`CMAKE_CUDA_ARCHITECTURES`）
- Modify: `README.md`（必要な GPU、`--device` の説明）
- Create: `docs/reports/2026-10-25-cpu-gpu-engine-selection.md`

- [ ] **Step 1: ビルドの設定を変える**: `set(CMAKE_CUDA_ARCHITECTURES 75-real 80-real 86-real 89-real 90-real 120)`（コメントで各世代名を書く）。全体を作り直し（`build\` を消してから `scripts\build.ps1`）、所要時間と `celv2enc.exe` の大きさを変更前と比べて記録する。
- [ ] **Step 2: 全テストを実行する** → 16/16 pass。
- [ ] **Step 3: 出力の一致を確かめる**: `%TEMP%\celv2_bench\bench720.y4m` と `src1080_30.y4m`、30枚、q150、`--tile-columns 0` と `1` の4構成で、`--device gpu` と `--device cpu` の IVF の MD5 が同じで、GPU 版は Task 4 Step 4 の値（1080p は変更前の `celv2enc` で取った値）と同じ。
- [ ] **Step 4: 速さを計る**: GPU 版は 720p 2タイルの `speed:` 行の fps を5回（変更前の値: 約 106.8 fps、1枚約 9.1 ms）。CPU 版は 720p と 1080p の2タイルで3回ずつ、1枚あたりの時間（この PC、48 コア）。`CUDA_VISIBLE_DEVICES=-1` で `--device auto` が CPU で最後まで符号化することも確かめる。
- [ ] **Step 5: README とレポートを書く**: README に「NVIDIA Turing 以降の GPU（ドライバ r580 以降）があれば GPU、なければ CPU で判定する。`--device` で指定できる」を足す。レポートに、ビルド時間と実行ファイルの大きさ、MD5 の表、GPU 版の速さが変わっていないこと、CPU 版の速さ、RTX 5090 以外では確かめていないことを書く。
- [ ] **Step 6: コミットする**: `build: GPU code for Turing through Blackwell; CPU/GPU engine report`
