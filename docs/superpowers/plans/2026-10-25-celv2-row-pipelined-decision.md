# 行ごとに進む GPU 判定と CPU 符号化の重ね合わせ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** GPU がスーパーブロックの行ごとに判定を仕上げ、CPU のタイルスレッドが届いた行から符号化することで、1フレームの処理時間を縮める（出力はバイト単位で今と同じ）。

**Architecture:** `decision_kernel` を `cost_kernel`（1スーパーブロックを K ブロックで分担）と `finalize_kernel`（今の後半）に分け、行ごとに積んで CUDA イベントで完了を知らせる。ホストの回収スレッドが行を `DecisionMap` に変換して `RowGate` で公開し、`Encoder` は各スーパーブロックの前にその行を待つ。

**Tech Stack:** C++17、CUDA 13.3（sm_120）、MSVC + Ninja、AVM v1.0.0、CTest。

**Spec:** `docs/superpowers/specs/2026-10-25-celv2-row-pipelined-decision-design.md`

## Global Constraints

- デコーダとのビット一致を保つ（各テストの MD5 / `first_diff` 比較）。
- 判定のロジックは変えない: 最小値は「最初の厳密な最小値」、モードは DC, V, H, SMOOTH, インターの順、分割は NONE, HORZ, VERT, SPLIT の順、インター候補はセルのラスタ順。
- bench.y4m 720p と 1080p（30枚、q150、1タイルと2タイル）で、今の main とバイト単位で同じストリームになる。
- K = ceil(2 × SM 数 / 1行のスーパーブロック数) を 1〜16 に収めた値。SM 数は `cudaGetDeviceProperties` から取る。
- `RowGate` と `DecisionMap` は CUDA にも AVM にも依存しない（`src/gpu/decision_map.h`）。
- av2/encoder を使わない（CTest `no_encoder_includes`）。コメントは英語。
- コミットはリポジトリの設定の作者で行う（`-c user.name/user.email` を付けない）。末尾に `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`。
- ビルド: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build.ps1`、テスト: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\test.ps1 [-R name]`。

## Review Focus

1. 高さが 64 の倍数でないフレーム（最後の行が半端）: 行数は切り上げで、最後の行も公開されること → Task 4 の 202x122 のバイト一致。
2. スーパーブロックが1行しかないフレーム（32x32 など）: K の計算と1行だけの公開で止まらないこと → Task 4 の 32x32 のバイト一致。
3. 縦に分かれたタイル（2x2）: 下のタイルのスレッドが上の行より先に待ち始めても正しく進むこと → Task 4 の 2x2 タイル。
4. 判定の途中でエラーになったとき: 待っているタイルスレッドが止まったままにならず、エラーが返ること → Task 1 の `fail` テストと Task 4 の `encode_fails_when_gate_fails`。
5. 途中に KEY フレームが入る列（keyint）: KEY と P で同じバッファとイベントを使い回しても壊れないこと → Task 4 の keyint 付きのバイト一致。

---

### Task 1: RowGate

**Files:**
- Modify: `src/gpu/decision_map.h`（`RowGate` を追加）
- Modify: `src/gpu/decision_map.cpp`
- Create: `tests/test_row_gate.cpp`
- Modify: `CMakeLists.txt`（`celv2_test(test_row_gate)`）

**Interfaces:**
- Produces:
  ```cpp
  class RowGate {
   public:
    void publish(int row);                  // rows 0..row are ready (monotonic; smaller values ignored)
    void wait_row(int row) const;           // blocks until row is ready; rethrows the failure if any
    void fail(std::exception_ptr e);        // records e and wakes every waiter
    int ready_rows() const;                 // number of rows published so far
  };
  ```
  `mutable std::mutex`、`mutable std::condition_variable`、`int ready_ = 0`、`std::exception_ptr error_`。コピー不可。

- [ ] **Step 1: 失敗するテストを書く**（`tests/test_row_gate.cpp`、`RUN()` で3つ）
  - `publish_releases_waiters`: 別スレッドが `wait_row(2)` で待ち、本スレッドが `publish(0)`, `publish(1)` の後もまだ待っていること（`std::atomic<bool> done` が false）、`publish(2)` の後に 1 秒以内に `done == true`。`ready_rows() == 3`。
  - `publish_out_of_order_is_monotonic`: `publish(3)` の後に `publish(1)` を呼んでも `ready_rows() == 4`。`wait_row(3)` はすぐ戻る。
  - `fail_wakes_and_rethrows`: 2 本のスレッドが `wait_row(5)` で待つ。`fail(std::make_exception_ptr(std::runtime_error("gpu")))` の後、両方が `std::runtime_error`（what が `"gpu"`）を受け取って 1 秒以内に終わる。その後の `wait_row(0)` も投げる。
- [ ] **Step 2: 実行して失敗を確認する**
  Run: `scripts\build.ps1` → Expected: `RowGate` が未定義でビルドエラー。
- [ ] **Step 3: `RowGate` を実装する**（`decision_map.h` に宣言、`decision_map.cpp` に定義）
- [ ] **Step 4: テストを実行する**
  Run: `scripts\test.ps1 -R test_row_gate` → Expected: `100% tests passed`。全体でも 15/15。
- [ ] **Step 5: コミットする**: `feat: RowGate for row-by-row decision hand-off`

### Task 2: 判定カーネルを cost_kernel と finalize_kernel に分ける

**Files:**
- Modify: `src/gpu/kernels.cu`（`decision_kernel` を2つに分割）
- Modify: `src/gpu/kernels.h`（起動関数）
- Modify: `src/gpu/engine.cu`（`run_decision` を新しい起動関数で呼ぶ。まだフレーム全体を1回で）

**Interfaces:**
- Produces（`kernels.h`、`namespace CelV2`）:
  ```cpp
  constexpr int kCostsPerSb = 1700 + 1024;  // per superblock: intra (425 x 4 modes) + inter slots
  // Costs of every candidate of superblock rows [row0, row0 + nrows), K blocks per superblock.
  void launch_costs(DevPlane src, const DevPlane* ref, const int* mvx, const int* mvy, int mv_cols,
                    int mv_rows, int sb_cols, int row0, int nrows, int k, int64_t lambda_q8,
                    int64_t* costs, cudaStream_t stream);
  // Mode / partition choice and SbOut emission for the same rows, from those costs.
  void launch_finalize(const int* mvx, const int* mvy, int mv_cols, int mv_rows, int sb_cols,
                       int row0, int nrows, bool has_ref, int64_t lambda_q8, const int64_t* costs,
                       SbOut* out, cudaStream_t stream);
  int blocks_per_sb(int sb_cols);  // K = clamp(ceil(2 * SMs / sb_cols), 1, 16)
  ```
  `costs` はスーパーブロック s（ラスタ順）につき `costs + s * kCostsPerSb`。先頭 1700 個はイントラ（`b * kNumModes + m`）、残りはインタータスク番号順。
- Consumes: 既存の `sub_satd`、`block_geom`、`cell_mv_index`、`dc_value`、`mode_bits`、`kInterBits`、`kPartBits`。

- [ ] **Step 1: 既存の比較テストで基準を確かめる**
  Run: `scripts\test.ps1 -R "test_gpu_intra|test_gpu_me|test_random_decisions|test_ippp"` → Expected: すべて pass（変更前の基準）。あわせて、変更前の `build\celv2enc.exe` を `%TEMP%\celv2_ref\celv2enc.exe` にコピーし、`bench.y4m` 30枚 q150 の出力（1タイル、`--tile-columns 1`）の MD5 を控える（Task 5 で使う）。
- [ ] **Step 2: タスク表を作る**: インタータスクの一覧（ブロック番号 b、セル番号 k、896 個）はスーパーブロックの位置に依存しない。`blocks_per_sb` と同じ場所で初回に一度だけ作り（関数ローカルの static。スレッド安全）、GPU のメモリに置く。
- [ ] **Step 3: `cost_kernel` を書く**: グリッド `(sb_cols, nrows * k)`、256 スレッド。スーパーブロック = `(blockIdx.x, row0 + blockIdx.y / k)`、分担番号 = `blockIdx.y % k`。タスク t（イントラ 1700 個 + インター 896 個、KEY ではイントラのみ）は `t % (k * 8) == 分担番号 * 8 + ワープ番号` のワープが受け持つ。SATD の和と `lam * bits` の足し方は今の `decision_kernel` と同じ。DC 予測値 425 個は各ブロックが共有メモリに計算する。
- [ ] **Step 4: `finalize_kernel` を書く**: グリッド `(sb_cols, nrows)`。今の `decision_kernel` の「インター候補の選択」以降を、共有メモリのコストの代わりに `costs` から読む形でそのまま移す。
- [ ] **Step 5: `run_decision` を差し替える**: `launch_costs(..., row0 = 0, nrows = sb_rows, k = blocks_per_sb(sb_cols), ...)` → `launch_finalize(...)` の順に呼ぶ。`costs` のバッファは `Impl` に持ち、`sb_cols * sb_rows * kCostsPerSb` 個を確保する。`decision_kernel` を削除する。
- [ ] **Step 6: テストを実行する**
  Run: `scripts\test.ps1` → Expected: 15/15 pass（`test_gpu_intra` と `test_random_decisions` が、GPU と CPU 参照実装の判定の一致を確かめる）。Step 1 の構成で `celv2enc` の出力 MD5 が控えと同じ。
- [ ] **Step 7: コミットする**: `perf: split the decision kernel into per-candidate costs and finalize`

### Task 3: 行ごとの起動と launch / collect

**Files:**
- Modify: `src/gpu/engine.h`、`src/gpu/engine.cu`
- Modify: `tests/test_gpu_intra.cpp`（テストを追加）

**Interfaces:**
- Consumes: Task 1 の `RowGate`、Task 2 の `launch_costs` / `launch_finalize` / `blocks_per_sb`。
- Produces（`GpuEngine` の public）:
  ```cpp
  // Queues uploads, motion search (ref != nullptr) and every superblock row's decision, then
  // returns; dm is sized for all superblocks. src and *ref may change once this returns.
  void launch(const Frame420& src, const Frame420* ref, int qindex, DecisionMap& dm);
  // Waits for each row in order, fills dm's superblocks of that row, publishes it on gate.
  // On a CUDA error calls gate.fail() and rethrows.
  void collect(DecisionMap& dm, RowGate& gate);
  ```
  `decide_intra(src, q)` は `launch(src, nullptr, q, dm)` + `collect(dm, local_gate)`、`decide_inter(src, prev, q)` は `launch(src, &prev, q, dm)` + `collect(...)`。

- [ ] **Step 1: 失敗するテストを書く**（`tests/test_gpu_intra.cpp` に追加）
  - `launch_collect_matches_decide`: 202x122 と 640x360 で、`decide_inter(src, prev, 120)` の結果と、別の `GpuEngine` で `launch(src, &prev, 120, dm)` + `collect(dm, gate)` した結果が、全スーパーブロックで `tree` と `blocks`（w4, h4, inter, intra, mv_row, mv_col）が等しいこと。`gate.ready_rows() == dm.sb_rows`。KEY（`ref == nullptr`）でも同じ比較。
  - `rows_publish_before_collect_ends`: 640x360 で `collect` を別スレッドで走らせ、本スレッドが `gate.wait_row(0)` から戻った時点で `gate.ready_rows() >= 1`、最後に `ready_rows() == sb_rows`（行が順に公開されること）。
- [ ] **Step 2: 実行して失敗を確認する** → Expected: `launch` / `collect` が未定義でビルドエラー。
- [ ] **Step 3: 実装する**: 
  - `Impl` に `cudaStream_t stream`（コンストラクタで作成、デストラクタで破棄）、行ごとの `cudaEvent_t`（`cudaEventDisableTiming`、`sb_rows` 個）、固定メモリの `host_out` を持つ。既存の `cudaMemcpyAsync` と同期もこのストリームに移す。
  - `launch`: アップロードと動き探索（今と同じ）の後、各行 r について `launch_costs(row0 = r, nrows = 1)` → `launch_finalize(row0 = r, nrows = 1)` → その行の `SbOut` を `host_out` へ `cudaMemcpyAsync` → `cudaEventRecord(ev[r])`。`dm` の `sb_size / sb_cols / sb_rows / sbs` をここで整える。
  - `collect`: r = 0.. で `cudaEventSynchronize(ev[r])`、行 r の `SbOut` を今の変換ループで `dm.sbs` に入れ、`gate.publish(r)`。CUDA エラーは `gate.fail(std::current_exception())` してから再送出。
- [ ] **Step 4: テストを実行する** → Run: `scripts\test.ps1` → Expected: 15/15 pass。Task 2 Step 1 の構成で `celv2enc` の出力 MD5 が控えと同じ。
- [ ] **Step 5: コミットする**: `feat: GPU decision launched and collected row by row`

### Task 4: エンコーダの行待ちと Pipeline の回収スレッド

**Files:**
- Modify: `src/core/encoder.h`、`src/core/encoder.cpp`（`encode` に `gate` 引数、`TileCoder` で行待ち）
- Modify: `src/pipeline.cpp`
- Modify: `tests/test_ippp.cpp`（テストを追加）
- Modify: `tests/test_cli.cpp`（決定性のテストを追加）

**Interfaces:**
- Consumes: Task 1 の `RowGate`、Task 3 の `launch` / `collect`。
- Produces: `std::vector<uint8_t> Encoder::encode(const Frame420& src, const DecisionMap& dm, bool key, const RowGate* gate = nullptr);`（`nullptr` = 全行準備済み）。`Pipeline::encode` のシグネチャは変えない。

- [ ] **Step 1: 失敗するテストを書く**
  - `tests/test_ippp.cpp` の `pipeline_matches_sync`: 同じ入力（`scene_frame`、dx=3, dy=2）を、(a) `Pipeline`、(b) `GpuEngine::decide_intra / decide_inter`（参照は `Encoder::last_recon()`）+ `Encoder::encode(src, dm, key)` で符号化し、全フレームの TU が等しいこと。構成: 640x360（1 / 2x1 / 2x2 タイル、8枚）、202x122（2x1 タイル、6枚）、32x32（1タイル、4枚）、640x360 で keyint 3（2x1 タイル、7枚）。各構成でデコード一致（`check_decode`）も確かめる。
  - `tests/test_ippp.cpp` の `encode_fails_when_gate_fails`: 640x360、2x1 タイルの `Encoder` に、全行分の大きさの `DecisionMap`（`uniform_decision`）と、`fail(std::make_exception_ptr(std::runtime_error("gpu")))` 済みの `RowGate` を渡すと、`encode` が `std::runtime_error` を投げて 10 秒以内に戻ること（止まったままにならない）。
  - `tests/test_cli.cpp` の `cli_tiles_deterministic`: 64x64 の y4m（`scene_frame`、3枚）を `--tile-columns 1 --tile-rows 1` で 5 回、それぞれ新しいプロセスで符号化し、5 つの IVF のバイト列がすべて同じこと。
- [ ] **Step 2: 実行して失敗を確認する** → Expected: `encode` の4引数版がなくビルドエラー。
- [ ] **Step 3: `Encoder` を実装する**: `encode` に `gate` を足し、`TileCoder` に `const RowGate* gate_` を渡す。`code_tile` で `sb_ = &dm_.sbs[...]` の直前に `if (gate_) gate_->wait_row(mi_row >> cm_->mib_size_log2);`。タイルスレッドが投げた例外は今の `TileOut::error` の経路で返る。
- [ ] **Step 4: `Pipeline::encode` を実装する**: `DecisionMap dm; RowGate gate;` → `gpu_.launch(src, key ? nullptr : &enc_.last_recon(), cfg_.qindex, dm)` → `auto collector = std::async(std::launch::async, [&] { gpu_.collect(dm, gate); });` → `enc_.encode(src, dm, key, &gate)` を try で呼び、例外のときも `collector.wait()` してから再送出 → 成功時は `collector.get()`（GPU のエラーをここで返す）。
- [ ] **Step 5: テストを実行する** → Run: `scripts\test.ps1` → Expected: 15/15 pass（追加したケースを含む）。
- [ ] **Step 6: コミットする**: `perf: tile threads code each superblock row as soon as the GPU publishes it`

### Task 5: 計測と報告

**Files:**
- Create: `docs/reports/2026-10-25-row-pipelined-decision.md`

**Interfaces:**
- Consumes: Task 2 Step 1 で控えた変更前のバイナリと MD5。

- [ ] **Step 1: 出力が今の main と同じか確かめる**: bench.y4m 720p と `src1080.mp4` の先頭30枚（y4m）を、30枚、q150、`--tile-columns 0` と `1` で、変更前のバイナリと新しいバイナリの両方で符号化し、IVF の MD5（`Get-FileHash`）が 4 組とも同じこと。nsys は日本語のパスを扱えないので、入力は ASCII のパス（`%TEMP%\celv2_bench\`）に置く。
- [ ] **Step 2: 速さを計る**: 720p、2タイル、30枚を 3 回ずつ、変更前後で `speed:` 行の fps を記録する。`nsys profile -t cuda` で1回分を記録し、sqlite から「CPU のタイル符号化（タイマー）と GPU の行ごとのカーネル」の重なりと、定常状態の P フレーム1枚あたりの時間を出す。CPU 側の時間は一時的なタイマー（計測後に `git checkout` で戻す）で取る。
- [ ] **Step 3: レポートを書く**: 変更前後の1枚あたりの時間（定常状態、目標 12 ms 以下）、fps、GPU と CPU の重なり、残っている待ちの場所と時間、行間の隙間（`finalize_kernel` の時間）を数字で書く。隙間が大きければ、範囲外とした「分割決定の2本目のストリーム」を次の候補として挙げる。
- [ ] **Step 4: コミットする**: `docs: row-pipelined decision measurements`
