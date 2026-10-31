# 場面ごとの並列符号化 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** キーフレームから次のキーフレームまでの区間を複数の `Pipeline` で同時に符号化し、サイズ指定の精度を 1 本のときと同じかそれ以上に保つ。

**Architecture:** main.cpp のフレームのループを `ChunkCoder`（区間 1 つを符号化する）に移し、読み込み
スレッドが区間を作り、`--workers` 本の `ChunkCoder` が並べて符号化し、書き出しは区間の順に行う。
サイズ指定は `SizeControl` を区間どうしで共有し、決定性のため区間 k は区間 k − workers までの結果だけを見る。

**Tech Stack:** C++20（celv2enc、CMake + Ninja、MSVC）、C#（.NET Framework 4.8）。

**Spec:** `docs/superpowers/specs/2026-10-31-chunk-parallel-design.md`（サイズ指定の仕様 `2026-10-31-target-size-design.md` と計画 `2026-10-31-target-size.md` の上に作る）

## Global Constraints

- `--workers 1`（既定）は今と同じバイト列。サイズ指定なしでは `--workers` を変えても同じバイト列。
- 同じ入力・設定・`--workers` なら毎回同じバイト列（スレッドの速さで変わらない）。
- 区間はキーフレームで区切るだけ。キーフレームを足さない。48 フレーム未満の区間は次とつなげる。
- メモリ: celv2enc のプロセスのメモリは搭載メモリの 80% まで（`--memory-limit` の既定 `auto`）。ワーカーはこれに収まる数だけ、先読み（`--chunk-memory` の既定）は残りの分。超えたら読み込みを待つ。
- サイズ指定: 超えない。補正は ±2 / 区間の中のミニ GOP ごと（今どおり）、qindex 1〜255、s（qindex いくつで半分）は 12〜48、初期値 24。
- パイプ入力（`-i -`）でも動くこと。
- `src/` に AVM のエンコーダのヘッダを入れない。ビルドは PowerShell のツールから（Bash からのビルドはヘッダの依存が 0 になったことがある。`ninja -t deps | Select-String "#deps 0"` が空であること）。git の作者は既存の設定のまま。push しない。
- 新しい GPU のコードを書く場合は Vulkan（今回は予定なし）。

## Review Focus

1. 1 場面だけの動画、またはすべての区間が 48 フレーム未満: 区間は 1 つ（または少数）になり、正しく符号化でき、`--workers 1` と同じバイト列（Task 3）。
2. `--chunk-memory` より長い場面: 読み込みを待ちながら、その区間を 1 本で符号化し終える（固まらない）（Task 3）。
3. 区間の途中で例外（GPU のメモリ不足など）: 他のワーカーを止め、書きかけの IVF を残さずエラーで終わる。GOP の試しの失敗は今どおり試しだけを切る（Task 3）。
4. `--frames N` が区間の途中で終わる: 最後の区間はそこで終わり、サイズ指定の統計のフレーム数と合う（Task 3 と Task 5）。
5. サイズ指定で最後の区間をやり直しても超える: 今どおり終了コード 3 と「over --target-size」の行（GUI のやり直しはそのまま動く）（Task 5）。

---

### Task 1: キーフレームから始めた新しい `Pipeline` が同じバイト列を出す

**Files:** Modify: `src/pipeline.h/.cpp`、必要なら `src/state/codec_state.*`、`src/core/encoder.*`。Test: `tests/test_pipeline_sync.cpp`

**Interfaces:**
- Produces: `void Pipeline::start_at(int display)` — 次に符号化するキーフレームが表示順 `display` から始まる区間の頭であることを伝える（1 本で通したときにそのキーフレームの時点で持っている値を、カウンタに入れる）。区間の頭でなければ呼ばない。

- [ ] **Step 1: 失敗するテストを書く** `pipeline_fresh_at_keyframe`: 1 本の `Pipeline` で「キーフレーム、ミニ GOP 2 つ、キーフレーム（display 9）、ミニ GOP 1 つ」を符号化する。別の新しい `Pipeline` に `start_at(9)` してから後半（display 9 からのキーフレームとミニ GOP）を符号化し、後半の OBU がフレームごとに一致すること（CPU と Auto の両方）。
- [ ] **Step 2:** ビルドして実行。Expected: コンパイルエラー（`start_at` がない）。
- [ ] **Step 3:** `start_at` を作る。キーフレームで初期化されないカウンタ（`frame_number_`、`display_counter_` など、テストが一致しない原因になっているもの）を洗い出し、1 本で通したときの値にそろえる。洗い出した状態とその扱いを ledger に書く。
- [ ] **Step 4:** `ctest -j 8`。Expected: 全部 PASS。
- [ ] **Step 5:** コミット `feat: Pipeline::start_at, a fresh pipeline codes a chunk from its keyframe as one pipeline would`。

### Task 2: フレームのループを `ChunkCoder` に移す（バイト列は同じ）

**Files:** Create: `src/chunk_coder.h`, `src/chunk_coder.cpp`（`celv2_gpu` に追加）。Modify: `src/main.cpp`、`CMakeLists.txt`。

**Interfaces:**
- Produces:
  ```cpp
  struct ChunkFrame { Frame420 f; int display; bool key; };
  struct CodedFrame { std::vector<uint8_t> tu; int display; Frame420 recon; Frame420 source; };  // recon / source: --recon / --psnr のときだけ
  struct LoopConfig { /* main のループが読む Args の値すべてと bframes、max_coded_q など */ };
  class ChunkCoder {
   public:
    ChunkCoder(const SeqConfig& cfg, Device device, const LoopConfig& lc);
    // 区間 1 つ（frames[0] はキーフレーム、または動画の先頭）。区間の頭なら start_at を呼んでから符号化する。
    std::vector<CodedFrame> code(const std::vector<ChunkFrame>& frames, const std::function<void(std::vector<CodedFrame>&)>& after_group = {});
    // 基準 qindex のずれ（本体と GOP の試しの Pipeline の両方に。サイズ指定なしでは 0 のまま）。after_group の中から呼んでよい。
    void set_qindex_offset(int d);
  };
  ```
  `after_group` はキーフレーム 1 枚、1 枚のフレーム、ミニ GOP の符号化ごとに、その出力を渡して呼ばれる（サイズ指定の補正が今と同じ点で動く）。
- 場面の判定（`classify`）、`ReadAhead`、IVF・`--recon`・`--psnr`・`--progress` の書き出し、`--pass` の統計は main に残す。

- [ ] **Step 1: 確かめ方を用意する:** 移す前の `build/celv2enc.exe` を `build/celv2enc_before.exe` に写し、6 通り（既定、`--tune animation`、`--tune film`、`--preset max`、`--preset max --tune animation`、`--keyint 30 --bframes 3 --deltaq 1 --tplq 2`）で、場面転換を含む 256x144 60 フレームを符号化した IVF を保存する（スクリプトは scratchpad）。
- [ ] **Step 2:** 移す。`--workers` はまだ足さない（main は `ChunkCoder` 1 つに、読んだ順に区間を渡す）。
- [ ] **Step 3:** 6 通りを新しい exe で符号化し、IVF がバイト単位で一致すること。`ctest -j 8` が全部 PASS。
- [ ] **Step 4:** コミット `refactor: the frame loop in ChunkCoder (the same bytes)`。

### Task 3: `--workers` と `--chunk-memory`

**Files:** Modify: `src/main.cpp`（読み込みスレッド、区間の列、ワーカー、順番どおりの書き出し）。Test: `tests/test_cli.cpp`

**Interfaces:**
- Consumes: Task 1 の `start_at`、Task 2 の `ChunkCoder`。
- Produces: `--workers N`（1〜16、既定 1）、`--chunk-memory MB`（64 以上、既定 4096）。区間は「キーフレームから次のキーフレームの手前まで、48 フレーム未満なら次とつなげる」。

- [ ] **Step 1: 失敗するテストを書く** `cli_workers`: 128x64、場面転換 3 つ（各場面 50 フレーム前後、合計 200 フレーム）の動画で:
  - `--workers 1` / `2` / `4` の IVF が同じ（既定と `--preset max --tune animation` の 2 通り）、`--recon` が復号と一致。
  - 場面 1 つだけの 60 フレームで `--workers 4` と `1` が同じ（Review Focus 1）。
  - `--chunk-memory 64` で 1 場面 200 フレーム（64 MB を超える量）を `--workers 2`: 終わり、`--workers 1` と同じ（Review Focus 2）。
  - `--frames 120` で `--workers 2` と `1` が同じ（Review Focus 4）。
  - `--workers 0`、`--workers 17`、`--chunk-memory 10` が理由つきで拒否される。
- [ ] **Step 2:** ビルドして `test_cli`。Expected: FAIL（`--workers` を知らない）。
- [ ] **Step 3:** 実装する。読み込みスレッドが判定と区間づくりを行い、区間の列（上限は `--chunk-memory` 分のフレーム）に積む。ワーカーは区間を順に取り、出力は区間の番号順に書く。どこかの区間で例外が出たら、全ワーカーを止めて最初の例外で終わる（Review Focus 3）。
- [ ] **Step 4:** `ctest -j 8`。Expected: 全部 PASS。
- [ ] **Step 5:** コミット `feat: --workers and --chunk-memory, scene chunks coded at once (the same bytes)`。

### Task 4: `SizeControl` を区間で共有できる形にする

**Files:** Modify: `src/core/size_control.h/.cpp`。Test: `tests/test_size_control.cpp`

**Interfaces:**
- Produces（今の API はそのまま残し、`workers` 1 のときは今と同じ値を返す）:
  ```cpp
  // 区間 k を始めてよいのは、区間 k - workers までがすべて終わってから（呼ぶ側が待つ）。
  int SizeControl::chunk_start_qindex(size_t chunk, size_t first_frame, size_t frames);  // 区間 k の初めの基準 qindex
  void SizeControl::coded(size_t chunk, size_t frame, uint64_t payload, int qindex);       // どの順でもよい
  int SizeControl::chunk_update(size_t chunk);       // 区間の中の補正（±2）。見るのは「区間 chunk - workers まで」と区間 chunk 自身
  void SizeControl::chunk_done(size_t chunk);
  double SizeControl::slope() const;                  // s（12〜48、データが足りないと 24）
  ```
  コンストラクタに `int workers`（既定 1）を足す。最初の `workers` 個の区間の初めの qindex は、基準に 0、+1、−1、+2、−2 … を足したもの（方法 4）。
- s の当てはめ: 見えるフレームごとに y = log2(実際 ÷ (下調べ × 比 1)) と x = q − 下調べの q を、バイト数で重みづけした最小二乗で y = a − x / s に当てはめる。違う q が 2 つ未満、または q ごと 8 フレーム未満なら 24。

- [ ] **Step 1: 失敗するテストを書く**:
  - `slope_fit`: 下調べ 10,000 バイトのフレームに、q 140 と 150 の実際を 2^(−(q−120)/30) 倍で入れると `slope()` が 30 ± 0.5。q が 1 つだけなら 24。y が s = 5 相当なら 12、s = 200 相当なら 48。
  - `chunks_see_only_finished`: workers 2 で、区間 0 と 1 を始め、区間 1 の `coded` を見積もりの 2 倍で入れても区間 0 の `chunk_update` は変わらない。区間 2 の `chunk_start_qindex` は区間 0 の分だけを見る。
  - `dither_first_chunks`: workers 3 で、区間 0 / 1 / 2 の初めの qindex が基準 / +1 / −1。
  - `workers_1_unchanged`: workers 1 で今の `update_moves_at_most_2` と同じ qindex の並び。
- [ ] **Step 2:** ビルド。Expected: コンパイルエラー。
- [ ] **Step 3:** 実装する。
- [ ] **Step 4:** `ctest -j 8`。Expected: 全部 PASS。
- [ ] **Step 5:** コミット `feat: SizeControl shared by chunks (deterministic), the qindex slope fitted to the video`。

### Task 5: サイズ指定と区間をつなぐ（方法 1〜3）

**Files:** Modify: `src/main.cpp`。Test: `tests/test_cli.cpp`

**Interfaces:**
- Consumes: Task 3 の区間の列とワーカー、Task 4 の区間の API、Task 2 の `after_group`。
- 決めごと: 区間 k は区間 k − workers が終わるまで始めない。最後の区間（入力の終わりで決まる）は他の区間がすべて終わってから始める。ファイル全体が予算を超えたら、最後の区間だけを基準 qindex +2 ずつで符号化し直す（最大 3 回）。狙いは 98% のまま（Task 7 で決め直す）。

- [ ] **Step 1: 失敗するテストを書く** `cli_target_size_workers`（Task 3 の 200 フレームの動画、256x144 にして）:
  - `--pass 1` → `--pass 2 --workers 2 --target-size B`（B = 1 回目の 90%）が 0 を返し、IVF ≤ B、復号が `--recon` と一致。
  - 同じ 2 回目をもう 1 度実行して同じバイト列（決定性）。
  - 予算を「1 回目の `--workers 2` の出力 − 1 バイト」にすると、最後の区間をやり直して 0 を返し、IVF ≤ 予算（方法 3）。stderr に `re-coded the last chunk`。
  - B = 200 で終了コード 3 と `over --target-size`（Review Focus 5）。
  - `--frames 150` で 1 回目と 2 回目（`--workers 2`）が通る（Review Focus 4）。
- [ ] **Step 2:** `test_cli`。Expected: FAIL。
- [ ] **Step 3:** 実装する（最後の区間のフレームは書き終わるまで持つ）。
- [ ] **Step 4:** `ctest -j 8`。Expected: 全部 PASS。
- [ ] **Step 5:** コミット `feat: target size across chunks: shared control, the last chunk last, and re-coded when over`。

### Task 6: `--workers auto` と GUI

**Files:** Modify: `src/main.cpp`、`gui/CelV2.Settings/EncodeSettings.cs`、`gui/CelV2.Settings/CommandBuilder.cs`、`gui/CelV2.Ui/AdvancedDialog.cs`、`gui/README_ja.txt`。Test: `tests/test_cli.cpp`、`gui/tests/CommandTests.cs`、`gui/tests/UiTests.cs`

**Interfaces:**
- Produces: `--workers auto`。`EncodeSettings.Workers`（`int`、0 = 自動、既定 0）。詳細設定「速度・処理」に「場面ごとの並列」（自動 / 1 / 2 / 4、名前 `workers`）。`EncodeArgs` は 0 なら `--workers auto`、ほかは数。

- [ ] **Step 1: 計測（空いた機械で）:** 1 本の変換の CPU 使用率と GPU 使用率（FHD、4K、既定と `--preset max`）。続けて `--workers` 1 / 2 / 3 / 4 の速さを、場面の多い FHD 600 フレーム（AM、Angel Beats の混在区間）と 4K 120 フレームで。結果から `auto` の規則（解像度ごとの数）を決めて ledger に書く。
- [ ] **Step 2: 失敗するテストを書く**: `--workers auto` が受け付けられ、`--workers 1` と同じバイト列（サイズ指定なし）。`CELV2_TEST_PIPELINE_LIMIT=1` のとき `--workers 4` が 1 本に減り、stderr に `--workers` の警告、出力は同じ。GUI: `Workers` 0 → `--workers auto`、2 → `--workers 2`、詳細設定の往復、古いプリセットで 0。
- [ ] **Step 3:** 実装する（`Pipeline` を作れなければ、作れた数で続けて警告を 1 行。`CELV2_TEST_PIPELINE_LIMIT` はテスト用で、`CELV2_TEST_TRIAL_FAIL` と同じ書き方）。README の「速度・処理」に 1 行（場面の少ない動画は速くならないことも）。
- [ ] **Step 4:** `ctest -j 8`（GUI のテストも含む）。Expected: 全部 PASS。
- [ ] **Step 5:** コミット `feat: --workers auto, and the GUI's 場面ごとの並列`。

### Task 6b: メモリの上限（搭載メモリの 8 割）

**Files:** Create: `src/io/memory.h`, `src/io/memory.cpp`（`celv2_core`）。Modify: `src/main.cpp`。Test: `tests/test_cli.cpp`、`tests/test_io.cpp`

**Interfaces:**
- Produces: `uint64_t installed_memory_bytes()`（`CELV2_TEST_TOTAL_RAM_MB` があればその値。テスト用、`CELV2_TEST_TRIAL_FAIL` と同じ書き方）、`uint64_t process_private_bytes()`。`--memory-limit MB|auto`。

- [ ] **Step 1: 計測:** 1 本の変換のピークのメモリ（FHD / 4K、既定 / `--preset max`）と、ワーカーを 1 本足したときの増分。ledger に書く。
- [ ] **Step 2: 失敗するテストを書く**: `installed_memory_bytes()` が 0 でない、`CELV2_TEST_TOTAL_RAM_MB=1024` で 1 GiB。`CELV2_TEST_TOTAL_RAM_MB` を小さくした（ワーカー 2 本分に届かない値）状態で `--workers 4` が 1 本に減り、stderr に `--memory-limit` の警告、出力は `--workers 1` と同じ。`--memory-limit 0` と `--memory-limit x` が理由つきで拒否される。
- [ ] **Step 3:** 実装する。上限に収める順番（ピークのメモリは B フレームの並列処理の `Encoder` の写しが大半: 2026-10-31、AM 120 フレーム、
  FHD 4.6 → 1.9 GiB、FHD max 8.3 → 3.0、4K 13.8 → 5.0、4K max 26.0 → 9.2 GiB を `--parallel-b 0` で。出力は同じ）:
  1. ワーカーを減らす。2. `--parallel-b` を自動で切る（同じバイト列、警告 1 行）。3. GOP の試しの `Pipeline` を作らない（出力が変わる、警告 1 行）。
  4. それでも超えるなら警告を 1 行出して続ける。`--parallel-b` や `--gop-trial` を明示したときは、そのまま（警告だけ）。
  テストに追加: `CELV2_TEST_TOTAL_RAM_MB` を小さくすると、`--parallel-b` 自動のとき stderr に `--parallel-b` の警告が出て、出力は `--parallel-b 0` と同じ。
- [ ] **Step 4:** `ctest -j 8`。Expected: 全部 PASS。
- [ ] **Step 5:** コミット `feat: memory at most 80% of the installed RAM (--memory-limit)`。

### Task 7: 計測と記録

**Files:** Create: `docs/design/2026-10-31-chunk-parallel.md`。必要なら `docs/design/tried-not-kept.md`。

- [ ] **Step 1:** 速さ: Task 6 の計測の表（`auto` で 1 本の何倍か）。
- [ ] **Step 2:** 精度: サイズ指定の 12 本 × 3 段階（サイズ指定の記録と同じ素材・同じ指定）を `--workers auto` で。指定比 %、PSNR-Y の差、時間、やり直しの回数。方法 3 があるので、狙い 98% / 99% / 99.5% を比べ、超えずに最も近いものを `kAim` にする。
- [ ] **Step 3:** 下調べに `--rd-pass 0` を足した場合（2 倍速）を同じ表で測り、精度の目標（超えない、98〜99.5%、PSNR 差 0.1 dB 以内）を満たせば `--pass 1` の既定にする。
- [ ] **Step 4:** 記録を書いてコミット（採らなかったものは tried-not-kept にも行を足す）`docs: chunk-parallel results`。
