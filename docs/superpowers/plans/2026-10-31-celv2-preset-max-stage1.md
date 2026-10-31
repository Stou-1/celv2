# `--preset max` 第 1 段 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `--preset max` で、16 フレームのミニ GOP と 8 フレーム ×2 を実際に符号化して良い方を残し、デブロックの強さをフレームごとに選ぶ。

**Architecture:** 2 本目の `Pipeline`（試し用）を本体から状態ごとコピーし（`Pipeline::sync_from`）、16 フレーム案をそちらで、8+8 案を本体で符号化する。J = D + λ·R で 16 が勝てば 2 本を入れ替える。ミニ GOP の符号化を main.cpp の 1 つの関数にまとめ、出力を後回しにできるようにする。デブロックは `CodecState::end_frame` の中で候補ごとにかけて比べる。

**Tech Stack:** C++17（MSVC、ninja）、CUDA / Vulkan の判定エンジン（変更なし）、C# WinForms（GUI）。

**Spec:** docs/superpowers/specs/2026-10-31-celv2-preset-max-design.md

## Global Constraints

- `src/` に AVM のエンコーダのヘッダ（`av2/encoder/*`）を入れない。
- 出力は avmdec で復号でき、`--recon` と一致する。
- コマンドラインで個別に指定したオプションは `--preset max` より優先する（`given` の一覧）。
- `--preset max` の既定: `--bframes` 15（`--refs` 5 以上、`--qindex` 100 以上のとき）、`--gop-still` 0.5、GOP の試し符号化オン、デブロックの探索オン。
- λ: ビットあたり 0.025 × step²（`Encoder::code` が `end_frame` に渡している値と同じ）。D は表示される各フレームの Y + U + V の 2 乗誤差（符号化の深さの値で）。
- デブロックの候補: {-4, -2, 0, +2, +4}。探索するときは `df_par_bits_minus2` = 3。
- コミットの末尾: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`。プッシュしない。

## Review Focus

1. 動画の終わりで 9〜15 フレームのミニ GOP になる場合: 試しは「m フレーム 1 つ」と「8 + (m−8)」で比べる（Task 3 のテストの 20 フレームの動画が、最後の短いミニ GOP を含む）。
2. 試し用の Pipeline を作れない（GPU のメモリ不足など、16K）: 警告を 1 行出して、GOP の試しなしで続ける（Task 3）。
3. 参照の鍵が 2 本の Pipeline でぶつかる: 鍵はプロセス全体で一意にする（Task 1）。
4. `--recon` / `--psnr` / `--progress` の出力が、試しの有無で順番も内容も変わらない（Task 2 と Task 3 のテスト）。
5. `--preset max` と `--tune` を両方指定したとき: `--tune` の既定のうち `--preset max` と重なるもの（`--gop-still`）は `--preset max` が優先し、どちらも明示した値には負ける（Task 3）。

---

### Task 1: `Pipeline::sync_from` と一意な参照の鍵

**Files:**
- Modify: `src/pipeline.h`, `src/pipeline.cpp`, `src/core/encoder.h`, `src/core/encoder.cpp`
- Create: `tests/test_pipeline_sync.cpp`、`CMakeLists.txt` に `celv2_test(test_pipeline_sync)`

**Interfaces:**
- Produces: `void Pipeline::sync_from(const Pipeline& o)`: 次に o で符号化するのと同じ結果になるよう、o の Encoder の状態（`Encoder::sync_from`）、`recons_`、`last_shown_` を写す。
- Produces: `double Pipeline::sse_per_bit() const` と `double Encoder::sse_per_bit()`: `Encoder::code` が `end_frame` に渡している 0.025 × step²（その場の式を関数にして両方から使う）。
- 鍵: `next_key_` をやめ、pipeline.cpp の `static std::atomic<long long>` から取る。

- [ ] **Step 1: テストを書く** `pipeline_sync_codes_alike`: 64x48 の `scene_frame`、`Device::Cpu`、`refs` 7。A で KEY、隠しアンカー（4 フレーム目、reference）、B フレーム 3 つ、show_existing を符号化。B を A から sync し、両方で次のミニ GOP（同じ FrameSpec の並び）を符号化して、各フレームの OBU と `last_recon()` が一致すること。続けて B だけで別の FrameSpec（qindex_delta を変えたアンカー）を符号化した後、もう一度 A から sync して、次の 1 フレームが一致すること。
- [ ] **Step 2: 失敗を確かめる**（`sync_from` がなくてビルドエラー）
- [ ] **Step 3: 実装する。** `CodecState::sync_refs_from` で足りない状態があれば（テストが一致しなければ）そこに足す。
- [ ] **Step 4: テストと全テストを通す** `ctest -R "pipeline_sync|ippp|cli"`
- [ ] **Step 5: コミット**

### Task 2: ミニ GOP の符号化を 1 つの関数に（動作は同じ）

**Files:** Modify: `src/main.cpp`

**Interfaces:**
- Produces（main.cpp の中のラムダ）: `code_minigop(Pipeline& p, size_t start, size_t m) -> std::vector<Shown>`。`struct Shown { std::vector<uint8_t> obus; int display; Frame420 recon; const Frame420* src; }`（`obus` は隠しフレームの分も含む、時間区切りの前までの 1 単位）。q[start..start+m) を今と同じ順で符号化する。`last_ref` は呼び出し側が更新する。
- `pipe` を `std::unique_ptr<Pipeline>` に。
- 再構成は、試しがあるときは常に、ないときは今と同じく `--recon` / `--psnr` のときだけ持つ。

- [ ] **Step 1: 変更前の exe で基準を作る**: `a.y4m` 17 フレームと 1080p の AM（30 s から 60 フレーム）を、`--refs 7 --qindex 130`、`--tune animation`、`--parallel-b 0` / `1` で符号化して保存。
- [ ] **Step 2: 書き換える。**
- [ ] **Step 3: 同じバイト列であることを確かめる**（Step 1 の全部を cmp）。`ctest` 全部。
- [ ] **Step 4: コミット**

### Task 3: `--preset max` と GOP の試し符号化

**Files:** Modify: `src/main.cpp`、`tests/test_cli.cpp`

**Interfaces:**
- Consumes: Task 1 の `sync_from` / `sse_per_bit`、Task 2 の `code_minigop`。
- オプション: `--preset max`（他の値は "--preset must be max" で拒否）、`--gop-trial 0|1`（`--preset max` で既定 1）。
- 試し: m > 8 で `gop_trial` のとき、試し用 Pipeline T を（初回に作って）`T.sync_from(M)`、`r1 = code_minigop(T, 0, m)`、`r2 = code_minigop(M, 0, 8) + code_minigop(M, 8, m − 8)`。J が小さい方を出力し、r1 なら M と T を入れ替える。T を作れなければ "celv2enc: --gop-trial off: <理由>" を出して試しを止める。

- [ ] **Step 1: テストを書く** `cli_preset_max`: `cli_tunes` と同じ 20 フレームの動画（`scene_frame(64, 48, i / 8, 0)`）、`--refs 7 --qindex 120`。(a) `--preset max --recon` の復号が recon と一致。(b) `--preset max` の出力が、`--preset max --gop-trial 0`（16 を取る）か `--preset max --bframes 7`（8+8）のどちらかとバイト単位で一致。(c) `--preset fast` が "--preset must be" で拒否される（`cli_rejects_bad_args` に追加）。
- [ ] **Step 2: 失敗を確かめる**
- [ ] **Step 3: 実装する**（オプション、既定値、試し、出力）。usage のコメントに `--preset max` と `--gop-trial` を書く。
- [ ] **Step 4: テストを通す** `ctest`
- [ ] **Step 5: 1080p で確かめる**: AM 混在区間（0 s、120 フレーム）で、`--preset max --deblock-search 0` 相当（Task 4 の前なので素の状態）と既定のサイズ・PSNR・速さ。
- [ ] **Step 6: コミット**

### Task 4: フレームごとのデブロックの強さ

**Files:** Modify: `src/bitstream/seq_header.h`（`SeqConfig::deblock_search`）、`src/bitstream/seq_header.cpp`、`src/state/codec_state.cpp`（`end_frame`）、`src/main.cpp`、`tests/test_cli.cpp`

**Interfaces:**
- `--deblock-search 0|1`（`--preset max` で既定 1）。`cfg.deblock_search`。
- `end_frame`: デブロックの前の再構成を保存し、各候補で差分（輝度・色差の delta_q と delta_side を同じ値）を設定して `deblock()`、元の映像との Y + U + V の 2 乗誤差を測って戻す。最小のものを設定してから本番の `deblock()`。

- [ ] **Step 1: タイルの符号化がデブロックの差分を読んでいないことを確かめる**（`delta_q_luma` などを grep。読んでいれば計画を止めて見直す）。
- [ ] **Step 2: テストを書く** `cli_preset_max` に `--deblock-search 1 --recon` の復号一致を足す。
- [ ] **Step 3: 実装する**
- [ ] **Step 4: テストを通す** `ctest`
- [ ] **Step 5: コミット**

### Task 5: GUI のチェックボックス

**Files:** Modify: `gui/CelV2.Settings/EncodeSettings.cs`（`[DataMember] public bool MaxEffort`）、`gui/CelV2.Settings/CommandBuilder.cs`（`--preset max`）、`gui/CelV2Gui/MainForm.cs`（「内容」の右に「とても遅いが最高画質」のチェックボックス、変換中は無効）、`gui/CelV2Gui/AppSettings.cs`（`LastMaxEffort`）、`gui/tests/CommandTests.cs`、`gui/README_ja.txt`、仕様書の「使い方」

- [ ] **Step 1: テストを書く**: `MaxEffort = true` で引数に `--preset max` が入り、false で入らない。
- [ ] **Step 2: 失敗を確かめる → 実装 → `ctest -R gui_tests` を通す**
- [ ] **Step 3: コミット**

### Task 6: 計測と記録

**Files:** Create: `docs/design/2026-10-31-preset-max.md`。不採用があれば `docs/design/tried-not-kept.md`。

- [ ] **Step 1: VMAF NEG の BD を出すスクリプト**（scratchpad。avmdec で y4m に戻し、ffmpeg `libvmaf=model=version=vmaf_v0.6.1neg` で 4 点）。
- [ ] **Step 2: AV1 の最も遅い設定**（libaom cpu-used 0、SVT-AV1 preset 0）を、アニメ 5 本（AM 30 s、Dragon Maid、AIR、chika 30 s、Angel Beats 40 s）と実写 3 本（吉村、crowd_run、park_joy）の 120 フレームで符号化して保存（PSNR と VMAF）。
- [ ] **Step 3: CelV2 の既定、`--preset max`、`--preset max --gop-trial 0`、`--preset max --deblock-search 0` を同じ素材と混在区間 600 フレームで**（アニメは `--tune animation`、実写は `--tune film`）。
- [ ] **Step 4: 採否**: PSNR の BD が下がり VMAF NEG が悪化しない部分だけ `--preset max` に残す。速さ（秒/フレーム）を記録。
- [ ] **Step 5: 記録を書いてコミット**
