# サイズ指定エンコード（2 パス）実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 指定したファイルサイズを超えない動画を、下調べ（1 回目）と本番（2 回目）の 2 回の実行で作る。

**Architecture:** 1 回目は IVF のフレームごとのバイト数を統計ファイルに書く。2 回目は統計から
動画全体の qindex を 1 つ選び、`Pipeline` に「基準の qindex のずれ」を渡してミニ GOP ごとに
±2 まで直す（`SizeControl`）。GUI は 1 回目 → 音声 → 映像の予算 → 2 回目 → 多重化 → 確認を、
超えたら予算を減らしてやり直す。

**Tech Stack:** C++20（celv2enc、CMake + Ninja、MSVC）、C#（.NET Framework 4.8、`gui\build.ps1` の csc）。

**Spec:** `docs/superpowers/specs/2026-10-31-target-size-design.md`

## Global Constraints

- 上限は必ず守る。狙いは指定の 97〜100%。本番中の見積もりは予算の **98%** に収める。
- qindex とサイズの見積もり: **qindex 24 でビット数が半分**（`QualityProbe` と同じ）。
- 本番中に 1 回で動かす qindex の幅は **±2** まで、範囲は 1〜255。
- CLI のサイズは IVF ファイル全体（ヘッダ 32 バイト、フレームごと 12 バイトを含む）。GUI のサイズはファイル全体（映像＋音声＋コンテナ）、**1 MB = 1,000,000 バイト**（MiB の上限にも収まる側）。
- 2 回目の出力が予算を超えたら **終了コード 3**。GUI のやり直しは **最大 3 回**。
- 拒否するもの: `--pass 2` に `--stats` か `--target-size` がない、`--target-size` が 0 以下、統計の解像度・フレーム数が入力と合わない、`--pass` が 1 と 2 以外。理由をエラーに書く。
- 音声の形式は設定のまま（コピー / AAC / HE-AACv2）。サイズ指定のときは「元の動画より大きくしない」を使わない。
- `src/` に AVM のエンコーダのヘッダ（`av2/encoder/*`）を入れない。git の作者は既存の設定のまま（上書き禁止）。push しない。
- ビルド: PowerShell で `cmd /c '"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul && cmake --build <repo>\build 2>&1'`、テスト: `cd build; ctest -j 8`（GUI のテストも含む）。GUI だけなら `gui\test.ps1`（`FDKAAC_DIR`、`AVM_BIN`、`MP4BOX_DIR` を設定すると全部走る）。

## Review Focus

1. 予算が qindex 255 でも収まらないほど小さい: CLI は終了コード 3 と「qindex 255 でも N バイト」の
   メッセージ、GUI は 3 回やり直した後に理由つきで失敗（Task 4 と Task 6 のテスト）。
2. 2 回目に別の動画（同じ解像度、違うフレーム数）を渡した: 多いときは超えた時点で、少ないときは
   最後に、理由つきで失敗する（Task 4 のテスト）。
3. キーフレームだけの短い動画（3 フレーム、ミニ GOP なし）: 補正が一度も走らなくても予算に収まる（Task 4）。
4. `--preset max` と一緒に使う: GOP の試し用の `Pipeline` も同じずれで符号化し、予算に収まり、
   avmdec で復号できる（Task 2 と Task 4）。
5. 音声だけで指定サイズを超える: 映像を符号化する前に「音声だけで指定サイズを超えます」で失敗する（Task 6）。

---

### Task 1: 下調べの設定を決める（計測だけ、コードなし）

**Files:** なし（スクリプトは scratchpad）。結果は Task 8 の記録に入れる。

**Interfaces:**
- Produces: `--pass 1` が既定で切るオプションの一覧（Task 4 が使う）。

- [ ] **Step 1:** アニメ 4 本の混在区間 600 フレーム（AM 0 s、Angel Beats 155.36 s、AIR 54.45 s、
  Dragon Maid 64.73 s）と吉村 120 フレームを、q150 で次の 4 通りに符号化する: 本番
  （`--tune animation --preset max`、吉村は `--tune film --preset max`）、本番＋`--gop-trial 0
  --deblock-search 0`、それに `--rd-pass 0` を足したもの、さらに `--refs 4` を足したもの。
- [ ] **Step 2:** 120 フレームずつの区間ごとに「本番のバイト数 ÷ 候補のバイト数」を出し、素材ごとの
  最大 ÷ 最小と、候補の時間を表にする。
- [ ] **Step 3: 決める:** 比の最大 ÷ 最小が全素材で 1.15 以下の候補のうち、いちばん速いもの。どれも
  満たさなければ `--gop-trial 0 --deblock-search 0`（本番との違いが最も小さい）。決めたものを ledger に
  `Task 1: Ruling: 下調べ = <オプション>（比の幅 <数字>、時間 <数字>）` と書く。

### Task 2: `Pipeline` の基準 qindex のずれ

**Files:**
- Modify: `src/pipeline.h`, `src/pipeline.cpp`
- Test: `tests/test_pipeline_sync.cpp`

**Interfaces:**
- Produces: `void Pipeline::set_qindex_offset(int d)`、`int Pipeline::qindex_offset() const`。
  `code` / `code_group` / `show_existing` に渡された `FrameSpec::qindex_delta` に `d` を足してから
  使う（`CodecState` にも足した値が渡る）。`sse_per_bit()` は `cfg_.qindex + d`（1〜255 に丸める）で
  計算する。`sync_from` はずれも写す。

- [ ] **Step 1: 失敗するテストを書く** `pipeline_qindex_offset`:
  - `cfg.qindex = 130` の `Pipeline` と、`cfg.qindex = 120` に `set_qindex_offset(10)` の `Pipeline` で、
    同じ 3 フレーム（キーフレーム、`qindex_delta = 16` の非キーフレーム、`qindex_delta = 0` の非キーフレーム）
    を符号化する。`CHECK(obus_a == obus_b)` をフレームごとに、`CHECK(a.sse_per_bit() == b.sse_per_bit())`。
  - `c.sync_from(b)` の後 `CHECK(c.qindex_offset() == 10)`、次のフレームの出力が `b` と同じ。
- [ ] **Step 2:** ビルドして `ctest -R test_pipeline_sync`。Expected: コンパイルエラー（`set_qindex_offset` がない）。
- [ ] **Step 3:** 実装する。
- [ ] **Step 4:** `ctest -j 8`。Expected: 全部 PASS。
- [ ] **Step 5:** コミット `feat: Pipeline::set_qindex_offset, the sequence qindex moved without a new Pipeline`。

### Task 3: 統計ファイルと `SizeControl`

**Files:**
- Create: `src/io/stats.h`, `src/io/stats.cpp`, `src/core/size_control.h`, `src/core/size_control.cpp`
- Modify: `CMakeLists.txt`（`celv2_core` に 2 つの .cpp、`celv2_test(test_size_control)`）
- Test: `tests/test_size_control.cpp`

**Interfaces:**
- Produces:
  ```cpp
  namespace CelV2 {
  struct PassStats { int width = 0, height = 0, qindex = 0; std::vector<uint64_t> bytes; };  // bytes[i]: IVF の i 番目のフレームの中身（12 バイトの見出しを除く）
  void write_stats(const std::string& path, const PassStats& s);  // 失敗で std::runtime_error
  PassStats read_stats(const std::string& path);                  // 壊れた・違う形式で std::runtime_error（理由つき）
  class SizeControl {
   public:
    SizeControl(PassStats s, uint64_t budget);   // budget: IVF ファイル全体のバイト数
    static int start_qindex(const PassStats& s, uint64_t budget);
    int qindex() const;                          // 次のフレームの基準 qindex
    void coded(uint64_t payload);                // IVF に 1 フレーム書いた（中身のバイト数）。統計より多いと runtime_error
    void update();                               // ミニ GOP（またはキーフレーム 1 枚）の後
    size_t frames() const;                       // coded の回数
  };
  }
  ```
- 形式: 1 行目 `celv2-stats 1 <width> <height> <qindex> <frames>`、続いてフレームごとに 1 行、バイト数。
- 見積もり: フレーム i を qindex q で符号化したときの中身 = `bytes[i] · 2^(−(q − s.qindex)/24)`。
  ファイル = 32 + Σ(中身 + 12)。比 r = Σ 実際の中身 ÷ Σ（そのフレームを符号化した qindex での見積もり）、
  最初は 1。`start_qindex`: 全フレームの見積もりが `0.98 · budget` 以下になる最小の q（1〜255、255 でも
  超えるなら 255）。`update`: 32 + 使った分 + r · 残りの見積もり(q) + 12 · 残りのフレーム数 が
  `0.98 · budget` 以下になる最小の q を、今の q ± 2 と 1〜255 の中から選ぶ（なければ今の q + 2）。

- [ ] **Step 1: 失敗するテストを書く**（`tests/test_size_control.cpp`）:
  - `stats_round_trip`: `{64, 48, 120, {100, 0, 5000}}` を書いて読み、全部の値が同じ。1 行目が
    `celv2-stats 2 ...` のファイルと、フレームの行が足りないファイルで `read_stats` が例外。
  - `start_qindex_halves_per_24`: 100 フレーム各 10,000 バイト、q 120、予算 501,232（= 32 + 100·12 + 500,000）→
    `CHECK(start_qindex(s, budget) == 145)`（0.98 · 予算 − 1,232 = 489,975 に、q 144 の 500,000 は入らず q 145 の 485,670 が入る）。予算が
    十分大きいとき 1、qindex 255 の見積もりでも超えるとき 255。
  - `update_moves_at_most_2`: 同じ統計で、実際の中身を毎回見積もりの 2 倍にして 8 フレームずつ
    `coded` → `update` を繰り返すと、q が 1 回に 2 ずつ上がる。統計どおりの中身なら、q はいつも
    start − 2 以上 start 以下で、最後の 32 + Σ(中身 + 12) は 0.98 · 予算以下。
  - `coded_beyond_stats_throws`: 3 フレームの統計で 4 回目の `coded` が例外。
- [ ] **Step 2:** ビルド。Expected: `test_size_control` がコンパイルエラー（ヘッダがない）。
- [ ] **Step 3:** 実装する。
- [ ] **Step 4:** `ctest -R test_size_control`、続けて `ctest -j 8`。Expected: 全部 PASS。
- [ ] **Step 5:** コミット `feat: pass statistics and SizeControl (one qindex for the whole video, moved by 2 at most)`。

### Task 4: celv2enc の `--pass` / `--stats` / `--target-size`

**Files:**
- Modify: `src/main.cpp`（使い方の説明、`Args`、`parse`、メインのループ）
- Test: `tests/test_cli.cpp`

**Interfaces:**
- Consumes: Task 2 の `set_qindex_offset`、Task 3 の `PassStats` / `write_stats` / `read_stats` / `SizeControl`、Task 1 の下調べの設定。
- Produces: CLI。2 回目は最後に標準エラーへ `target-size: <ファイルのバイト数> of <予算> bytes` を
  出す。超えたら加えて `celv2enc: over --target-size by <N> bytes`（qindex 255 だったときは
  `even at qindex 255` を足す）を出し、IVF は残して終了コード 3。

- [ ] **Step 1: 失敗するテストを書く** `cli_target_size`（64x48、20 フレーム、前半 10 フレームは静止、
  後半は動く。`cli_preset_max` の作り方）:
  - 1 回目 `--pass 1 --stats s.txt --qindex 120` が 0 を返し、`read_stats` で 20 フレーム、64x48。
  - 2 回目 `--pass 2 --stats s.txt --target-size B` を、B = 1 回目の IVF の 60% で。0 を返し、IVF ≤ B、
    IVF ≥ 0.7·B（テストの動画は小さく、ミニ GOP も少ないので 97% は求めない）、stderr に `target-size: `、avmdec の復号が `--recon` と一致。
  - 同じことを `--preset max --tune animation` を付けて（Review Focus 4）。
  - B = 200 で終了コード 3、stderr に `over --target-size`（Review Focus 1）。
  - 3 フレームの動画（`--keyint 1`）で 1 回目 → 2 回目（その IVF の 80%）が 0 を返し予算以下（Review Focus 3）。
  - 拒否: `--pass 2 --stats s.txt`（`--target-size` なし）、`--pass 2 --target-size 1000`（`--stats` なし）、
    `--target-size 0`、`--pass 3`、存在しない統計、32x48 の動画に 64x48 の統計、19 フレームの動画に
    20 フレームの統計、21 フレームの動画に 20 フレームの統計（Review Focus 2）。それぞれ 0 以外を返し、
    stderr に理由（`--stats`、`--target-size`、`--pass`、`frames` などの語）が出る。
- [ ] **Step 2:** ビルドして `ctest -R test_cli`。Expected: FAIL（`--pass` を知らない）。
- [ ] **Step 3:** 実装する。
  - `--pass 1`: Task 1 で決めたオプションを、明示されていないときだけ切る（`unset` で判定、`--preset max`
    と `--tune` の既定の後に）。`output` で書いたフレームごとの中身のバイト数を集め、最後に `write_stats`。
  - `--pass 2`: `--qindex` は使わない。`read_stats` → 解像度を確かめる → `cfg.qindex =
    SizeControl::start_qindex(...)` を `SeqConfig` を作るところで使う。ループの各回の初め（キーフレーム、
    1 枚、ミニ GOP のどれでも）に `pipe->set_qindex_offset(sc.qindex() - cfg.qindex)`（試し用の
    `Pipeline` は `sync_from` で同じずれになる）、`output` で `sc.coded(obus.size())`、各回の終わりに `sc.update()`。
    最後に `sc.frames()` が統計のフレーム数より少なければエラー。
- [ ] **Step 4:** `ctest -j 8`。Expected: 全部 PASS。
- [ ] **Step 5:** コミット `feat: --pass 1/2, --stats and --target-size (exit 3 when over)`。

### Task 5: GUI の設定と引数

**Files:**
- Modify: `gui/CelV2.Settings/EncodeSettings.cs`, `gui/CelV2.Settings/CommandBuilder.cs`
- Test: `gui/tests/CommandTests.cs`, `gui/tests/SettingsTests.cs`

**Interfaces:**
- Produces:
  - `EncodeSettings.TargetSizeMB`（`double`、既定 0 = オフ）。
  - `CommandBuilder.EncodeArgs(EncodeSettings s, InputInfo i, string outputIvf, PassArgs pass = null)`。
    `PassArgs { int Pass; string Stats; long TargetBytes; }`。`Pass == 1` で `--pass 1 --stats <Stats>`、
    `Pass == 2` で `--pass 2 --stats <Stats> --target-size <TargetBytes>` を最後に足す。
  - `CommandBuilder.VideoBudget(long targetBytes, long audioBytes, long frames, bool mp4) → long`:
    `targetBytes − audioBytes − ContainerBytes(frames, mp4)`。`ContainerBytes`: 別ファイルは 0、
    MP4 は Task 6 の Step 1 で測った式。

- [ ] **Step 1: 失敗するテストを書く**:
  - `target_size_args`: `PassArgs{1, "s.txt", 0}` で引数の最後が `--pass 1 --stats s.txt`、`PassArgs{2, "s.txt",
    9000000}` で `--pass 2 --stats s.txt --target-size 9000000`、`null` で今と同じ引数。
  - `video_budget`: 別ファイル `VideoBudget(10_000_000, 1_000_000, 300, false) == 9_000_000`。MP4 は
    `ContainerBytes(300, true)` を引いた値。
  - 設定: 古いプリセット（`TargetSizeMB` なし）で 0。
- [ ] **Step 2:** `gui\test.ps1`。Expected: コンパイルエラー。
- [ ] **Step 3:** 実装する（`ContainerBytes` の MP4 の式は、Task 6 の Step 1 までは `4096 + 32 · frames` とし、Task 6 で直す）。
- [ ] **Step 4:** `gui\test.ps1`。Expected: 全部 PASS。
- [ ] **Step 5:** コミット `feat(gui): target size setting and the pass arguments`。

### Task 6: GUI の 2 回の実行とやり直し

**Files:**
- Modify: `gui/CelV2.Runner/EncodeJob.cs`, `gui/CelV2.Settings/CommandBuilder.cs`（`ContainerBytes` の式）
- Test: `gui/tests/RunnerTests.cs`

**Interfaces:**
- Consumes: Task 5 の `PassArgs`、`VideoBudget`、`TargetSizeMB`。

- [ ] **Step 1: MP4 のコンテナの分を測る:** 2 秒（48 フレーム）と 25 秒（600 フレーム）の音声つきの
  動画を MP4 にし、`mp4 − (ivf − 32 − 12·frames) − m4a` を出す。その最大を 1.5 倍して余裕を持たせた
  `a + b · frames` を `ContainerBytes` にし、測った値をコメントに書く。
- [ ] **Step 2: 失敗するテストを書く**:
  - `runner_target_size_separate`: 2 秒の音声つき、別ファイル、`TargetSizeMB` = 普通に変換したときの
    合計の 0.6 倍（MB に直す）。成功し、`ivf + m4a ≤ 指定`、`≥ 0.7 · 指定`、`Note` に `サイズ指定`、
    一時ファイル（`%TEMP%` の `celv2_*.stats` と 1 回目の IVF）が残らない。
  - `runner_target_size_mp4`（`Mp4Tools`）: 同じく MP4 で、`.mp4 ≤ 指定`。
  - `runner_target_size_too_small`: `TargetSizeMB = 0.001`（1,000 バイト）。失敗し、`Error` に
    `指定サイズ` を含む。出力フォルダにファイルが残らない。
  - `runner_target_size_audio_too_big`: 音声を 512 kbps の AAC にして、音声より小さい指定。映像の
    符号化の前に `音声だけで指定サイズを超えます` で失敗する（Review Focus 5）。
- [ ] **Step 3:** `gui\test.ps1`。Expected: 4 つが FAIL。
- [ ] **Step 4: 実装する**（`RunCore` で `TargetSizeMB > 0` のとき）:
  1. `KeepUnderSource` を使わない。
  2. `Stage("下調べ（1 回目）")`、`EncodeVideo` を `PassArgs{1, 統計}` で、`%TEMP%` の IVF に。
  3. 音声を今の方法で作る（長さは 1 回目の `framesCoded`）。音声 ≥ 指定なら上の理由で失敗。
  4. 予算 V = `VideoBudget(...)`。`Stage("本番（2 回目）")`、`EncodeVideo` を `PassArgs{2, 統計, V}` で。
     celv2enc の終了コード 3 は失敗にせず「超えた」と扱う。
  5. 多重化して、出力の合計 > 指定なら（または 3 で終わったなら）V を `V − 超えた分 − V/100` にして 4 から
     やり直す。やり直しは 3 回まで、それでも超えたら `指定サイズに収まりませんでした（最後: N バイト）`。
  6. 成功したら `Note` に `サイズ指定: N MB（指定の P%）`。一時ファイルはどの道でも消す。
- [ ] **Step 5:** `gui\test.ps1`。Expected: 全部 PASS。
- [ ] **Step 6:** コミット `feat(gui): target size runs two passes, and retries with a smaller video budget when over`。

### Task 7: GUI の欄と説明

**Files:**
- Modify: `gui/CelV2.Ui/MainForm.cs`, `gui/README_ja.txt`
- Test: `gui/tests/UiTests.cs`

- [ ] **Step 1: 失敗するテストを書く** `main_form_target_size`: `targetSize` という名前の
  `NumericUpDown`（0〜100000 MB、小数 1 桁）があり、値を 9.5 にすると渡す設定の `TargetSizeMB == 9.5`、0 で 0。
- [ ] **Step 2:** `gui\test.ps1`。Expected: FAIL。
- [ ] **Step 3:** 「画質」の行に `Caption("サイズ指定（MB）")` と欄を足す。0 のときは「オフ」と表示する。
  README の使い方の 5 の後に、サイズ指定の段落を足す（中身: 2 回変換する、時間は約 1.3〜1.5 倍、1 MB = 1,000,000
  バイト、音声の形式は設定のまま、「元の動画より大きくしない」は使わない）。
- [ ] **Step 4:** `gui\test.ps1`。Expected: 全部 PASS。
- [ ] **Step 5:** コミット `feat(gui): サイズ指定（MB） field`。

### Task 8: 計測と記録

**Files:** Create: `docs/design/2026-10-31-target-size.md`。

- [ ] **Step 1:** 仕様書の「目標と確かめ方」どおり: 120 フレームの 8 本と 600 フレームの混在区間 4 本、
  それぞれ q100、q150、q190 で qindex 一定に符号化したサイズの 0.9 倍を指定し、2 回目の出力のサイズ
  （指定比 %）、PSNR-Y、時間を記録する。比べる相手: 同じサイズになる qindex 一定の符号化（qindex を
  1 ずつ動かして指定以下で最も近いもの）の PSNR-Y。最後の 1 割のフレームの PSNR-Y を前半と比べる。
- [ ] **Step 2:** 目標（97〜100%、超えない、PSNR の差 0.1 dB 以内、時間 1.3〜1.5 倍）を満たさない素材が
  あれば原因を書き、直せるものは直して測り直す（直した内容は ledger の Ruling に）。
- [ ] **Step 3:** 記録を書いてコミット（Task 1 の決定と比の表も入れる）`docs: target-size results`。
