# CelV2 第1小プロジェクト 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** GPU（CUDA）で分割・イントラモード・動き探索を判定し、CPUでAVMの規格どおりの共通コードを使って正確に復元する、IPPP構成のAV2エンコーダ `celv2enc` を作る。出力は AVM デコーダでビット一致デコードできること。

**Architecture:** `io`（y4m/IVF）→ `gpu`（判定マップを作るだけ）→ `core`（判定マップどおりに予測・変換・量子化・復元）→ `bitstream`（デコーダの読み込み処理の鏡写しで書き出す）。デコーダと同じ状態を `state` が持ち、AVMの `av2/common` の関数をそのまま呼べるようにする。`verify` はプロセス内のAVMデコーダで、フレームごとのMD5を照合する。

**Tech Stack:** C++17、CUDA 13.3（nvcc + MSVC cl）、CMake + Ninja、AVM v1.0.0 の `avm.lib`（clang-cl、LTOなしでビルド）、CTest。

**Spec:** `<repo>\docs\superpowers\specs\2026-10-25-celv2-gpu-av2-encoder-design.md`

## Global Constraints

- リポジトリは `<repo>`。パスに日本語を含めない（nvcc対策）。
- AVMは `%AVM_ROOT%`、タグ `v1.0.0`。ライブラリ用ビルドは `%AVM_ROOT%\_build_lib`（clang-cl、Release、**LTOなし**、`-DENABLE_TESTS=0 -DENABLE_DOCS=0`）。
- **`av2/encoder/` 配下のヘッダをインクルードしない、関数も呼ばない。** `avm_dsp`（ビットライター、`entenc`、`fwd_txfm` を含む）、`av2/common`、`av2/decoder`（検証専用）は使ってよい。
- C++のコンパイルフラグに `/utf-8` を付ける（日本語環境でAVMのヘッダが誤読されるのを防ぐ）。
- 入力は8bit 4:2:0のy4mのみ。それ以外はメッセージを出して停止する。
- qindex は 1〜255。0（規格上ロスレスになる）は受け付けない。
- 不変条件：CelV2の復元フレームのMD5 == AVMデコーダの出力フレームのMD5（フレームごと、表示領域、I420）。
- 判定はGPU、復元はCPU。GPU→CPU の受け渡しは `DecisionMap` だけ。
- CUDAのエラーは `std::runtime_error`（メッセージ付き）で止める。黙って別の処理に切り替えない。
- ビルドは `powershell -File scripts\build.ps1`、テストは `powershell -File scripts\test.ps1 [-R 名前]`（Task 1で作る）。

## Review Focus

1. **幅と高さが8や64の倍数でない入力**（例：1920x1080、202x122）。下端と右端のSBが一部はみ出していても、ビット一致すること → Task 5 のテスト `keyframe_edge_sizes_bitexact`。
2. **奇数サイズ**（201x121）。4:2:0の色差サイズの切り上げ（(w+1)/2）が、デコーダと一致すること → Task 2 の `y4m_odd_size_chroma` と Task 5 の `keyframe_edge_sizes_bitexact`。
3. **SBより小さいフレーム**（16x16）でも、エンコードとデコードが通ること → Task 5 の `keyframe_edge_sizes_bitexact`。
4. **未対応の入力**（10bit、4:4:4、qindex 0）は、黙って壊れたストリームを出さずにメッセージで止まること → Task 2 の `y4m_rejects_unsupported`、Task 3 の `seqconfig_rejects_qindex0`。
5. **Pフレームの動きベクトルがフレームの外を指すとき**（端のブロックで大きなMV）でも、参照の端の外挿がデコーダと一致すること → Task 9 の `ippp_edge_motion_bitexact`。

---

## File Structure

```
celv2/
  CMakeLists.txt                 ルートのCMake（CXX、CUDA、AVMの取り込み、CTest）
  cmake/avm.cmake                avm.lib とインクルードディレクトリの取り込み
  scripts/env.ps1                vcvars64 と PATH の設定（他のスクリプトから読み込む）
  scripts/build_avm_lib.ps1      AVMのライブラリ用ビルド（_build_lib）
  scripts/build.ps1 / test.ps1   CelV2のビルドとCTest実行
  scripts/bench_vs_avm.ps1       M4の比較計測
  scripts/bdrate.py              BD-rateの計算
  src/io/y4m.{h,cpp}             y4mの読み込み（8bit 4:2:0、8の倍数へのパディング）
  src/io/ivf.{h,cpp}             IVFの書き出しと読み込み
  src/verify/avm_decode.{h,cpp}  プロセス内でのAVMデコードと、フレームごとのMD5
  src/bitstream/obu.{h,cpp}      OBUのヘッダとサイズの付与
  src/bitstream/seq_header.{h,cpp}   SeqConfig、シーケンスヘッダの書き出し、AVMのSequenceHeaderへの反映
  src/bitstream/frame_header.{h,cpp} フレームヘッダの書き出し（KEY/INTER）
  src/bitstream/block_writer.{h,cpp} 分割、モード、MV、係数の記号書き出し（デコーダの鏡写し）
  src/state/codec_state.{h,cpp}  AV2_COMMON、MACROBLOCKD、モード情報グリッド、参照バッファ
  src/core/quant.{h,cpp}         デッドゾーン量子化と、規格どおりの逆量子化
  src/core/encoder.{h,cpp}       SBループ（予測、変換、量子化、復元、block_writerの呼び出し）
  src/gpu/decision_map.h         GPUとCPUの境界になる型（CUDAに依存しない）
  src/gpu/engine.{h,cu}          GpuEngine（判定の入口）
  src/gpu/kernels.cu, kernels_ref.{h,cpp}  SATD、ピラミッド、動き探索のカーネルとCPU参照実装
  src/main.cpp                   CLI celv2enc
  tests/check.h                  最小のテスト用マクロ（CHECK、CHECK_EQ）
  tests/test_*.cpp               各タスクのテスト（1ファイル = 1つのCTest実行ファイル）
  docs/reports/                  M4のレポート
```

---

### Task 1: 土台（M0）— AVMライブラリのビルド、CMake、プロセス内デコードでの検証

**Files:**
- Create: `scripts/env.ps1`, `scripts/build_avm_lib.ps1`, `scripts/build.ps1`, `scripts/test.ps1`, `CMakeLists.txt`, `cmake/avm.cmake`, `src/verify/avm_decode.{h,cpp}`, `tests/check.h`, `tests/test_avm_decode.cpp`, `tests/no_encoder_includes.cmake`, `.gitignore`（`build/`）
- Test asset: `tests/data/avm_par8.ivf`（`(AVM で符号化した 1280x720 のストリーム)` のコピー。AVM製、1280x720、16フレーム）

**Interfaces:**
- Produces:
  - `std::vector<std::string> decode_frame_md5s(const std::vector<std::vector<uint8_t>>& temporal_units);` 出力フレームの表示領域を、I420のY→U→Vの順に連結してMD5を取り、16進小文字で返す。デコードエラーのときは `std::runtime_error`。
  - `std::string i420_md5(const uint8_t* y, int ys, const uint8_t* u, const uint8_t* v, int uvs, int w, int h);` 同じハッシュ規則。後のタスクで、自前の復元フレームに使う。
  - `std::vector<std::vector<uint8_t>> read_ivf(const std::string& path);` テスト用。Task 2 で `src/io/ivf` に移す。ここでは test 内のヘルパーでよい。
  - テストマクロ `CHECK(cond)` と `CHECK_EQ(a, b)`。失敗時はファイル、行、値を出して `return 1`。

- [ ] **Step 1: `scripts/env.ps1` と `scripts/build_avm_lib.ps1` を書き、AVMのライブラリ用ビルドを作る**
  - env.ps1：`C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat` の環境を取り込み、PATHに NASM（`$env:LOCALAPPDATA\bin\NASM`）、`C:\Program Files\LLVM\bin`、VS付属のNinjaを足す。`$env:CL='/utf-8'` も設定する。
  - build_avm_lib.ps1：`cmake -S %AVM_ROOT% -B %AVM_ROOT%\_build_lib -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl -DENABLE_TESTS=0 -DENABLE_DOCS=0 -DCMAKE_ASM_NASM_COMPILER=<nasm.exe>` を実行し、続けて `cmake --build ... --target avm`。
  - 実行して `%AVM_ROOT%\_build_lib\avm.lib` ができること。

- [ ] **Step 2: 失敗するテスト `test_avm_decode` を書く**
```cpp
// tests/test_avm_decode.cpp
int main() {
  auto tus = read_ivf(CELV2_TEST_DATA "/avm_par8.ivf");
  auto md5s = decode_frame_md5s(tus);
  CHECK_EQ(md5s.size(), 16u);
  CHECK_EQ(md5s[0].size(), 32u);
  CHECK(md5s[0] != md5s[1]);
  return 0;
}
```
  あわせて CTest `no_encoder_includes` を登録する。`src/` 配下で `av2/encoder/` を含む `#include` を検索し、1件でもあれば失敗するCMakeスクリプト。

- [ ] **Step 3: `CMakeLists.txt` と `cmake/avm.cmake` を書く**
  - `project(CelV2 LANGUAGES C CXX CUDA)`、`CMAKE_CUDA_ARCHITECTURES 120`、C++17、CXXには `/utf-8`。
  - avm.cmake：IMPORTED STATIC `avm` を作る。場所は `_build_lib/avm.lib`。インクルードは `${AVM_ROOT}`、`.../_build_lib`、`.../_build_lib/config` 相当のディレクトリ（`avm_config.h` があるディレクトリ）。
  - `${AVM_ROOT}/common/md5_utils.c` を `celv2_core` に含める。
  - 未解決シンボルが third_party（tflite、absl、flatbuffers、ruy など）から出た場合は、`_build_lib` 配下に生成された `*.lib` をすべて `avm` の `INTERFACE_LINK_LIBRARIES` に追加する。
  - テスト用の定数 `CELV2_TEST_DATA="${CMAKE_SOURCE_DIR}/tests/data"` を定義する。

- [ ] **Step 4: `src/verify/avm_decode.cpp` を実装する**
  `avm_codec_dec_init(&ctx, avm_codec_av2_dx(), nullptr, 0)` で初期化する。テンポラルユニットごとに `avm_codec_decode` を呼び、`avm_codec_get_frame` で取り出せたフレームをすべて `i420_md5` にかける。8bitの出力であることを前提に、`avm_image_t` の `planes` と `stride` を使う（`fmt` が高ビット深度のときは例外）。

- [ ] **Step 5: `scripts/build.ps1` と `scripts/test.ps1` を書いて、テストを実行する**
  - build.ps1：env.ps1 を読み込み、`cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl` を実行してからビルドする。
  - test.ps1：`ctest --test-dir build --output-on-failure` に `-R` を渡す。
  - 実行：`powershell -File scripts\build.ps1; powershell -File scripts\test.ps1`
  - 期待：`test_avm_decode` と `no_encoder_includes` が PASS。

- [ ] **Step 6: コミットする**
```bash
git add -A && git commit -m "build: scaffold CelV2 with AVM lib and in-process decode check"
```

---

### Task 2: y4mの読み込みとIVFの入出力

**Files:**
- Create: `src/io/y4m.{h,cpp}`, `src/io/ivf.{h,cpp}`, `tests/test_io.cpp`
- Modify: `tests/test_avm_decode.cpp`（`read_ivf` を `src/io/ivf.h` 版に置き換える）

**Interfaces:**
- Produces:
```cpp
struct Frame420 {               // 輝度は8の倍数までパディング（端の画素を複製）
  int width = 0, height = 0;    // 表示サイズ（奇数も可）
  int pw = 0, ph = 0;           // パディング後の輝度サイズ（8の倍数）
  std::vector<uint8_t> y, u, v; // y: pw*ph、u/v: (pw/2)*(ph/2)
  int stride_y() const { return pw; }
  int stride_uv() const { return pw / 2; }
};
class Y4mReader {
 public:
  explicit Y4mReader(const std::string& path);  // 8bit 4:2:0以外は std::runtime_error
  int width() const; int height() const; int fps_num() const; int fps_den() const;
  bool read(Frame420& out);                     // EOFでfalse
};
class IvfWriter {
 public:
  IvfWriter(const std::string& path, int w, int h, int fps_num, int fps_den);  // FourCC "AV02"
  void write(const std::vector<uint8_t>& tu, int64_t pts);
  void close();                                 // ヘッダのフレーム数を書き戻す
};
std::vector<std::vector<uint8_t>> read_ivf(const std::string& path);
std::vector<uint8_t> write_y4m_frame_bytes(const Frame420& f);  // テスト用。表示領域をy4mのフレームとして書く
```

- [ ] **Step 1: 失敗するテストを書く**（`tests/test_io.cpp`。一時ファイルは `std::filesystem::temp_directory_path()` に作る）
  - `y4m_roundtrip`：16x8のフレームを書いて読み、`width==16 && height==8 && pw==16 && ph==8`、各画素が一致する。
  - `y4m_odd_size_chroma`：201x121のフレームで `pw==208 && ph==128`。色差の表示領域は101x61の画素を保持し、パディング部は端の複製になっている（`u[60*104+101]==u[60*104+100]`）。
  - `y4m_rejects_unsupported`：ヘッダが `C444` のもの、`C420p10` のものは、どちらも `std::runtime_error` を投げる。
  - `ivf_roundtrip`：TUを2個書いて `read_ivf` で読むと、サイズと内容が一致する。ヘッダのFourCCが `AV02`、フレーム数が2。

- [ ] **Step 2: 実行して失敗を確認する**（`scripts\test.ps1 -R test_io`。未定義でビルドが通らない）
- [ ] **Step 3: `y4m.cpp` と `ivf.cpp` を実装する**（IVFの形式は32バイトのヘッダと、12バイトのフレームヘッダ。既存の `av2par.ps1` の連結処理と同じ形式）
- [ ] **Step 4: テストを実行する**（`scripts\test.ps1`。全テストPASS）
- [ ] **Step 5: コミットする**（`feat(io): y4m reader with padding and IVF read/write`）

---

### Task 3: OBUとシーケンスヘッダ（M1-a）。あわせて必須ツールの洗い出し

**Files:**
- Create: `src/bitstream/obu.{h,cpp}`, `src/bitstream/seq_header.{h,cpp}`, `tests/test_seq_header.cpp`
- Modify: 設計書 §2（必須ツールの一覧を追記する）

**Interfaces:**
- Produces:
```cpp
struct SeqConfig {
  int width, height;          // 表示サイズ
  int fps_num, fps_den;
  int qindex;                 // 1..255
  void validate() const;      // 範囲外は std::invalid_argument
};
// AVMのOBU_TYPEの値をそのまま使う（av2/common/enums.h、または obu_util.h の定義）
std::vector<uint8_t> make_obu(int obu_type, const std::vector<uint8_t>& payload);  // AVMのデコーダが受け付けるOBUヘッダとサイズフィールド
std::vector<uint8_t> write_sequence_header(const SeqConfig& c);   // 設計書§2のツールは無効
void fill_sequence_params(const SeqConfig& c, SequenceHeader* seq); // デコーダが解析した結果と同じ値を、AVMの構造体に入れる
```

- [ ] **Step 1: 必須ツールを洗い出す**（調査のみ。コードは書かない）
  `av2/decoder/decodeframe.c` のシーケンスヘッダ読み込み（`read_sequence_header_obu` と、ツールの有効化フラグを読む部分。6076行付近〜6290行付近）と、キーフレームの復号の流れを読む。そのうえで「どのフラグを0にしても、構文に残るもの」を一覧にし、設計書 §2 の末尾に「必須ツール」として追記する（例：IST、TCQ、DRL、変換分割の有無）。

- [ ] **Step 2: 失敗するテストを書く**（`tests/test_seq_header.cpp`）
  - `seqheader_peek_matches`：1920x1080の `SeqConfig` から、`make_obu(OBU_SEQUENCE_HEADER, write_sequence_header(c))` で作ったバイト列を `avm_codec_peek_stream_info(avm_codec_av2_dx(), ...)` に渡すと、`AVM_CODEC_OK`、`w==1920`、`h==1080` になる。
  - `seqheader_odd_size`：201x121 で `w==201`、`h==121`。
  - `seqconfig_rejects_qindex0`：`qindex=0` と `qindex=256` は、どちらも `validate()` が `std::invalid_argument` を投げる。

- [ ] **Step 3: 実行して失敗を確認する**
- [ ] **Step 4: 実装する**
  `make_obu` は、AVMの `av2/decoder/obu.c` のOBUヘッダ読み込み処理の鏡写しにする（拡張なし、サイズフィールドあり）。`write_sequence_header` は `avm_wb_write_*` を使い、読み込み処理を上から順に鏡写しにする。プロファイルは main、8bit、4:2:0、SBサイズは Step 1 の結論に従う。`fill_sequence_params` は、デコーダが同じヘッダを解析したときと同じ値を、`SequenceHeader` のフィールドに入れる。
- [ ] **Step 5: テストを実行する**（全PASS）
- [ ] **Step 6: コミットする**（`feat(bitstream): OBU framing and minimal-tool sequence header`。設計書の更新も含める）

---

### Task 4: 状態の構築とキーフレームの骨格（M1-b）。全ブロックDC予測、残差なし、64x64固定

**Files:**
- Create: `src/state/codec_state.{h,cpp}`, `src/bitstream/frame_header.{h,cpp}`, `src/bitstream/block_writer.{h,cpp}`, `src/gpu/decision_map.h`, `src/core/encoder.{h,cpp}`, `tests/test_keyframe_skip.cpp`

**Interfaces:**
- Consumes: Task 2 の `Frame420`、Task 3 の `SeqConfig`、`make_obu`、`write_sequence_header`、`fill_sequence_params`。Task 1 の `decode_frame_md5s`、`i420_md5`。
- Produces:
```cpp
// src/verify/avm_decode.h に追加：Frame420 の表示領域（width x height、色差は切り上げ）を i420_md5 にかける
std::string frame_md5(const Frame420& f);

// src/gpu/decision_map.h（CUDAに依存しない）
enum class Part : uint8_t { None, Horz, Vert, Split };
enum class IntraMode : uint8_t { DC, V, H, Smooth };
struct BlockDecision {
  uint8_t w4, h4;            // ブロックサイズ（4画素単位）
  bool inter = false;
  IntraMode intra = IntraMode::DC;
  int16_t mv_row = 0, mv_col = 0;  // 1/8画素単位
};
struct SbDecision {
  std::vector<Part> tree;          // 分割木の前順。SPLITは4つの子、HORZ/VERTは2つの子が続く
  std::vector<BlockDecision> blocks;  // 復号順の葉ブロック
};
struct DecisionMap { int sb_size; int sb_cols, sb_rows; std::vector<SbDecision> sbs; };
DecisionMap uniform_decision(int width, int height, int sb_size);  // 全SBをNONE＋DC（Task 4、5で使う）

// src/state/codec_state.h
class CodecState {
 public:
  explicit CodecState(const SeqConfig& c);   // av2_alloc_context_buffers、av2_init_mi_buffers 等を使う
  AV2_COMMON* cm(); MACROBLOCKD* xd();
  void begin_frame(bool key);                // フレームヘッダの値を cm に設定する。key のとき av2_setup_past_independence
  void end_frame();                          // 参照の更新（KEYは全スロット、INTERはスロット0）とCDFの保存
  Frame420& recon();                         // 現在フレームの復元バッファ（パディング込み）
};

// src/bitstream/frame_header.h
std::vector<uint8_t> write_frame_header(const SeqConfig& c, CodecState& st, bool key);

// src/bitstream/block_writer.h — avm_writer に書く。デコーダの読み込み処理の鏡写し
class BlockWriter {
 public:
  BlockWriter(CodecState& st, avm_writer* w);
  void partition(int mi_row, int mi_col, BLOCK_SIZE bsize, Part p);   // read_partition の鏡写し
  void intra_block(const BlockDecision& d, int mi_row, int mi_col);    // スキップ、モード（後で係数も）
};

// src/core/encoder.h
class Encoder {
 public:
  explicit Encoder(const SeqConfig& c);
  std::vector<uint8_t> encode(const Frame420& src, const DecisionMap& dm, bool key);  // 1フレームのTU（OBU列。最初のKEYのTUにはシーケンスヘッダを含む）
  const Frame420& last_recon() const;
};
```

- [ ] **Step 1: 失敗するテストを書く**（`tests/test_keyframe_skip.cpp`）
```cpp
// keyframe_skip_dc_bitexact: 128x128 の任意の原画（128で埋める）
SeqConfig c{128, 128, 30, 1, 120};
Encoder enc(c);
auto tu = enc.encode(src, uniform_decision(128, 128, sb), true);
auto md5s = decode_frame_md5s({tu});
CHECK_EQ(md5s.size(), 1u);
CHECK_EQ(md5s[0], frame_md5(enc.last_recon()));
```
  2つ目の画像として、640x360 でも同じ検査を行う。

- [ ] **Step 2: 実行して失敗を確認する**
- [ ] **Step 3: `CodecState` を実装する**
  AVMのデコーダが、フレームヘッダを読んだ直後に設定する状態（`av2/decoder/decodeframe.c` の `read_uncompressed_header` の後半と、タイル復号前の初期化処理）を、`av2/common` の関数で再現する。デコーダ専用の構造体（`AV2Decoder`）は使わない。
- [ ] **Step 4: `write_frame_header` を実装する**（`read_uncompressed_header` の鏡写し。KEY、表示する、qindex固定、ループフィルタはすべて無効、1タイル）
- [ ] **Step 5: `BlockWriter` を実装する**
  `partition` は `read_partition` の鏡写しにする。`intra_block` はスキップ（残差なし）の記号とDCモードの記号を、`av2/decoder/decodemv.c` のイントラの読み込みの鏡写しで書く。
  `Encoder::encode` は、SBのラスター順に分割木をたどる。各ブロックは `av2_predict_intra_block` で復元バッファに予測を書き（残差なしなので予測＝復元）、`BlockWriter` を呼ぶ。最後に `avm_stop_encode` でタイルを閉じ、タイルグループOBUとフレーム（ヘッダ）OBUを作る。OBUの構成は、AVMのデコーダが1フレームとして受け付ける形にする（AVMがエンコードしたIVFの1TUを `make_obu` の観点で分解し、同じ並びにする）。
- [ ] **Step 6: テストを実行する**（全PASS。不一致のときは、最初に食い違った4x4ブロックの座標をテストが出力するように、`tests/check.h` にヘルパー `report_first_diff(a, b, w, h)` を追加する）
- [ ] **Step 7: コミットする**（`feat: first bit-exact keyframe (DC, no residual)`）

---

### Task 5: 残差の符号化（M1-c）

**Files:**
- Create: `src/core/quant.{h,cpp}`, `tests/test_keyframe_residual.cpp`
- Modify: `src/core/encoder.cpp`, `src/bitstream/block_writer.{h,cpp}`

**Interfaces:**
- Consumes: Task 4 の `Encoder`、`BlockWriter`、`CodecState`。
- Produces:
```cpp
// src/core/quant.h
// 量子化（デッドゾーン、エンコーダ側なので自由）と、規格どおりの逆量子化（デコーダの decodetxb.c と同じ式）
int quantize(const tran_low_t* coeff, tran_low_t* qcoeff, tran_low_t* dqcoeff,
             TX_SIZE tx, int qindex, int plane);   // 戻り値 eob（全部0なら0）
// BlockWriter に追加
void BlockWriter::coeffs(int plane, int blk_row, int blk_col, TX_SIZE tx, TX_TYPE type,
                         const tran_low_t* qcoeff, int eob);  // av2_read_coeffs_txb の鏡写し
```

- [ ] **Step 1: 失敗するテストを書く**（`tests/test_keyframe_residual.cpp`。原画はテスト内で決定的に作る：斜めのグラデーション＋固定シードのノイズ）
  - `keyframe_residual_bitexact`：640x360、qindex 120 で、復元MD5とデコードMD5が一致する。
  - `keyframe_quality_monotonic`：qindex 60 の復元のPSNRが、qindex 160 の復元のPSNRより高い。qindex 60 のTUの方が大きい。
  - `keyframe_edge_sizes_bitexact`：1920x1080、202x122、201x121、16x16 のそれぞれでMD5が一致する。
- [ ] **Step 2: 実行して失敗を確認する**
- [ ] **Step 3: 実装する**
  - 順変換：`avm_dsp/fwd_txfm.c` の関数を使う。必要なサイズが無い場合は、`core` 内に整数近似のDCTを自作する（エンコーダ側なので規格の制約はない）。
  - 変換サイズ：各ブロックの最大変換サイズ（Task 3 で洗い出した必須の変換分割の構文に従う）。
  - 逆変換：`av2_inverse_transform_block` で復元バッファに加算する。
  - 係数：`BlockWriter::coeffs` で書く。コンテキストは `av2/common` の関数で求め、書き終えたらデコーダと同じように更新する。
- [ ] **Step 4: テストを実行する**（全PASS）
- [ ] **Step 5: コミットする**（`feat: residual coding for keyframes (M1 complete)`）

---

### Task 6: GPUの判定エンジン — SATDとイントラの分割・モード判定（M2-a）

**Files:**
- Create: `src/gpu/engine.{h,cu}`, `src/gpu/kernels.cu`, `src/gpu/kernels_ref.{h,cpp}`, `tests/test_gpu_intra.cpp`

**Interfaces:**
- Consumes: `Frame420`、`DecisionMap` とその関連の型。
- Produces:
```cpp
class GpuEngine {
 public:
  GpuEngine(int width, int height, int sb_size);  // デバイスが無い、またはサイズが0のときは std::runtime_error（メッセージ付き）
  DecisionMap decide_intra(const Frame420& src, int qindex);
};
// CPUの参照実装（テストの比較用）
int satd_ref(const uint8_t* a, int as, const uint8_t* b, int bs, int w4, int h4);  // 4x4アダマールの絶対値和
DecisionMap decide_intra_ref(const Frame420& src, int sb_size, int qindex);
```
  判定ルール（GPUとCPUで同一）：
  - 各ブロックで DC/V/H/SMOOTH の予測を**原画の隣接画素**で作り、`cost = SATD + λ·bits`。λ は `lambda_from_qindex(q) = 0.0005 * qstep(q)^2`。qstep は AVM の `av2_ac_quant_QTX(q,0,8)/8`。bits はモードごとに固定（DC=1、他=3、SPLIT/HORZ/VERT=2 ビット相当）。
  - 64→8 までボトムアップで、NONE/HORZ/VERT/SPLIT のうち最小コストを選ぶ。

- [ ] **Step 1: 失敗するテストを書く**
  - `gpu_satd_matches_ref`：ランダムな64x64ブロックの組100個で、GPUのSATDと `satd_ref` が完全一致する。
  - `gpu_intra_matches_ref`：640x360 の決定的な原画で、`decide_intra` と `decide_intra_ref` の DecisionMap が完全一致する（分割木とモードの全要素）。
  - `gpu_engine_rejects_zero_size`：`GpuEngine(0, 0, 64)` が `std::runtime_error` を投げ、メッセージが空でない。
- [ ] **Step 2: 実行して失敗を確認する**
- [ ] **Step 3: CPUの参照実装を書き、続けてCUDAのカーネルを書く**（1スレッドブロック = 1 SB。SB内のボトムアップは共有メモリで行う）
- [ ] **Step 4: テストを実行する**（全PASS）
- [ ] **Step 5: コミットする**（`feat(gpu): intra partition/mode decision with CPU reference`）

---

### Task 7: GPUの判定をキーフレームの符号化に接続する（M2-b）

**Files:**
- Create: `tests/test_keyframe_gpu.cpp`
- Modify: `src/core/encoder.cpp`, `src/bitstream/block_writer.cpp`（V/H/SMOOTH、HORZ/VERT/SPLIT の書き出し）

**Interfaces:**
- Consumes: Task 6 の `GpuEngine::decide_intra`、Task 5 の Encoder。

- [ ] **Step 1: 失敗するテストを書く**
  - `keyframe_gpu_bitexact`：640x360 と 1920x1080 で、`enc.encode(src, gpu.decide_intra(src, 120), true)` の復元MD5とデコードMD5が一致する。
  - `keyframe_gpu_better_than_uniform`：同じ qindex 120 で、GPUの判定を使った方が `uniform_decision` より TU が小さいか、PSNR が高い（どちらか一方を満たし、もう一方が悪化していない）。
- [ ] **Step 2: 実行して失敗を確認する**
- [ ] **Step 3: 実装する**（ブロックサイズとモードは DecisionMap どおり。分割の記号はTask 4の `partition` を使う）
- [ ] **Step 4: テストを実行する**
- [ ] **Step 5: コミットする**（`feat: GPU-decided keyframes (M2 complete)`）

---

### Task 8: インターのフレームヘッダと参照管理。GPUの階層動き探索（M3-a）

**Files:**
- Create: `tests/test_gpu_me.cpp`, `tests/test_inter_header.cpp`
- Modify: `src/gpu/engine.{h,cu}`, `src/gpu/kernels.cu`, `src/gpu/kernels_ref.{h,cpp}`, `src/bitstream/frame_header.cpp`, `src/state/codec_state.cpp`

**Interfaces:**
- Produces:
```cpp
DecisionMap GpuEngine::decide_inter(const Frame420& src, const Frame420& prev_src, int qindex);
DecisionMap decide_inter_ref(const Frame420& src, const Frame420& prev_src, int sb_size, int qindex);
```
  探索のルール（GPUとCPUで同一）：
  - 1/4、1/2、等倍の3段ピラミッド（2x2平均で縮小）。各段で ±8 の全探索をし、SADが最小のものを選ぶ（同点ならラスター順で先のもの）。
  - 等倍の段で、8x8ごとにMVを決める。分割・モードの判定は、インター候補（そのブロックのMVでSATD）とイントラ候補（Task 6 と同じ）の最小コストで行う。
  - MVは整数画素のみ（1/8単位で8の倍数）。小数画素はこの小プロジェクトの範囲外。

- [ ] **Step 1: 失敗するテストを書く**
  - `gpu_me_matches_ref`：決定的に作った原画と、それを (+5,-3) 画素ずらしたフレームで、`decide_inter` と `decide_inter_ref` が完全一致する。内部ブロックの大半（90%以上）のMVが (+5,-3)×8 になる。
  - `inter_header_parses`：キーフレーム1枚 + 全ブロックを `uniform_decision` のDC（イントラ）にしたINTERフレーム1枚をデコードでき、2フレームとも復元MD5とデコードMD5が一致する（ヘッダと参照管理の検査）。
- [ ] **Step 2: 実行して失敗を確認する**
- [ ] **Step 3: 実装する**
  - フレームヘッダ：`read_uncompressed_header` の INTER の分岐の鏡写し。参照は1枚（直前の復元フレームが入ったスロット）。
  - `CodecState::end_frame` のスロット更新規則：KEY は全スロット、INTER はスロット0。
  - ME のカーネルと参照実装。
- [ ] **Step 4: テストを実行する**
- [ ] **Step 5: コミットする**（`feat: inter frame header, refs, GPU hierarchical ME`）

---

### Task 9: インターブロックの符号化 — NEWMV/NEARMV、MV予測、インター予測（M3-b）

**Files:**
- Create: `tests/test_ippp.cpp`
- Modify: `src/core/encoder.cpp`, `src/bitstream/block_writer.{h,cpp}`

**Interfaces:**
- Produces:
```cpp
void BlockWriter::inter_block(const BlockDecision& d, int mi_row, int mi_col);
// MV候補は av2_find_mv_refs で、デコーダと同じ状態から作る。
// 判定のMVが候補と一致すればNEARMV系、そうでなければNEWMV（MV差分を書く）
```

- [ ] **Step 1: 失敗するテストを書く**
  - `ippp_bitexact`：決定的に作った、横にスクロールする映像 640x360 を8フレーム（I + P×7）。全フレームの復元MD5とデコードMD5が一致する。Pフレームの平均サイズがIフレームの半分未満。
  - `ippp_edge_motion_bitexact`：ずれ量 (+40,+24) 画素で、端のブロックがフレームの外を参照する映像 202x122 を4フレーム。全フレームで一致する。
- [ ] **Step 2: 実行して失敗を確認する**
- [ ] **Step 3: 実装する**
  - インター予測：`av2_build_inter_predictors` で、参照の復元フレーム（端の外挿込み）から予測する。
  - 残差は Task 5 と同じ処理。
  - モード、参照、MVの記号は、`av2/decoder/decodemv.c` のインターの読み込みの鏡写しで書く。
- [ ] **Step 4: テストを実行する**（全テストPASS）
- [ ] **Step 5: コミットする**（`feat: IPPP encoding bit-exact (M3 complete)`）

---

### Task 10: CLIとAVMとの比較計測（M4）

**Files:**
- Create: `src/main.cpp`, `scripts/bench_vs_avm.ps1`, `scripts/bdrate.py`, `tests/test_cli.cpp`, `docs/reports/2026-xx-xx-m4-celv2-vs-avm.md`（日付は実施日）

**Interfaces:**
- Produces: `celv2enc -i in.y4m -o out.ivf [--qindex 120] [--frames N] [--keyint 0]`。keyint 0 は「最初の1枚だけKEY」の意味。終了時に `speed: N frames / S s = X s/frame, Y fps` を表示する（`av2enc.bat` と同じ書式）。

- [ ] **Step 1: 失敗するテストを書く**（`tests/test_cli.cpp`。CLIを子プロセスで実行する）
  - `cli_encodes_and_decodes`：テスト内で作った32x32のy4mを3フレーム → 終了コード0、IVFが3TU、`decode_frame_md5s` で3フレーム取れる。
  - `cli_rejects_bad_args`：`--qindex 0` と存在しない入力は、終了コード≠0で、標準エラーにメッセージを出す。
- [ ] **Step 2: 実行して失敗を確認する**
- [ ] **Step 3: `main.cpp` を実装する**（Y4mReader → GpuEngine → Encoder → IvfWriter）
- [ ] **Step 4: テストを実行する**
- [ ] **Step 5: 比較計測を書いて実行する**
  - `bench_vs_avm.ps1 -Input <y4m> -Frames 30`：celv2enc と AVM（`%AVM_BIN%\avmenc.exe`、`--good --cpu-used=6 --end-usage=q --threads=1 --lag-in-frames=0`）を、qindex（AVMは `--qp`）120/150/180/210 の4点で実行する。
  - 時間、サイズ、PSNR（`bin\avmdec.exe` でデコードしてffmpegの`psnr`）をCSVに出す。
  - `bdrate.py` は、log(rate)対PSNRの3次多項式による標準のBD-rateを計算する。
  - 720p（`bench.y4m`）と1080p（`src1080.mp4` をy4mに変換したもの）で実行する。
- [ ] **Step 6: レポートを書いてコミットする**（速度比、BD-rate、気づいたこと、次の小プロジェクトの候補：復元のGPU化、デブロック、小数画素MV）。`feat: celv2enc CLI and M4 comparison report`
