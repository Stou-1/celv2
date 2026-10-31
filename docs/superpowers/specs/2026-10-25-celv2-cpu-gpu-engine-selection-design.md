# CelV2: 判定エンジンの CPU / GPU 自動選択と複数世代ビルド（設計）

日付: 2026-10-25

## 目的

GPU が使えない、または古い PC でも celv2enc が動くようにする。搭載されている GPU を起動時に調べ、判定（分割・モード・動きベクトルの決定）を GPU で行うか CPU で行うかを自動で選ぶ。

### 前提（ユーザーとの合意）

- CPU の判定経路は「動かない環境をなくす非常口」。速さは二の次で、行うのはマルチスレッド化だけ。出力は GPU と同じにする。
- GPU で判定する対象は Turing 以降（compute capability 7.5 以上）。それより古い GPU は CPU で判定する。
- NVIDIA 以外の GPU は対象外（将来 Vulkan で対応する候補）。
- 実行ファイルは `celv2enc.exe` の1つ。

### 現状（2026-10-25 の計測）

- GPU の判定は RTX 5090 専用のビルド（`CMAKE_CUDA_ARCHITECTURES 120`）で、`GpuEngine` は CUDA デバイスがないと例外を投げる（CPU の代わりはない）。
- CPU で同じ判定を出す参照実装 `src/gpu/kernels_ref.cpp`（`decide_intra_ref` / `decide_inter_ref`）がテスト用にある。1スレッドで、720p の P フレーム1枚に 1,433 ms、1080p で 3,114 ms（GPU は 2.64 ms / 4.10 ms）。
- 判定は整数だけで計算しているので、CPU と GPU の判定は一致する（既存のテストが確かめている）。

## 全体像

起動の流れ:

1. `celv2enc` が `--device auto|gpu|cpu`（既定 `auto`）を読む。
2. `auto` のときは GPU を調べ（下の「GPU の確認」）、使えなければ CPU 版にする。
3. 選んだ結果を標準エラー出力に1行表示する。例: `CelV2: decisions on GPU (NVIDIA GeForce RTX 5090, compute 12.0)`、`CelV2: decisions on CPU (no usable CUDA device: <理由>)`。
4. `--device gpu` で GPU が使えなければ、CPU に切り替えずにエラーで終了する（終了コード 1）。

### 判定エンジンの共通の入口

- 抽象クラス `DecisionEngine`（`src/gpu/decision_engine.h`、CUDA にも AVM にも依存しない）:
  ```cpp
  class DecisionEngine {
   public:
    virtual ~DecisionEngine() = default;
    // Queues (GPU) or starts (CPU) the decision of every superblock row; sizes dm. src and *ref
    // may change once this returns. ref == nullptr: KEY frame (intra only).
    virtual void launch(const Frame420& src, const Frame420* ref, int qindex, DecisionMap& dm) = 0;
    // Publishes rows on gate from the top as they complete; on failure calls gate.fail and
    // rethrows. Returns with no work pending. Must follow each launch.
    virtual void collect(DecisionMap& dm, RowGate& gate) = 0;
  };
  ```
- `GpuEngine` と新しい `CpuEngine` がこれを実装する。`GpuEngine` の `launch` / `collect` は今のものをそのまま使う。
- `Pipeline` は `std::unique_ptr<DecisionEngine>` を持つ。フレームごとの流れ（launch → 回収スレッドで collect → 行ごとに待ちながら符号化）は変えない。

## CPU 版（`CpuEngine`、`src/gpu/cpu_engine.{h,cpp}`）

### 参照実装の分け直し（`src/gpu/kernels_ref.{h,cpp}`）

判定のロジックは変えず、行単位で呼べるように分ける。

- ピラミッドと階層的な動き探索を、「原画と参照画像のピラミッドを作る」と「ピラミッドの1段のうち、8x8 ブロックの行 [row0, row1) だけを探索する」の2つの関数に分ける（後者をスレッドで分担する）。探索の結果（各段の動きベクトルの場）は今の `MvField` のまま。正確な関数名と引数は実装計画で決める。
- `SbDecision decide_sb_ref(const Frame420& src, const MvField* mf, int sb_col, int sb_row, int sb_size, int qindex)`: 1つのスーパーブロックの判定（今の `decide()` のループの中身）。`mf == nullptr` はイントラのみ。
- 既存の `decide_intra_ref` / `decide_inter_ref` は、これらを呼ぶ形で残す（結果は変わらない）。

### 流れ

1. `launch`: 原画と参照画像を `CpuEngine` 自身のバッファにコピーし、`dm` の大きさを整え、作業スレッドを `decision_threads(...)` 本（下の「スレッドの本数」）起動してすぐ戻る。
2. 作業スレッド:
   - 動き探索（P フレームのみ）: ピラミッドの段ごとに 8x8 ブロックの行を分担し、段の変わり目で全スレッドがそろうのを待つ。
   - 判定: 共有のカウンタで次の行を取り、その行の全スーパーブロックを判定して `dm.sbs` に書き、その行を「完了」と記録する。
3. `collect`: 上の行から順に「完了」を待ち、完了済みの連続した行を `gate.publish(r)` する（行は順不同で終わるが、公開は上から順）。最後に作業スレッドを `join` する。

### 安全性

- 作業スレッドは自分が取った行の `dm.sbs` だけを書く。「完了」の記録と `collect` での確認はミューテックスを通すので、`publish` を受けたタイルスレッドからその行の中身が見える。
- 判定はスーパーブロックごとに独立しているので、スレッドの順番によらず結果は同じ。
- 作業スレッドの例外は記録して残りの作業を止め、`collect` が `gate.fail` を呼んでから投げ直す。`collect` は戻る前に必ず全スレッドを `join` する。

### テスト用の差し込み口

- `CpuEngine` のコンストラクタは、省略可能な `std::function<void(int row)>`（行を判定する直前に呼ぶ）を受け取る。既定は空で、通常の実行には影響しない。テストはこれで作業スレッドに例外を起こさせる。

### スレッドの本数（タイル符号化との取り合いの回避）

判定の作業スレッドとタイル符号化のスレッドが同時に動くと、合計がコア数を超えて取り合いになる。そこで、両方を合わせてコア数に収める。

- 判定の作業スレッドの本数 = max(1, コア数 − タイル符号化のスレッド数)。コア数は `std::thread::hardware_concurrency()`（0 なら 1 とみなす）、タイル符号化のスレッド数は `Encoder` が使う本数（タイル数、コア数、`--threads` の最小値）。
- 例（48 コア）: 2タイルなら判定 46 本 + 符号化 2 本、1タイルなら 47 本 + 1 本。
- CPU 版では判定の方がずっと遅い（1スレッドで 720p の P フレームに約 1.4 秒）ので、コアの大半を判定に回すのが全体として速い。タイル符号化のスレッドは、行が届くまで `RowGate` の条件変数で眠って待つので、その間はコアを使わない。
- 本数は `decision_threads(int cores, int tile_threads)` という小さな関数で決め、単体テストで確かめる。`Encoder` は、自分が使うタイル符号化のスレッド数を返す関数（例 `Encoder::tile_threads()`）を公開し、`Pipeline` がそれを `CpuEngine` に渡す。
- GPU 版にはこの調整は関係しない（GPU 版の回収スレッドは1本だけ）。

## GPU の確認（`probe_gpu()`、`src/gpu/engine.cu`）

- 戻り値: `struct ProbeResult { bool ok; std::string name; int cc_major, cc_minor; std::string reason; };`（`src/gpu/decision_engine.h` に置く。CUDA に依存しない）。使えるなら `ok = true` と名前・compute capability、使えないなら `ok = false` と理由。
- 順に確かめ、最初に失敗したところで理由を返す:
  1. `cudaGetDeviceCount` が成功し、1つ以上ある（ドライバがない・古い、デバイスがないなどは `cudaGetErrorString` の文言を理由にする）。
  2. デバイス 0 の compute capability が 7.5 以上（`compute X.Y is older than Turing (7.5)`）。
  3. 小さなカーネル（既存の `blocks_per_sb` の初回のタスク表作成）が成功する。ビルドにその GPU 用のコードがないと `cudaErrorNoKernelImageForDevice` がここで返る。
- 使うのはデバイス 0 だけ。CUDA 13 のランタイムは r580 以降のドライバが必要で、それより古いと 1 で失敗して CPU に切り替わる。

## エンジンの選択（`make_engine`、`src/pipeline.{h,cpp}`）

- `enum class Device { Auto, Gpu, Cpu };`
- `std::unique_ptr<DecisionEngine> make_engine(Device device, int width, int height, const std::function<ProbeResult()>& probe, std::string* description);` — `probe` を引数で受けるので、テストで GPU の確認結果を差し替えられる。`description` に、選んだ結果の1行を入れる。
  - `Auto`: 確認が成功すれば `GpuEngine`、失敗すれば `CpuEngine`（理由を `description` に入れる）。
  - `Gpu`: 確認が失敗すれば `std::runtime_error`（理由を含む）を投げる。
  - `Cpu`: 常に `CpuEngine`。
- `Pipeline(const SeqConfig& c, Device device = Device::Auto)`。`Pipeline::engine_description()` で1行の説明を返し、`celv2enc` がそれを表示する。既定の `Auto` は、今の GPU がある環境では今と同じ動き。

## ビルドの設定（`CMakeLists.txt`）

- `CMAKE_CUDA_ARCHITECTURES` を `75-real;80-real;86-real;89-real;90-real;120` にする（Turing、Ampere、Ada、Hopper は GPU コード、Blackwell は GPU コード + PTX）。RTX 5090 より新しい GPU は PTX を起動時にコンパイルして動く。
- CUDA ランタイムは静的リンクのまま（今の CMake の既定）。ドライバのない PC でも起動でき、GPU の確認が失敗して CPU に切り替わる。
- 代償: CUDA のコンパイル時間と実行ファイルの大きさが世代の数だけ増える。

## エラー処理

| 状況 | 動き |
|---|---|
| `auto` で GPU が使えない | CPU に切り替え、理由を1行表示する |
| `--device gpu` で GPU が使えない | 理由を表示して終了コード 1（切り替えない） |
| CPU の作業スレッドが失敗 | `gate.fail` でタイルスレッドを止め、エラーを返す |
| GPU が途中で失敗 | 今と同じくエラーで終了する（途中で CPU に切り替えない） |
| `--device` の値が不正 | 使い方の誤りとして終了コード 1 |

## テスト（実装より先に書く）

1. **CPU 版と GPU 版の判定が同じ**（`test_gpu_intra`）: 同じ入力を `CpuEngine` と `GpuEngine` に `launch` + `collect` で渡し、全スーパーブロックの `tree` と `blocks` が一致する。KEY と P、202x122、640x360、32x32。
2. **出力のバイト一致**（`test_ippp`）: `Pipeline(c, Device::Cpu)` と `Pipeline(c, Device::Gpu)` の全フレームの TU が同じ。640x360（1 / 2x1 タイル）、202x122（2x1 タイル）、640x360 の keyint 3。デコード一致も確かめる。
3. **CPU 版の行の公開**（新規 `test_cpu_engine`）: 全行が公開され、`ready_rows` が戻らずに行数に達する。待ちには時間制限を付け、超えたら理由を表示して `std::abort`。
4. **CPU 版のエラー**（`test_cpu_engine`）: 差し込み口で2行目に例外を起こすと、`collect` が `gate.fail` を呼んでから例外を投げ直し、待っているスレッドが止まったまま残らない。
5. **エンジンの選択とスレッドの本数**（`test_cpu_engine`、CPU のみ）: 偽の確認結果で、`Auto` + 失敗 → CPU 版と理由、`Gpu` + 失敗 → 例外、`Cpu` → 常に CPU 版。`decision_threads(48, 2) == 46`、`decision_threads(4, 4) == 1`、`decision_threads(1, 1) == 1`、`decision_threads(0, 1) == 1`。
6. **CLI**（`test_cli`）: 小さな入力で `--device cpu` と `--device gpu` の出力が同じ。`--device bogus` はエラー。環境変数 `CUDA_VISIBLE_DEVICES=-1` のもとで `--device auto` が CPU に切り替わり、`decisions on CPU` を表示し、出力が GPU 版と同じ。

## 成功の基準

- bench.y4m 720p と 1080p（30枚、q150、1タイルと2タイル）で、CPU 版と GPU 版の出力がバイト単位で同じ。GPU 版の出力は今の main と同じ。
- GPU 版の速さが今と変わらない（720p 2タイルで1枚約 9.1 ms、計測のばらつきの範囲）。
- `CUDA_VISIBLE_DEVICES=-1` で、CPU に切り替わって最後まで符号化できる。
- 新しいテストを含む全テストが通る。
- CPU 版の速さ（この PC での 720p と 1080p の1枚あたりの時間）を記録として報告する（目標は設けない）。

## 範囲外

- CPU 版の高速化（SIMD、候補の削減）
- NVIDIA 以外の GPU（Vulkan）
- Pascal 以前の GPU（CUDA 12 での別ビルドが必要）
- 複数の GPU からの選択
- 符号化の途中での CPU への切り替え
- RTX 5090 以外の GPU での実機確認（手元に機材がない）
