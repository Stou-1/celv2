# CPU/GPU 判定エンジンの選択と複数世代のビルド: 計測（2026-10-25）

## 結論

- **出力は全部同じ**。720p と 1080p、1タイルと2タイルの4組で、`--device gpu`、`--device cpu`、変更前のバイナリの IVF の MD5 が一致した。
- **GPU 版の速さは変わらない**。720p 2タイルで中央値 107.3 fps（1枚 9.3 ms）。変更前の約 106.8 fps と同じ範囲。
- **CPU 版**は 720p 2タイルで 1枚 105 ms（9.5 fps）、1080p 2タイルで 1枚 187 ms（5.4 fps）。同じ素材の GPU 版（107 fps、57 fps）の 1/11〜1/10。
- **GPU が見えないとき**（`CUDA_VISIBLE_DEVICES=-1`）、`--device auto` は CPU に切り替わり、最後まで符号化して、GPU 版と同じ MD5 を出した。
- **確かめたのは RTX 5090（compute 12.0）だけ**。Turing〜Hopper 向けのコード（sm_75/80/86/89/90）はビルドできることまでしか確かめておらず、実機では動かしていない。

## ビルド

`CMAKE_CUDA_ARCHITECTURES` を `120` から `75-real 80-real 86-real 89-real 90-real 120` に変えた（Turing、Ampere 2種、Ada、Hopper は実行コードのみ。Blackwell は実行コードと PTX）。`build\` を消して `scripts\build.ps1` で全体を作り直した。

| 項目 | 変更前（120 のみ） | 変更後（6世代） |
|---|---|---|
| 全体のビルド時間（新規、48 コア） | 5.1 秒 | 16.9 秒 |
| `celv2enc.exe` の大きさ | 11,077,120 バイト | 11,551,744 バイト（+4.3%） |

注意: 変更前の 5.1 秒は、6世代のビルドの後に、別のディレクトリ（`build_dbg`）で、`CMakeLists.txt` を変更前に戻して（`git stash`）ビルドして測った値。もとの `build\` を消して測ったものではなく、同じ手順での厳密な比較ではない（キャッシュなどの影響もありうる）。ビルド時間は目安として読むこと。exe の大きさは、変更前の `build\` の `celv2enc.exe` の値。

`build\build.ninja` の nvcc の引数に、世代ごとの `--generate-code` が入っていることを確かめた。

```
--generate-code=arch=compute_75,code=[sm_75]
--generate-code=arch=compute_80,code=[sm_80]
--generate-code=arch=compute_86,code=[sm_86]
--generate-code=arch=compute_89,code=[sm_89]
--generate-code=arch=compute_90,code=[sm_90]
--generate-code=arch=compute_120,code=[compute_120,sm_120]
```

テストは 16/16 が通った（作り直した後の全体）。

## 計測条件

| 項目 | 内容 |
|---|---|
| 素材 | `bench720.y4m`（1280x720）と `src1080_30.y4m`（1920x1080）、どちらも30枚 |
| 符号化 | `--qindex 150 --frames 30`、`--tile-columns 0 / 1`（1 / 2タイル） |
| 参照 | 720p は main と同じ出力の MD5。1080p は `--device` の無い変更前のバイナリ（`celv2_ref`）の出力 |
| PC | RTX 5090、48 コア、Windows 11 |
| MD5 | `Get-FileHash -Algorithm MD5` |

## 出力の一致

| 素材 | タイル | 大きさ（バイト） | 参照 | `--device gpu` | `--device cpu` |
|---|---|---|---|---|---|
| 720p | 1 | 117276 | `6D53C123849E0467E3901255C3337C04` | `6D53C123849E0467E3901255C3337C04` | `6D53C123849E0467E3901255C3337C04` |
| 720p | 2 | 118685 | `596938CFC45108D927756465844D5CB8` | `596938CFC45108D927756465844D5CB8` | `596938CFC45108D927756465844D5CB8` |
| 1080p | 1 | 208388 | `DDBC17597687FBBA6AB71DFD112A383D` | `DDBC17597687FBBA6AB71DFD112A383D` | `DDBC17597687FBBA6AB71DFD112A383D` |
| 1080p | 2 | 210053 | `48EC3C048F538C8284B7EA91E63158F7` | `48EC3C048F538C8284B7EA91E63158F7` | `48EC3C048F538C8284B7EA91E63158F7` |

## 速さ（2タイル、`speed:` 行）

GPU 版（720p、5回）:

| 回 | 1 | 2 | 3 | 4 | 5 | 中央値 | 幅 |
|---|---|---|---|---|---|---|---|
| fps | 106.4 | 107.6 | 107.9 | 105.0 | 107.3 | **107.3**（9.3 ms/枚） | 105.0〜107.9 |

変更前の約 106.8 fps（約 9.1 ms/枚）と比べて、中央値で +0.5%。GPU のクロックによる揺れの範囲内。1080p の GPU 版は1回だけ測って 57.5 fps。

CPU 版（3回ずつ。1枚あたりは 1000 / fps）:

| 素材 | fps（3回） | 中央値 | 1枚あたり |
|---|---|---|---|
| 720p | 9.54 / 9.53 / 9.43 | 9.53 | **約 105 ms**（104.8〜106.0） |
| 1080p | 5.36 / 5.33 / 5.37 | 5.36 | **約 187 ms**（186.3〜187.6） |

## GPU が無いときの動き

`CUDA_VISIBLE_DEVICES=-1` を付けて `--device auto`（720p、2タイル）:

- stderr に `CelV2: decisions on CPU (no CUDA-capable device is detected)`。
- 終了コード 0、9.6 fps（9.555）で最後まで符号化。
- IVF の MD5 は `596938CFC45108D927756465844D5CB8`（`--device gpu` と同じ）。
- 確かめた後、環境変数は消した。

## 確かめていないこと

- **RTX 5090 以外の GPU では動かしていない**。sm_75〜sm_90 のコードはコンパイルされて exe に入っているが、実機での実行と、その出力が同じになることは未確認。
- ドライバ r580 未満や、Turing より古い GPU のときの `--device auto` の動き（CPU に切り替わるはず）は、`CUDA_VISIBLE_DEVICES=-1` で GPU を見えなくした場合だけ確かめた。
- CPU 版の時間はこの PC（48 コア）での値。コア数が少ない PC ではもっとかかる。
