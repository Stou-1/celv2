# AVM ツール除去実験: CelV2 にないツールはどれだけ効くか（2026-10-25）

## 結論

- **ツールの差が大きく、判定（λ・RDO）と整数 MV の差は小さい**。AVM で CelV2 にないツールをまとめて切った `celv2like` は、AVM 既定より **+143%**。CelV2 は AVM 既定より +174%、`celv2like` より **+20.5%** だけ悪い。この 20.5% に、判定の質、整数 MV、残りの細かい差がすべて入っている。
- **小数画素 MV はこの素材ではほぼ効かない**。AVM を改造して MV を整数画素に固定すると、どのツール構成でも −0.7〜−0.2%（誤差の範囲）だった（下の「整数 MV の計測」）。つまり上の 20.5% は、ほぼ判定の質の差である。
- **候補3つの順位**: 判定/λ の調整（最大約 +20%）＞ デブロック（+1.2%、ループフィルタ全部でも +3.1%）＞ 小数画素 MV（約 0%）。
- **候補の外に、もっと効くものがある**。参照を1枚に減らすと **+34.9%**、イントラの追加ツール（CfL、MRLS、IBP、角度の微調整、Paeth、パレット、IntraBC、SDP など）を切ると **+24.7%**。ADST/IDTX（DCT のみで −1.4%）と拡張分割（−7.7%）は、この条件では差が見えない。

## 計測条件

| 項目 | 内容 |
|---|---|
| 素材 | `bench.y4m`（1280x720, 30fps）の先頭10枚、IPPP（最初の1枚だけ KEY） |
| 基準（base） | `avmenc --ivf --good --cpu-used=6 --end-usage=q --qp=Q --threads=1 --lag-in-frames=0 --limit=10` |
| 変種 | 基準に下表のオプションを足したもの。1変種×1Q を1プロセスとして、全部を並列に実行 |
| Q | 120, 150, 180, 210 |
| PSNR-Y | `avmdec --i420` で復号し、ffmpeg の psnr フィルタ（`-framerate 30/1`）で原画と比較 |
| BD-rate | 各変種を base と比べる（`python -I scripts/bdrate.py ablation.csv --anchor base`） |
| 速度 | エンコーダのプロセスの CPU 時間の合計（4つの Q の合計）を base と比べた比。並列実行で実時間は伸びるので CPU 時間を使う |
| 実行 | `scripts/ablate_avm.ps1`、生データは [ablation.csv](ablation.csv) |
| CelV2 | 参考として、同じ10枚を `build\celv2enc.exe` で符号化 |

## 結果（BD-rate の悪い順）

| 変種 | 切ったもの（avmenc のオプション） | BD-rate（base 比） | CPU 時間比 |
|---|---|---|---|
| CelV2（参考） | celv2enc そのもの（GPU の時間は CPU 時間に入らない） | +173.8% | 0.004 |
| celv2like | all_off ＋ 下の4グループ全部 | +143.0% | 0.085 |
| (i) all_off | (a)〜(d)、GDF、(f)、(g)、(h) を全部 | +45.6% | 0.346 |
| (h) one_ref | `--max-reference-frames=1` | +34.9% | 0.892 |
| intra_off | `--enable-intrabc/palette/cfl-intra/mrls/ibp/intra-dip/mhccp/paeth-intra/angle-delta/intra-edge-filter/sdp/extended-sdp=0` | +24.7% | 0.904 |
| lf_off | デブロック、CDEF、LR、CCSO、GDF を全部 | +3.1% | 0.672 |
| (d) no_ccso | `--enable-ccso=0` | +1.8% | 0.683 |
| (a) no_deblock | `--enable-deblocking=0` | +1.2% | 1.090 |
| (c) no_lr | `--enable-restoration=0` | 0.0%（出力が base とバイト単位で同じ） | 0.997 |
| no_gdf | `--enable-gdf=0` | −0.2% | 0.989 |
| (b) no_cdef | `--enable-cdef=0` | −0.6% | 1.005 |
| (e) 小数画素 MV | 改造した AVM で整数 MV に固定（下の「整数 MV の計測」） | −0.7% | 0.68 |
| inter_off | `--enable-warped-motion/warp-causal/warp-delta/refinemv/opfl-refine/tip/bawp/skip-mode=0` | −0.9% | 0.798 |
| quant_off | `--enable-trellis-quant=0 --enable-tcq=0` | −1.4% | 0.997 |
| (f) dct_only | `--use-intra-dct-only=1 --use-inter-dct-only=1` | −1.4% | 0.967 |
| (g) no_ext | `--enable-ext-partitions=0 --enable-uneven-4way-partitions=0` | −7.7% | 0.811 |
| sectx_off | `--enable-ist/inter-ist/inter-ddt/cctx/fsc=0 --enable-chroma-dctonly=1` | −9.7% | 0.987 |

関連する BD-rate: CelV2 は all_off 比 +99.8%、celv2like 比 +20.5%。celv2like は all_off 比 +76.2%。

**誤差の目安**: 素材1本、10枚、4点の計測なので、ツールを切って −9.7% や −7.7% のように「良くなる」変種も出ている。±10% 程度の差は、この計測では 0 と区別できない。意味のある差として読めるのは、one_ref、intra_off、all_off、celv2like の4つだけ。

## ツールごとの、CelV2 にとっての意味

- **(a) デブロック**: AVM では +1.2%（ほかのループフィルタも全部切ると +3.1%）。デコーダ側の処理なので入れやすいが、BD-rate の差はほとんど縮まない。
- **(b) CDEF**: −0.6%。この条件では効果が見えない。後回しでよい。
- **(c) ループ復元（LR）**: cpu-used=6 では AVM 自身が使っていない（`speed_features.c:556`、speed 5 以上で `disable_lr_filter = 1`）。CelV2 の比較相手としては不要。価値を知るには cpu-used≤4 での計測が必要。
- **(d) CCSO**: +1.8%。CPU 時間は 0.68 倍になるので、AVM の時間の約3割は CCSO の探索に使われている。CelV2 では優先度は低い。
- **GDF（追加）**: −0.2%。不要。
- **(e) 小数画素 MV**: −0.7%（celv2like の構成でも −0.2%）。この素材では作る価値がない。動きの多い素材（1080p の src1080 など）で確かめてから判断する。
- **(f) ADST/IDTX**: −1.4%。M4 レポートの候補にあった「ADST の追加」は、この条件では効果が見えない。
- **(g) 拡張分割**: −7.7%（誤差の範囲）。不要。矩形分割は CelV2 にすでにある（HORZ/VERT）。
- **(h) 複数参照**: **+34.9%。単独のツールでは最大**。Q120 で PSNR が 48.7 → 45.6 dB に落ちる。CelV2 の次の候補として、デブロックより先に検討する価値がある。
- **(i) 全部オフ**: +45.6%。個々の値の合計（約 +28%）より大きく、ツール同士が互いを補っている。
- **イントラの追加ツール（追加）**: 単独で +24.7%、all_off に重ねると差がさらに広がる（celv2like は all_off 比 +76%）。CelV2 のイントラは DC/V/H/Smooth だけなので、ここは大きな未開拓の部分。
- **判定/λ の調整**: CelV2 と celv2like の差 +20.5% が目安。整数 MV の影響はほぼ 0 なので、この差のほとんどは判定の質による。候補3つの中では最大。

## 整数 MV の計測

AV2 で整数 MV を伝える手段は `cur_frame_force_integer_mv` だけで、このフラグはスクリーンコンテンツツールが有効なフレームでしか書けない（`bitstream.c:5135`）。avmenc にはこれを強制するオプションがない（`encoder.c:4804`、エンコーダが内容を見て決める）。そこで AVM v1.0.0 を別のワークツリー（`%AVM_ROOT%-intmv`）で改造した（[avm-intmv.patch](avm-intmv.patch)）。環境変数 `AVM_ABL_INTMV` の意味は次のとおり。

- `1`: inter フレームでスクリーンコンテンツツールを有効にする（対照）
- `2`: さらに整数 MV を強制する

IntraBC はループフィルタを止めてしまうので、触っていない。整数 MV の効果は `_intmv` と `_sct` の差として読む。

| 構成 | スクリーンコンテンツツール（`_sct` 対 元） | 整数 MV（`_intmv` 対 `_sct`） | 整数 MV の CPU 時間比 |
|---|---|---|---|
| base | −4.9% | **−0.7%** | 0.68 |
| all_off | +3.2% | **−0.6%** | 0.68 |
| celv2like | −1.8% | **−0.2%** | 0.82 |

確認したこと:
- 改造版でパッチを切ったとき、12行すべてが標準のバイナリとバイト単位で一致した。
- 36本すべてが標準の avmdec で復号できた。
- PSNR を計り直しても、全行で同じ値になった。

生データは [ablation_intmv.csv](ablation_intmv.csv)。改造版は LTO 付きでビルドする必要がある。LTO なしの clang-cl ビルドは、パッチの有無に関係なく、毎回違う、復号できないストリームを出した（気づいたこと6）。

## 気づいたこと

1. **整数 MV を強制するオプションはない**。測るために AVM を改造した（上の「整数 MV の計測」）。
2. **`--enable-rect-partitions=0` は AVM のエンコーダが壊れる**。キーフレーム1枚、Q120 で、エンコーダ自身が報告する PSNR が 48.2 → 37.9 dB に落ち、バイト数は 1.38 倍になる。ML による枝刈りを切っても（`--use-ml-erp-pruning=0 --erp-pruning-level=0`、`--disable-ml-partition-speed-features=1`）、`--sb-size=64` にしても同じだった。最初の計測では (g) と (i) にこのオプションを入れて +925% と +1885% が出たので、矩形分割は外して計り直した。
3. **並列実行中の PSNR が2件だけ間違っていた**。64プロセス（48スレッド）で動かしたとき、`intra_off` Q180 と `CelV2` Q150 の PSNR が約3.4 dB 低く出た（34.94 と 34.99）。同じ .ivf を単独で計り直すと 38.09 と 38.42 になった。全68行を計り直し、違ったのはこの2件だけだったので、CSV は計り直した値に直してある。原因はわかっていない。
4. **10枚では CelV2 と AVM の差が小さく出る**。CelV2 対 AVM 既定は、10枚で +174%、M4 の30枚で +228%。P フレームが増えるほど、複数参照や判定の差が積み重なるとみられる。
5. **AVM の速さは 720p で1枚あたり約 13〜32 s**（cpu-used=6、1スレッド。base の Q150 で約31 s/枚）。celv2like にすると約12倍速くなる。
6. **LTO なしの clang-cl でビルドした AVM の avmenc は壊れている**。`%AVM_ROOT%\_build_lib`（CelV2 がリンクしている `avm.lib` を作るビルド）の avmenc は、同じ入力でも毎回違うストリームを出し、720p q210 の10枚では6枚目で復号できなくなった。LTO 付きのビルド（`_build_clang-native`、標準のバイナリ）は正常。CelV2 自身のテストは通っているが、このビルドは疑ってかかるべき。
7. **CelV2 の時間は `Start-Process -Wait` では正しく計れない**。この方法は終了を約1秒ごとにしか確かめないので、短い処理ほど長く出る。表の速度は CPU 時間で比べているので影響はない。CelV2 自身の `speed:` 行によると、30枚で約17 fps（高速 DCT を入れると35〜45 fps）。

## 次の一手

1. 判定/λ の調整（候補3つの中で最大の約20%）。
2. 複数参照（+35%）とイントラのツール（+25%）を、デブロックより先の候補に加えるかどうかを決める。
3. 小数画素 MV とデブロックの効果を、動きの多い素材で確かめる（この素材では約0%と約1%）。
