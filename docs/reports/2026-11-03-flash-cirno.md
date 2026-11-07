# Flash の録画（チルノのパーフェクトさんすう教室）で CelV2 と AV1 を比べる（2026-11-03）

## 結論

- CelV2（feat/gui-details 2042138）は AV1 の最も遅い設定より **SVT-AV1 p0 比 +59.4%、libaom cpu0 比 +88.4%** 大きい（同じ画質で）。2026-11-02 の他の素材（+30〜+84%、2026-11-02-av1-gap.md）と同じ幅で、Flash だけが特に悪いわけではない。
- AV1 側でスクリーンコンテンツの道具（SVT `scm=0`、libaom のパレットと IntraBC）や warp / global motion を切っても、差はほとんど動かない（libaom: +90.5% → +87.8% / +89.2%）。AV1 の勝ちはそれらの道具ではない。
- 1 フレームだけ（イントラ）で比べると、CelV2 は同じ道具の avmenc（cpu-used 6）より Flash で +43〜+62%、普通のアニメでも +17〜+47% 大きい。キーフレームの符号化の効率が大きく劣っている。長い静止とカットの多い Flash では、これが全体に効く。
- Flash では AV2 の道具にも効くものがある（avmenc、30 フレーム）。warp-delta が −19.6%、パレット＋IntraBC が −13.4%。普通のアニメでは ±1% 以内だった道具（2026-10-30、2026-11-02）。

## 結果（BD-rate、CelV2 の AV1 比。正は CelV2 が大きい）

| 区間（各 240 フレーム） | SVT-AV1 preset 0 | libaom cpu-used 0 |
|---|---|---|
| 20 秒から | +59.9% | +91.7% |
| 60 秒から | +56.4% | +90.5% |
| 100 秒から | +60.3% | +83.5% |
| 3 区間の合計 | +59.4% | +88.4% |

60 秒からの区間で AV1 側の道具を切ったとき（CelV2 の比）:

| AV1 の設定 | BD-rate |
|---|---|
| SVT p0 そのまま / `scm=0` | +56.4% / +57.3% |
| libaom cpu0 そのまま / パレットと IntraBC なし / warp と global motion なし | +90.5% / +87.8% / +89.2% |

## 1 フレーム（イントラ）での差

CelV2 `--tune animation` の、各エンコーダーに対する BD-rate（4 点、q100〜190 / crf 20〜56）。

| 画 | avmenc cpu6（CelV2 の道具） | SVT p0 | libaom cpu0 |
|---|---|---|---|
| チルノ 60 秒から 0 枚目 | +62.0% | +67.0% | +84.8% |
| チルノ 20 秒から 0 枚目 | +43.5% | | |
| チルノ 100 秒から 0 枚目 | +57.9% | | |
| AIR OP 60 枚目 | +46.9% | | |
| Dragon Maid 0 枚目 | +42.0% | | |
| AMBITIOUS MISSION 0 枚目 | +17.3% | | |

- CelV2 の設定で差はほとんど縮まない（チルノ 60 秒から 0 枚目、CelV2 の基準比）: `--preset max` / `--rd-intra 1` +0.2%、`--tcq 0` −3.3%、`--ist 0` +1.2%、`--intra-txp 0` +9.3%、`--mrls 0` +1.1%、`--cfl 0` +0.9%、`--rd-pass 0` +7.0%、`--cdef 0` +8.7%、`--lr 0` −0.1%、`--ccso 0` +2.0%、`--deltaq 0` / `--deblock-search 1` ±0%。
- avmenc 側で 1 つずつ制限したとき（avmenc の基準比）: cpu-used 0 で −28.6%、Paeth なし +8.5%、イントラの IDTX なし +5.8%、DCT のみ +2.5%、縦長・横長の分割なしでバイト数がおよそ 2 倍、ほか（64x64 上限、角度の微調整、smooth、エッジフィルタ、変換分割、8x8 下限）は ±2% 以内。CelV2 は縦長・横長の分割も Paeth も使えるので、差は道具ではなく探索と判定にある。

## AV2 の道具（avmenc cpu-used 6、60 秒からの区間の 40〜69 枚目、q145〜190、CelV2 の道具を基準）

| 足した道具 | BD-rate |
|---|---|
| warped motion のみ / ＋warp causal | ±0% / ±0% |
| ＋warp-delta | −19.6% |
| ＋warp-delta、warp-extend | −19.5% |
| ＋warp-delta、6 パラメータ | −19.9% |
| warp 全部 | −20.3% |
| パレット＋IntraBC | −13.4% |

同じ 30 フレームで、CelV2 は avmenc の基準より +9.3%。

- 0〜29 枚目（シーンの切り替えを含む）、q160 / 190 の 2 点では、パレットのみ −12〜16%、IntraBC のみ −22%、両方（＋`--tune-content=screen`）−22〜30%（バイト数、PSNR はほぼ同じ）。
- avmenc はこの素材で q100 / 130 のとき異常終了することがある（0xC0000028、CelV2 の道具の組み合わせで 30 フレーム）。

## 計測条件

| 項目 | 内容 |
|---|---|
| 素材 | 1440x1080 30 fps、可逆 RGB の AVI を yuv420p にした 3 区間（20 / 60 / 100 秒から各 240 フレーム）。240 枚のうち重複しない画は 168 枚 |
| CelV2 | `--qindex 100/130/160/190 --refs 7 --keyint 240 --tune animation` |
| AV1 | ffmpeg の libsvtav1 `-preset 0 -crf q -svtav1-params tune=1` / libaom-av1 `-cpu-used 0 -crf q -b:v 0 -row-mt 1 -tiles 1x1`、q 20/32/44/56 |
| 画質 | ffmpeg psnr の全体 MSE による PSNR-Y。合計は区間ごとのバイト数の和と MSE の平均 |
| スクリプト | scratchpad gap/flash.py、flash_sc.py、flash_intra.py、flash_iabl.py、intra_vs_avm.py、flash_avmi.py、flash_avm2.py、flash_avm3.py |
