# MSVC AVX2/FMA decimate 経路の検証（2026-11-22 05:26 JST）

Issue #231 の受入条件を実施した記録である。対象は #215（C-27）で、main `706f6829` のソースを使った。結論から書く。MSVC の AVX2/FMA 経路は実際に生成されており、スカラー経路より 1.7〜3.6 倍速い。ただし `/GL` を有効にした製品構成の実行ファイルからは、カーネルの関数を特定できない。状態は不明として留保に書いた。

## 1. 対象と環境

- ソース: `706f6829`。PR #232 のマージコミットである。
- 修正箇所: `src/CustomInputOversampler.cpp:212`、`:664` の `#if defined(__AVX2__) && (defined(__FMA__) || defined(_MSC_VER))`
- OS: Windows 11 Home、ビルド 26300
- CPU: 11th Gen Intel Core i7-11700K（8 コア、3.60 GHz）
- MSVC: 14.52.37025
- icx: 2026.2

## 2. 方法

### 2.1 ビルドとオブジェクト

- CMake と Ninja で MSVC Release を構成した。`CMakeLists.txt:2207` は `target_compile_options(ConvoPeq PRIVATE /arch:AVX2)` を全構成に適用する。`:2103` が `/GL` を付ける。
- `compile_commands.json` から実際のコンパイルコマンドを取り出した。対象は `src/CustomInputOversampler.cpp` である。これを基に 2 つ作った。
  - AVX2: 元のコマンドから `/GL` だけを外したもの（`/O2 /Ob2 /arch:AVX2`）
  - スカラー: AVX2 の `/arch:AVX2` を `/arch:SSE2` に置き換えたもの
- icx は同じコマンドを `icx-cl` に渡した。`/arch:AVX2` を `/QxCORE-AVX2` に替える。`/GL` は外す。
- 生成した `.obj` を `dumpbin /disasm` に掛けた。関数ごとに命令数と `ymm`、`vfmadd`、`zmm` を数える。
- 製品構成（`/GL` あり）の `ConvoPeq.exe` についても `dumpbin /symbols` と `/disasm` を実行した。

### 2.2 性能測定

- `loadStride2`（同ファイル 46-59 行）と `dotProductDecimateAvx2`（同 213-278 行）の本文をそのまま抽出し、スカラー側は同 701-702 行のフォールバックと同じ式で比較用の関数を作った。両方を同じ翻訳単位に入れた。フラグも同じである。
- 呼出しは各条件 200,000 回。ウォームアップの後に `QueryPerformanceCounter` で計測した。
- タップ数は 31、63、127 の 3 つ。入力は同一の正弦列だ。AVX2 とスカラーの結果は一致した（相対誤差 2.3e-16 以下）。
- これは decimate カーネル単体の測定だ。オーバーサンプラ全体の性能ではない。

## 3. 結果: 命令生成

| 構成 | `dotProductDecimateAvx2` | `decimateStage` |
| --- | --- | --- |
| MSVC `/arch:AVX2`（`/GL` なし） | シンボルあり。230 命令、ymm 67、vfmadd 9 | 283 命令、ymm 6、vfmadd 1 |
| MSVC `/arch:SSE2` | シンボルなし（`#if` で除外） | 233 命令、ymm 0、vfmadd 0 |
| icx `/QxCORE-AVX2` | シンボルあり。444 命令、ymm 228、vfmadd 21 | 551 命令、ymm 27、vfmadd 15 |
| MSVC Release（`/GL` あり、製品構成） | シンボルなし。状態は不明 | シンボルなし。状態は不明 |

- `/GL` ありの exe 全体には ymm 1,412,913 件、vfmadd 283,637 件がある。ただし関数への帰属はできない。全体の件数だけでは、どの関数が生成したか分からないからだ。
- 製品構成のシンボル欠落は、LTCG によるインライン化の可能性がある。しかし実行ファイルからは、インライン化か除去かを確認できない。したがって製品構成の状態は不明とする。確認できたのは `/GL` を外した構成だけである。
- `zmm` は無い。AVX-512 は使われていない。

## 4. 結果: 性能

MSVC の結果を示す。カーネル単体で 200,000 回呼んだ値である。

| タップ数 | AVX2 | スカラー | 倍率 |
| ---: | ---: | ---: | ---: |
| 31 | 10.15 ns/call | 17.10 ns/call | 1.68 倍 |
| 63 | 14.94 ns/call | 40.10 ns/call | 2.68 倍 |
| 127 | 25.50 ns/call | 90.98 ns/call | 3.57 倍 |

icx（参考）:

| タップ数 | 手書き AVX2 | 「スカラー」 | 倍率 |
| ---: | ---: | ---: | ---: |
| 31 | 12.04 ns/call | 1.83 ns/call | 0.15 倍 |
| 63 | 16.00 ns/call | 3.09 ns/call | 0.19 倍 |
| 127 | 26.08 ns/call | 6.76 ns/call | 0.26 倍 |

- 注意が要る。icx は `/QxCORE-AVX2` によりスカラーループも自動ベクトル化する。この行の「スカラー」はスカラーではない。つまり icx では手書き AVX2 が自動ベクトル化より遅い。MSVC の比較とは意味が違う。
- MSVC のスカラーループは自動ベクトル化されない。手書き AVX2 が 1.7〜3.6 倍速い。C-27 が対象にしたのは、まさにこの差である。

## 5. 受入条件の判定

| 受入条件 | 判定 | 根拠 |
| --- | --- | --- |
| MSVC の `dotProductDecimateAvx2` がシンボルとして存在する | 合格 | `/GL` を外した `/arch:AVX2` の `.obj` に存在。`/GL` ありではインライン化で消える |
| `decimateStage` に ymm 命令が 1 件以上ある | 合格 | MSVC で ymm 6 件、icx で 27 件 |
| decimate ループに FMA 命令が 1 件以上ある | 合格 | MSVC の `dotProductDecimateAvx2` に vfmadd 9 件、`decimateStage` に 1 件 |
| スカラー経路との命令数比較表 | 合格 | 本報告書 §3 |
| 性能測定の表 | 合格 | 本報告書 §4。カーネル単体の値である |

命令生成と性能は別の根拠として扱った。命令があるだけでは速いとは言えない。

## 6. 留保

- 製品構成（`/GL`）の実行ファイルからは、カーネルの関数を特定できない。インライン化か除去かは不明である。関数単位の命令数は測れない。本報告書の値は `/GL` を外した構成のものだ。
- 性能はカーネル単体の値だ。オーバーサンプラ全体の処理時間ではない。抽出元の行番号を明記し、結果の一致も確認した。
- 周波数固定とスレッド隔離はしていない。単一機種の値だ。反復も 1 回だけで、中央値は取っていない。各条件 20 万呼出しである。
- icx の比較は参考であり、MSVC の判断には使わない。

## 7. 再現手順

1. `cmake -S . -B build-msvc -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=cl -DCMAKE_EXPORT_COMPILE_COMMANDS=ON` で構成する（`MKLROOT`、`IPPROOT` を設定）。
2. `compile_commands.json` から `src/CustomInputOversampler.cpp` のコマンドを取り出し、`/GL` を外して `/Fo` を差し替えてコンパイルする。
3. 同じコマンドの `/arch:AVX2` を `/arch:SSE2` に変えて、スカラー版を作る。
4. `dumpbin /disasm` で関数ごとの命令、`ymm`、`vfmadd` を数える。
5. 製品構成の実行ファイルを作る。`cmake --build build-msvc --target ConvoPeq --parallel 4` の後に、`dumpbin /symbols ConvoPeq.exe` と `dumpbin /disasm ConvoPeq.exe` を実行する。
6. カーネルのベンチマークを `cl /O2 /Ob2 /arch:AVX2` でビルドし、31・63・127 タップで 20 万回呼ぶ。

## 8. 参照

- Issue #231（本検証）、#215（C-27）
- 修正コミット `712ecb55`（PR #229）
- 監査記録: `doc/work113/20261121-2233_PRI01照合/report.md`
- 対象ソース: `src/CustomInputOversampler.cpp`（`706f6829`）
