# 外部監査 Part 8 指摘の不具合詳細調査（実行時検証 追加分を含む）

- 初回調査: 2026-11-09 02:10
- 追加実測: 2026-11-09 10:00〜11:00
- 基準 HEAD: `2aecd73e823067e8cf26c2d4a8fa2e4e6c6d3166`（main）
- working tree: クリーン（本報告書のフォルダのみ未追跡）
- 方針: 現行ソースと実行時測定のみを根拠とする。過去の監査記述は現行ソースと一致する場合に限って参照した
- 制約遵守: Production コード・テスト・CMakeLists の変更なし、git commit / PR / merge なし、リポジトリ外の測定アプリのみ新規作成

## §1 Executive Summary

Part 8 の指摘 7 系統（A〜G）を、現行ソースと実行時測定で再検証した。追加実測により **判定 1 件を格上げ（A）、影響度 1 件を下方修正（C）** した。

| 分類 | 件数 | 内訳 |
|---|---|---|
| Confirmed Bug | 3 | A stderr 未ドレインによる読み込み停止（新規格上げ）、E ポリフェーズ利得損失（既知）、F バイパス時の IR 切断（既知） |
| Potential Bug | 3 | updateConvolverState の競合時破棄順、リビルド起源コミットの世代刻印、TruePeak の容量超サニタイズ省略 |
| Confirmed Issue / Non-Bug | 7 | C 群遅延クランプ（実測で未発動）、D README 件数、G CI 構成、リミッタ NaN（本番到達不能）、EQ mono コピー（本番到達不能）、`[IR_RATE_GEN]` 表示、`type only` 残留文言 |
| Not Reproduced | 0 | — |
| Not Applicable | 1 | B 最小位相スキップ（トリム上限により到達不能） |

今回の追加実測で確定した事実は次の 5 点である。

1. **A（stderr）は Low ではなく Medium**。IR ロード 1 回あたり 16.0 行・約 2.5 KB を無条件に stderr へ書く。stderr が「読まれないパイプ」の場合、ロードは 90 秒経過しても完了せず、読み出しを開始した直後に `IR load failed: len=0 finalized=0` で失敗した（exit 2）。診断出力の存在自体ではなく、**書き込みがブロックして読み込みが失敗する**点が不具合である。
2. **E の補正フラグを ON にすると利得が復元する**。同一測定で DC 利得が 0.750000000 / 0.562500000 / 0.421875000 から、すべて 1.000000000 になった。
3. **F の切断を波形で確認した**。Bypass（tailMode=2）では保持長（block 256/512/1024 で 0.1707 / 0.3413 / 0.6827 秒）までが誤差 3e-16 で一致し、それ以降は 3e-17 以下の無音になる。IR 末尾 100 ms には −11.9 dB のエネルギーがある。
4. **C の 120 サンプルクランプは、5 種類の IR で一度も発動しなかった**。クランプ後の目標群遅延はすべて 5 サンプルで、制限が掛かっていない。文書化ギャップは残るが、実際の影響は小さい。
5. **G は `build/compile_commands.json` と生成ヘッダ `JuceHeader.h` の両方が必要**。DB だけでは `JuceHeader.h not found` で監査が落ち、ヘッダを生成すると PASS する。ワークフローに configure とヘッダ生成のステップが無いことが原因で、コードの問題ではない。

## §2 調査環境

| 項目 | 値 |
|---|---|
| HEAD | `2aecd73e823067e8cf26c2d4a8fa2e4e6c6d3166` |
| ブランチ / working tree | main / クリーン |
| ビルド構成 | CMake Ninja Multi-Config、Release |
| コンパイラ | MSVC 14.51.36231（Visual Studio 18 Enterprise） |
| ライブラリ | JUCE 8.0.12、Intel oneMKL 2026.2 / IPP 2026.1、r8brain |
| 既存実行ファイル | `PolyphaseGainMetrics.exe`、`MTNUPCMeasurement.exe`、`AudioEngineHarness.exe` |
| 追加ビルド | `build-polyon`（`CONVOPEQ_CORRECT_POLYPHASE_GAIN=ON`）、リポジトリ外の測定アプリ `TailTrunc` |
| 測定用スクリプト | Python 3.15（リポジトリ外） |
| 変更禁止の遵守 | リポジトリ内のソース・テスト・CMake は無変更。追加したのはリポジトリ内の未追跡ビルドツリー `build-polyon` と、リポジトリ外の測定アプリ・スクリプトのみ |

## §3 指摘別詳細

### A. LoaderThread の無条件 stderr 出力

- 判定: **Confirmed Bug（Medium、条件付き）**
- 関連ファイル: `src/convolver/ConvolverProcessor.LoaderThread.cpp`
- 呼び出し経路: `loadImpulseResponse` → LoaderThread → 各ステップの診断出力
- Production 到達性: あり（IR ロードのたび）。ただし停止に至るのは stderr が未ドレインのパイプのとき
- 発生条件: 親プロセスが stderr をパイプで受け取り、読み出さないまま子の完了を待つ
- 期待される挙動: 診断出力は既定で抑止される、または書き込みがブロックしない
- 実際の挙動:
  - 1 ロードあたり **16.0 行 / 約 2,495 B** を stderr へ書く（実測 12 ロードで 29,940 B）
  - stderr を読まないパイプにすると、**90 秒経過しても終了しない**（OS のパイプバッファ約 4.7 KB で停止）
  - 読み出しを開始すると 4,719 B を読んだ時点で `[N1-analysis] IR load failed: len=0 finalized=0 peak=0 err=''` を出して **exit 2**
- 対照実験: stderr=DEVNULL と stderr=ファイルはともに **1.6 秒で exit 0**
- 根拠（ソース）: `fprintf(stderr, ...)` 7 箇所、`printChainCheckpoint` 5 回（本体 2 箇所）、`[IR_TAIL_ENV]` は IR 長に比例（1 ロード最大 8 行）。ゲート（`#if`・環境変数）は無い
- 影響範囲: 非 RT スレッドのみ。音声スレッドへの直接の波及は無い。ただし読み込みが完了しないため、結果として IR が適用されない
- 既知事項との関係: Part 8 は「約 16 行/ロード（Low）」としていた。行数の測定値は一致するが、**ブロックによる読み込み失敗は今回の追加実測で初めて確認した**
- 修正の必要性: 中。通常のコンソール・ファイルリダイレクトでは問題ないが、ログ収集のために stderr をパイプする起動形態では再現する
- 修正案: `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` 等のコンパイル時ゲートへ移す、または `juce::Logger` 経由でログレベルを付ける
- リスク: 既定で診断が見えなくなるため、調査手順の文書化が必要
- 未検証: 実ホスト（DAW・ランチャー）での stderr 取り扱い。今回は Python の `subprocess.PIPE` で再現した

### B. Minimum / Mixed Phase の最小位相変換スキップ

- 判定: **Not Applicable（現行コードでは到達不能）**
- 関連ファイル: `src/convolver/ConvolverProcessor.ResampleAndFallback.cpp`、`ConvolverProcessor.LoaderThread.cpp`、`src/ConvolverProcessor.h`、`ConvolverProcessor.StateAndUI.cpp`
- 呼び出し経路: `doTransformStep` → `convertToMinimumPhase`（呼び出しは 1 箇所のみ）
- Production 到達性: **なし**
- 根拠: `fftSize = nextPowerOfTwo(numSamples * 4)` が `MAX_MINPHASE_FFT_SIZE = 8388608`（2^23）を超えると空を返す。一方 `computeTargetIRLength` が `MAX_IR_LATENCY = 2097152`（2^21）でトリム長を制限する。`numSamples = 2^21` のとき `numSamples * 4 = 2^23` で、判定は `>` のため偽
- 数値確認: スキップには `numSamples > 2^21` が必要で、トリム上限と矛盾する
- 影響範囲: なし（現行）。将来 `MAX_IR_LATENCY` を増やすと復活する
- 既知事項との関係: **Part 8 の「Confirmed（無言スキップ）」と前回報告の「妥当」は誤り**。現行ソースでは到達不能
- 修正の必要性: なし（防御コードとして維持）。到達不能である旨のコメント追加は有益
- 未検証: 2^21+1 サンプルの実ロードはトリムで切られるため構成できない（静的証明のみ）

### C. 群遅延 120 サンプルのクランプ

- 判定: **Confirmed Issue / Non-Bug（Low、文書化ギャップ）**
- 関連ファイル: `src/convolver/ConvolverProcessor.MixedPhase.cpp`（`maxAllowedGD = 120.0`、322 行。クランプ 417 行）、`manual/IR_ADVANCED_WINDOW_EN.md`、`manual/MAIN_WINDOW_EN.md`
- Production 到達性: あり（Mixed 選択時）。ただし**クランプ自体は実測 5 例で発動しなかった**
- 根拠（ソース）: 目標群遅延は `-dPhi - peakDelay` を 5 ビン移動平均し、最小値が負なら `-min + 5` を加算して正規化したうえで `clamp(gd, 0.0, 120.0)` する。したがって 120 に達するには、正規化後のビン間の広がりが 115 サンプルを超える必要がある
- 実行時確認: `--buzz`（Mixed、遷移 200〜1000 Hz、目標長 0.5 s）で 5 種類の IR を投入。クランプ前の範囲は −804 / −7,109 / −11,299 / −172,800 / −2,554 サンプルだったが、**クランプ後はすべて「5 to 5」**で、制限は掛からなかった
- 解釈: 120 は常用の制限ではなく防御的上限として働いている。マニュアル・UI に記載が無いのは事実だが、ユーザーが遭遇する可能性は低い
- 修正の必要性: 低（情報開示）。記載するなら「目標群遅延の上限は 120 サンプル（防御的上限）」と正確に書く
- 未検証: クランプが発動する IR の構成方法。今回は 5 例で発動しなかった

### D. README のソース件数不一致

- 判定: **Confirmed Issue / Non-Bug（Trivial）**
- 根拠: README は 2 箇所で「385 files」。`git ls-files 'src/*'` は **386 件**、うち `src/audioengine/` は **128 件**
- 修正案: 385 → 386
- 未検証: テスト実行ファイル数「53」の再計測は未実施

### E. ポリフェーズ処理の −2.50 dB/段の損失

- 判定: **Confirmed Bug（Medium、既知・HOLD）**
- 関連ファイル: `src/CustomInputOversampler.cpp`（549〜552 行）、`CMakeLists.txt:134`（既定 OFF）
- 発生条件: OS 係数 2×/4×/8× のいずれか
- 根拠（ソース）: 畳み込み枝は `convValue *= 2.0` を無条件に実行するが、中心枝の `centerValue *= 2.0` は `#if CONVOPEQ_CORRECT_POLYPHASE_GAIN` の中にしかない。マクロは既定 OFF。エンジン側に補償ゲインは無い
- 実測（OFF ビルド）: DC 利得 0.750000000 / 0.562500000 / 0.421875000（2×/4×/8×）。帯域 0.05〜0.40×Fs も同値で平坦
- 実測（ON ビルド）: DC 利得 **1.000000000**（全構成）。帯域は 0.9993〜1.0009
- 差分: ON/OFF の DC 比は 1/0.75 = 1.3333（2×）、1/0.5625 = 1.7778（4×）、1/0.421875 = 2.3704（8×）
- 影響範囲: 既定 Auto（48 kHz 以下で 8×）では −7.50 dB。ユーザーがメイクアップで補えるが、レベル規約としては不整合
- HOLD 判断の評価: 損失は実在し、平坦で、`PolyphaseGainFidelityTests` は 0.75 を期待値として固定している。すなわち「意図した音作り」ではなく「未修正の利得規約」である。HOLD は製品判断として成立するが、修正しない限り既定 8× で −7.50 dB が残る
- 修正案: マクロを既定 ON にする、または `decimateStage` に段数分の補償を入れる
- リスク: 既存プロジェクトのレベルが +2.5〜+7.5 dB 変化する。プリセット互換の告知が必要
- 未検証: ON ビルドの音質評価、既存プロジェクトの互換影響

### F. Bypass 時の IR 切断（N1）

- 判定: **Confirmed Bug（Medium、既知）**
- 関連ファイル: `src/MKLNonUniformConvolver.cpp` / `.h`、`src/ConvolverControlPanel.cpp`（Tail Mode に Bypass = item id 3）
- 呼び出し経路: UI で Bypass を選択 → `tailMode = 2` → `tailEnabled = false` → L0 のみ
- 根拠（ソース）: `computeL0MaxParts` は tail 無効時に `kL0PartsFloor = 32` を返し、`l1Len`/`l2Len` は 0 になる
- 実測（層構成・既存 `MTNUPCMeasurement`）: `M4 768k/bs512 bypass: L0 partSize=512 numPartsIR=32 (want 32) -> PASS`、B13 グリッド 8 構成で `[NG]=0`
- 実測（波形・リポジトリ外の測定アプリ）: 48 kHz、IR 1.0 s（末尾 100 ms のエネルギー −11.9 dB）

| block | Bypass の保持長 | 秒換算 | 保持長以降の出力 | 保持長までの一致 |
|---|---|---|---|---|
| 256 | 8,192 サンプル | 0.1707 s | −309.5 dB（peak 6.9e-17） | 3.3e-16 |
| 512 | 16,384 サンプル | 0.3413 s | −313.4 dB（peak 3.1e-17） | 3.9e-16 |
| 1024 | 32,768 サンプル | 0.6827 s | −330.8 dB（peak 3.5e-18） | 3.3e-16 |

- 同じ測定で tailMode=1（Layer Tail Contouring）は、L0 が 0.12〜0.13 秒でも L1 が残りを担い、末尾区間のエネルギーが −11.9 dB の IR に対して 11.7〜22.6 dB を出力した（テール整形のため厳密一致ではない）
- 影響範囲: Bypass を選んだユーザーに無言のテール欠損。既定 `tailMode=1` では Bypass 固有の切断は発生しない
- **追加発見（H）**: ただし block ≥ 1024（48 kHz）では、既定 `tailMode=1` でもテール層のパート長（8 × block）が L0 保持長（0.12 秒 = 5,760 サンプル）を超え、末尾が欠落する。実測では末尾 896 サンプルが無音になり、テールエネルギーが 22.28 dB → 11.71 dB に落ちた（block 1024、IR 1.0 秒）。F の幾何修正で同時に解消できることを試作で確認した
- クロスフェードとの関係: PR #168 の 20 ms クロスフェードは遷移のクリック対策であり、保持長を変えない
- 修正案: tail 無効時も L1/L2 を確保する、または Bypass の意味を UI/マニュアルで「L0 のみ（IR は 0.17〜0.68 秒で打ち切り）」と明示する
- リスク: 層を確保すると Bypass のメモリ・CPU が増える。設計意図が「軽量 Bypass」なら文書化が妥当

### G. ISR Verification の HEAD failure

- 判定: **Confirmed Issue / Non-Bug（CI 構成、Medium・プロセス）**
- 関連: `.github/workflows/isr-verification.yml`、`.github/scripts/isr-verify-clang-tidy-audit.ps1`
- 失敗条件: exhaustive tier（週次 `0 19 * * 0`）で監査が enforce され、`build/compile_commands.json` が無い
- 根拠（CI ログ）: 失敗 run `42513179942` の `[ERROR] compile_commands.json missing: D:\a\ConvoPeq\ConvoPeq\build\compile_commands.json`
- 根拠（構成）: 監査スクリプトは `$repoRoot/build/compile_commands.json` を固定パスで要求する。`CMakeLists.txt:35` は `CMAKE_EXPORT_COMPILE_COMMANDS ON` を設定しているが、ISR Verification のワークフローに CMake の configure/build ステップが無い
- ローカル確認（DB あり）: `build/compile_commands.json`（5,163,809 B）が存在する状態で `-Tier exhaustive -RequireClangTidy` を実行すると **`[PASS] clang-tidy audit gate verified`（exit 0）**
- ローカル再現（DB なし）: 空の `build/` を持つ擬似ツリーに `.github` のスクリプト・registry・policy だけをコピーして同じオプションで実行すると、CI と同一の `[ERROR] compile_commands.json missing: ...` と `clang-tidy audit gate failed. count=1`（exit 1）になった
- 追加確認（クリーン configure のみ）: リポジトリ外の新しいビルドディレクトリで `cmake -S . -B <dir> -G "Ninja Multi-Config" -DCONVOPEQ_REQUIRE_MKL=ON` を実行すると DB は生成される（5,368,562 B）が、監査は `[ERROR] clang-tidy invocation failed (exit=1) target=.../src/audioengine/PublicationAdmission.cpp` で失敗する。原因は `JuceHeader.h` が未生成で、`AudioEngine.h:41` の `#include <JuceHeader.h>` が解決できないこと
- 追加確認（ヘッダ生成後）: `ninja -C <dir> -f build-Release.ninja ConvoPeq_artefacts/JuceLibraryCode/JuceHeader.h` で生成ヘッダだけを作ると、同じ監査が **PASS** する
- したがって必要なのは configure だけでなく、生成ヘッダの用意までである
- 実行履歴: 同一 HEAD `9bb8515c` で standard tier は success、exhaustive tier は failure。standard は監査が monitor 扱いで skip されるため
- コード起因か: **環境・構成起因**。ソースの警告ではない
- 修正案: 監査の前に (1) `cmake -S . -B build -G "Ninja Multi-Config" -DCONVOPEQ_REQUIRE_MKL=ON` で configure し、(2) 生成ヘッダ `ConvoPeq_artefacts/JuceLibraryCode/JuceHeader.h` を作り、(3) DB の存在を確認してから監査を実行する。ヘッダ生成は `ninja` のファイル指定か、専用のカスタムターゲットで行う
- リスク: configure には oneAPI（MKL+IPP）が必要。CI 時間は oneAPI の復元（キャッシュあり）と configure で数分増える
- 未検証: 修正後の CI 動作（未実装）

## §4 実行時検証結果

### 4.1 ポリフェーズ損失（E・OFF ビルド）

- 実行: `build\Release\PolyphaseGainMetrics.exe`（既存ターゲット）
- 結果: `[METRIC-BUILD] flag=OFF`、`IIR3_r2 dc=0.750000000` / `r4 0.562500000` / `r8 0.421875000`、`LP3_r2 0.750000000` / `r8 0.421875000`
- 帯域: g005/g010/g020/g040 はいずれも DC と同値（例: r8 で 0.4218〜0.4219）
- 判定: 理論値 0.75^段数と一致。損失は平坦な定数

### 4.2 ポリフェーズ損失（E・ON ビルド）

- 実行: `build-polyon\Release\PolyphaseGainMetrics.exe`（`-DCONVOPEQ_CORRECT_POLYPHASE_GAIN=ON` で構成した別ツリー、既存ターゲットのみ）
- 結果: `[METRIC-BUILD] flag=ON`、DC 利得は 5 構成すべて **1.000000000**
- 帯域: `IIR3_r2` g005=1.000883715 / g010=1.000000004 / g020=1.000000012 / g040=1.000000035、`LP3_r8` g005=1.000513720 / g010=0.999303902 / g020=1.000075122 / g040=0.999557255
- 遅延: r2=255、r4=286、r8=290、LP3_r2=501、LP3_r8=461（argmax による測定。OFF では 255 / 287 / 290 / 501 / 475）
- 判定: フラグ ON で利得規約が復元することを実測で確認。遅延値の小さな差は分岐の重みが変わることによる argmax の移動で、処理経路の変更ではない
- 参考: rmsRatio は OFF 0.7401 → ON 0.9875（r2）、peakRatio は OFF 0.7858 → ON 1.0348。いずれも約 1/0.75 倍
- 参考（production chain の測定は不可）: `AudioEngineHarness.exe --p1-char` は OFF ビルドで `summary flag_macro=0 cases=1 failures=1` となり、OS=1 の 1 ケースだけが走って中断した。production chain の利得特性評価には使えなかった

### 4.3 Bypass の IR 切断（F）

- 実行 1（既存）: `build\MTNUPCMeasurement_artefacts\Release\MTNUPCMeasurement.exe` → `M4 768k/bs512 bypass: L0 partSize=512 numPartsIR=32 (want 32) -> PASS`、`M4 B13 grid: 8 configs, [NG]=0 -> PASS`、`EXIT: structural failures = 0`
- 実行 2（リポジトリ外の測定アプリ `TailTrunc`）: production の `convo::MKLNonUniformConvolver` をそのままリンクし、48 kHz / IR 1.0 s のインパルス応答を描画して IR と直接比較
- 結果: 表（§3 F）のとおり。Bypass は保持長まで誤差 3e-16 で一致し、以降は無音。tailMode=1 は末尾までエネルギーが存在
- 判定: 切断は波形レベルで確認（理論計算ではなく実測）

### 4.4 stderr 出力（A）

- 実行 1（ファイルリダイレクト）: `AudioEngineHarness.exe --n1-analysis <dir> post` → exit 0、stderr 29,940 B / 12 ロード = **2,495 B/ロード**、`[N1-analysis] ALL PASS`
- 実行 2（DEVNULL）: exit 0、1.6 秒
- 実行 3（未ドレインのパイプ）: 90 秒経過時点で**未終了**。読み出し開始後に 4,719 B を読んだ時点で `[N1-analysis] IR load failed: len=0 finalized=0 peak=0 err=''`、exit 2
- 内訳（実行 1）: `[IR_TAIL_ENV]` 84、`[IR_CHAIN]` 72、`[IR_RATE_GEN]` 12、`[IR_TAIL_ENERGY]` 12、`[IR_TAIL_GEOM]` 12
- 判定: 出力量は 16.0 行/ロード。未ドレイン時はロードが完了しない

### 4.5 群遅延クランプ（C）

- 実行: `AudioEngineHarness.exe --buzz --buzz-quick --buzz-dur=0.2 --buzz-quiet=5000 --buzz-ir=<IR>`（Mixed、遷移 200〜1000 Hz、目標長 0.5 s）
- 投入した IR: N1 の減衰 IR、40 Hz 減衰（τ0.5 s）、25 Hz 減衰（τ1.0 s）、末尾反射付き、低域共鳴＋高域成分
- 結果: クランプ前の範囲は −804 / −7,109 / −11,299 / −172,800 / −2,554 サンプル。**クランプ後はすべて「5 to 5」**で、120 の制限は掛からなかった
- 判定: 120 は防御的上限として動作しており、通常の IR では発動しない
- 参考（範囲外の観測）: 短縮実行（`--buzz-dur=0.2`）では終了時に `[FAULT] ~AudioEngine: coordinator in Faulted state after markShutdownComplete` が出て exit 1 になった。本調査の対象外であり、原因は追跡していない

### 4.6 ISR Verification（G）

- 実行 1（DB あり）: `pwsh -File .github\scripts\isr-verify-clang-tidy-audit.ps1 -Tier exhaustive -RequireClangTidy`
- 前提: `build/compile_commands.json` が存在（5,163,809 B）
- 結果: `[PASS] clang-tidy audit gate verified`、exit 0、レポートは `evidence/clang_tidy_audit_report.json`（`evidence/` は `.gitignore` 対象）
- 実行 2（DB なし・リポジトリ外の擬似ツリー）: `[ERROR] compile_commands.json missing: <sim>\build\compile_commands.json`、`clang-tidy audit gate failed. count=1`、exit 1。CI の失敗と同じ挙動
- 実行 3（クリーン configure のみ）: DB は生成されるが `[ERROR] clang-tidy invocation failed (exit=1) target=...PublicationAdmission.cpp`、`clang-diagnostic-error: 'JuceHeader.h' file not found`（`AudioEngine.h:41`）
- 実行 4（生成ヘッダを作成後）: 同じ構成で `[PASS] clang-tidy audit gate verified`（exit 0）
- 判定: 成否を決めるのは DB の有無だけではなく、生成ヘッダ（`JuceHeader.h`）の有無も必要である。CI の失敗は configure とヘッダ生成のステップが無いことによる

### 4.7 最小位相スキップ（B）

- 実行: なし（トリム上限 2^21 により構成不能）
- 判定: 静的証明（`numSamples ≤ 2^21` → `fftSize ≤ 2^23` → スキップ条件は偽）
- 状態: 未実測。到達不能の根拠はソースと数値のみ

## §5 修正提案

| 優先 | 対象 | 最小修正 | 代替 | 副作用 | 必要な回帰テスト |
|---|---|---|---|---|---|
| P1 | E ポリフェーズ | `CONVOPEQ_CORRECT_POLYPHASE_GAIN` を既定 ON | decimate 側に段数分の補償 | 既存プロジェクトのレベルが +2.5〜+7.5 dB 変化 | `PolyphaseGainFidelityTests` の期待値更新、ON/OFF の DC・周波数応答 |
| P1 | F バイパス切断 | tail 無効時も L1/L2 を確保 | Bypass の意味を UI/マニュアルで明示 | メモリ・CPU 増、または仕様変更の告知 | 長 IR の末尾波形、Bypass 切替のクリック |
| P2 | A stderr | コンパイル時ゲートまたはログレベルへ移す | 起動時に stderr を捨てる | 既定で診断が見えなくなる | ロード成否、`--n1-analysis` の ALL PASS |
| P2 | G CI | 監査前に configure（`CMAKE_EXPORT_COMPILE_COMMANDS=ON`） | スクリプト内で最小 DB 生成 | CI 時間増 | 週次 exhaustive の success |
| P3 | C 群遅延 | マニュアルと UI に「目標群遅延の上限 120 サンプル（防御的上限）」と明記 | ログのみ現状維持 | なし | 文書のみ |
| P3 | D README | 385 → 386 | — | なし | なし |
| P3 | `[IR_RATE_GEN]` | `requestGenerationId` を表示 | — | なし | なし |

`type only` の残留文言（`ISRShutdown.h:299`、`:365`）は動作に影響しないが、現行実装（Permit は production で使用）と矛盾するため、次回の文書整理で直すのが望ましい。

## §6 未検証事項

- A: 実ホスト（DAW・ランチャー）での stderr 取り扱い。Python の `subprocess.PIPE` による再現のみ
- B: 2^21 超 IR の実ロード（トリム上限のため構成不能）
- C: 120 サンプルのクランプが発動する IR の構成方法。5 例では未発動
- E: ON ビルドの音質評価、既存プロジェクトの互換影響。production chain の ON/OFF 比較は `--p1-char` が OFF ビルドで `failures=1` となり実施できなかった
- F: Bypass 切替時のクリック、長 IR でのレイテンシ・CPU 変化
- G: 修正後の CI 動作
- 全体: 実レイテンシ、負荷時の XRUN、ホスト適合

これらは「未検証」であり、問題なしの証明ではない。

## §7 結論

- 追加実測で **A を Confirmed Bug（Medium）へ格上げ**した。行数（16.0 行/ロード）は Part 8 の記載と一致するが、stderr を読まないパイプでは LoaderThread が書き込みで停止し、読み込みが失敗する（exit 2）ことを再現した。診断出力の量ではなく、ブロックによる機能停止が問題である。
- **E は補正フラグ ON で DC 利得 1.000000000 に戻る**ことを実測した。損失は OFF 既定の帰結であり、修正の方向性は実証済みである。HOLD は製品判断として成立するが、利得規約の不整合は残る。
- **F は波形レベルで切断を確認**した。Bypass の保持長は block 256/512/1024 で 0.1707 / 0.3413 / 0.6827 秒、以降は 3e-17 以下の無音である。Bypass 固有の切断は既定 `tailMode=1` では発生しないが、**block ≥ 1024 では既定モードにも別のテール欠落（H）がある**（テール 22.28 → 11.71 dB）。
- **C の 120 サンプルクランプは 5 種類の IR で発動しなかった**。防御的上限として妥当であり、文書化ギャップの重要度は前回想定より低い。
- **B は現行コードでは到達不能**（Part 8 の判定を訂正）。**G は compile DB の欠落による構成問題**で、ローカルでは DB を用意すると PASS する。
- 修正優先度は E → F → A → G。いずれも本調査では実装していない。

## §8 再現手順

```
# 環境
git rev-parse HEAD                      # 2aecd73e...
cmake --build build --config Release --target PolyphaseGainMetrics
cmake --build build --config Release --target MTNUPCMeasurement

# E: ポリフェーズ損失（OFF）
build\Release\PolyphaseGainMetrics.exe

# E: 補正 ON（別ツリー）
cmake -S . -B build-polyon -DCONVOPEQ_REQUIRE_MKL=ON -DCONVOPEQ_CORRECT_POLYPHASE_GAIN=ON
cmake --build build-polyon --config Release --target PolyphaseGainMetrics
build-polyon\Release\PolyphaseGainMetrics.exe

# F: 層構成（Bypass 32 分割）
build\MTNUPCMeasurement_artefacts\Release\MTNUPCMeasurement.exe

# F: 波形（リポジトリ外の測定アプリ。production の MKLNonUniformConvolver をリンク）
#    C:\Users\user\AppData\Local\Temp\opencode\b0612\tailtrunc\CMakeLists.txt
cmake -S <tailtrunc> -B <tailtrunc>\build -G "Ninja Multi-Config"
cmake --build <tailtrunc>\build --config Release
<TailTrunc.exe>

# A: 診断出力と未ドレイン時の停止
build\Release\AudioEngineHarness.exe --n1-analysis <出力先> post 2> stderr.log
#    （未ドレイン再現は b0612\stderr_block_test2.py）

# C: 群遅延クランプ（Mixed）
build\Release\AudioEngineHarness.exe --buzz --buzz-quick --buzz-dur=0.2 --buzz-quiet=5000 --buzz-ir=<IR.wav>

# G: 監査ゲート（compile DB がある状態）
pwsh -File .github\scripts\isr-verify-clang-tidy-audit.ps1 -Tier exhaustive -RequireClangTidy

# B: 到達不能の根拠
grep -n 'MAX_IR_LATENCY' src/ConvolverProcessor.h
grep -n 'computeTargetIRLength' -A 26 src/convolver/ConvolverProcessor.StateAndUI.cpp
grep -n 'MAX_MINPHASE_FFT_SIZE' src/convolver/ConvolverProcessor.ResampleAndFallback.cpp
```

測定用スクリプト・アプリ・ログは `C:\Users\user\AppData\Local\Temp\opencode\b0612\` に置いた（リポジトリ外）。

