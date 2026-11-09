# 外部監査 Part 8（追加検証）の実態確認

- 実施: 2026-11-09 01:20
- 対象: `main` = `9bb8515c`（v0.6.12）
- 方法: 現行ソースの静的読解、`git ls-files` による件数確認、GitHub API による CI 実行結果の取得
- コード変更: なし（本報告のみ）

## 結論

添付レポート Part 8 の「新規の本番到達可能な Confirmed 欠陥は発見されなかった」という結論は、現行ソースで裏付けられた。指摘のうち Confirmed とされたものは、いずれも Low（非 RT の診断出力、無言スキップ、開示ギャップ、文献の件数）である。影響度 Medium の 2 件（バイパス時の IR 切断 N1、ポリフェーズの利得損失）は、いずれも既知で開発側が HOLD と決めている事項であり、Part 8 が新たに見つけた欠陥ではない。

Part 8 が「原因未調査」とした CI の HEAD failure は、`build/compile_commands.json` 不在によるもので、コード起因ではなかった。同じ HEAD の別のスケジュール実行は success しており、実行環境の状態差で再現性が変わる。

| # | Part 8 の指摘 | 判定 | 根拠の要約 | 影響 |
|---|---|---|---|---|
| 1 | LoaderThread の無条件 stderr（約 16 行／ロード） | **妥当** | `fprintf(stderr)` 7 箇所、チェックポイント 5 回 | Low（NonRT） |
| 2 | 2^21 超 IR で最小位相が無言スキップ | **妥当** | `MAX_MINPHASE_FFT_SIZE=8388608`、空を返し原 IR のまま成功 | Low |
| 3 | 群遅延 120 サンプルクランプが未開示 | **妥当** | `maxAllowedGD=120.0`、manual / UI に記載なし | Low（情報） |
| 4 | README の src 385 件（実数 386） | **妥当** | `git ls-files 'src/*'` = 386、audioengine = 128 | Trivial |
| 5 | ポリフェーズ −2.50 dB／段（既定 8×） | **妥当（既知・HOLD）** | 中心枝の ×2 が `#if` 内のみ | Medium |
| 6 | Bypass 時の IR 切断（0.186〜0.341 s） | **妥当（既知）** | tailMode=2 で L0 32 分割・L1/L2=0 | Medium |
| 7 | `[IR_RATE_GEN]` が要求世代でなく現在世代を表示 | **妥当** | `getCurrentGeneration()` を出力 | Trivial |
| 8 | 競合時破棄が世代検査より先 | **妥当（設計前提）** | CAS 失敗時に無条件破棄 | Low |
| 9 | リビルド起源コミットが現在世代を刻む | **妥当（意図的）** | `Rebuild.cpp` の `getCurrentGeneration()` 刻印 | Low |
| 10 | TruePeak のサニタイズ容量超で検査スキップ | **妥当（限定的）** | `sanitizedCapacity >= numSamples` 条件 | Low |
| 11 | リミッタ後段が NaN-safe でない | **妥当（本番到達不能）** | double 末尾は `jlimit`、AVX2 は NaN→−0.8913 | Low |
| 12 | EQ mono＋Mid の nullptr コピー | **妥当（本番到達不能）** | 内部は常時 2ch | Low |
| 13 | Sanitizer CI が HEAD 未実行 / ISR Verification failure | **前者は妥当、後者は原因判明** | 最新 ASan は success、ISR は compile_commands.json 不在 | 環境 |
| 14 | 旧 Part の「型のみ・未接続」等 | **解消済み** | コメント整合は PR #201 で実施 | — |

## 1. 個別検証

### 1.1 LoaderThread の診断出力

`src/convolver/ConvolverProcessor.LoaderThread.cpp` に `fprintf(stderr, ...)` が 7 箇所ある。`printChainCheckpoint` は `A_resampled`・`B_dcblock`・`C_tukey`・`D_trimfade`・`E_phase` の 5 回呼ばれ、本体に 2 つの `fprintf` を持つ。これに `[IR_RATE_GEN]`、`[IR_TAIL_GEOM]`、`[IR_TAIL_ENV]`（最大 8 行）、`[IR_TAIL_ENERGY]`、`F_scale` が加わるため、1 回のロードで最大 17 行程度になる。ゲート（`#if`・環境変数）は無い。

いずれも LoaderThread（非 RT）で、音声スレッドには到達しない。Part 8 の「Low（NonRT）」は妥当である。

### 1.2 最小位相の無言スキップ

`convertToMinimumPhase`（`ConvolverProcessor.ResampleAndFallback.cpp`）は `fftSize = nextPowerOfTwo(numSamples * 4)` を計算し、`MAX_MINPHASE_FFT_SIZE = 8388608`（2^23）を超えるとログを出して空のバッファを返す。したがって `numSamples > 2^21`（48 kHz で約 43.7 秒）の IR は変換されない。

呼出側（`LoaderThread.cpp` の `doTransformStep`）は `validateBuffer(minPhaseIR)` が偽なら何もせず、`stepTrimmed` は元の IR のまま `return true` する。エラーメッセージも UI 通知も無く、モード表示は Minimum / Mixed のまま残る。Part 8 の「原 IR のまま無言成功」は正しい。

### 1.3 群遅延クランプの未開示

`ConvolverProcessor.MixedPhase.cpp` に `constexpr double maxAllowedGD = 120.0` があり、位相勾配ガード（`maxAllowedGD * dOmega`）と最終クランプ `std::clamp(gd, 0.0, maxAllowedGD)` に使われる。サンプル値でありサンプルレートに依存しない。

`manual/` 配下を検索すると「120」はディザ学習の「約 120 秒」のみで、群遅延上限の記載は無い。`ConvolverControlPanel.cpp` と `MixedPhaseOptimizationComponent.cpp` にも該当する文言・ツールチップは無い。Part 8 の「文書化ギャップ」は妥当である。

### 1.4 README のファイル数

README は 2 箇所で「385 files」と書く。`git ls-files 'src/*'` は 386 件、うち `src/audioengine/` は 128 件である。Part 8 の「1 件過少」は正しい（Trivial）。

### 1.5 ポリフェーズの利得

`CustomInputOversampler.cpp` は補間時に畳み込み枝を無条件で `convValue *= 2.0` する一方、中心枝の `centerValue *= 2.0` は `#if CONVOPEQ_CORRECT_POLYPHASE_GAIN` の中にしかない。マクロは既定 OFF（`CMakeLists.txt` の `option(... OFF)`）であるため、1 段あたり 0.75（−2.4988 dB）の損失が残る。3 段（8×）では −7.4963 dB になる。

Part 8 は「既知・未修正」「テストは現行規約を契約として固定している」とする。実装と一致する。既定の Auto は 48 kHz 以下で 8×（3 段）を選ぶため、通常の 48 kHz 運用ではこの損失が既定で発生する。開発側が HOLD と決めている事項であり、本検証では新規欠陥とは扱わない。

### 1.6 バイパス時の IR 切断（N1）

`MKLNonUniformConvolver.cpp` は `tailMode == 2`（Bypass）で `tailEnabled = false` とし、`computeL0MaxParts` は tail 無効時に `kL0PartsFloor = 32` を返す。`l1Len` と `l2Len` は 0 になる。したがって保持長は `32 × l0Part` で決まる。

| 条件 | l0Part | 保持サンプル | 保持時間 |
|---|---|---|---|
| 44.1 kHz / 256 | 256 | 8,192 | 0.1858 s |
| 48 kHz / 256 | 256 | 8,192 | 0.1707 s |
| 48 kHz / 512 | 512 | 16,384 | 0.3413 s |
| 44.1 kHz / 512 | 512 | 16,384 | 0.3715 s |
| 96 kHz / 1024 | 1024 | 32,768 | 0.3413 s |

既定の `tailMode = 1`（Layer Tail Contouring）では L0 が tailStart 基準で拡張され、残りを L1/L2 が担うため切断は起きない。切断が起きるのは Bypass を明示選択した場合に限られる。

PR #168 の 20 ms クロスフェードはバイパス遷移のクリック対策であり、IR の保持長を変えない。Part 8 の指摘は正しい。

### 1.7 その他の確認

- `[IR_RATE_GEN]` は `owner.convolverStateGeneration.getCurrentGeneration()` を出力しており、要求世代（`requestGenerationId`）ではない。診断表示のみの不整合。
- `updateConvolverState` は `writerActive` の CAS に失敗すると世代を検査せず破棄する。メッセージスレッド専用という前提の上での設計であり、Part 8 の「設計前提依存」は妥当。
- リビルド経路（`Rebuild.cpp`）は finalize 時点の `getCurrentGeneration()` を刻むため、`executePendingCommit` の鮮度検査は実質空転する。通貨性は `shouldCancel` が担う設計で、コメントにも明記されている。
- `TruePeakDetector.cpp` は `sanitizedCapacity >= numSamples` のときだけサニタイズする。prepare 時の maxBlockSize を超えるブロックでは検査が省かれる。作者が明示している残存事項である。
- 出力段のハードクランプは、double 経路の AVX2 部が `_mm256_min_pd(_mm256_max_pd(v, -lim), lim)` で NaN を −0.8913 に変換し、末尾のスカラ部は `juce::jlimit` で NaN を素通しする。ただしリミッタ前の scrub とリミッタ算術の有界性により、本番で NaN がデバイスへ届く経路は無い。
- EQ の mono＋Mid 分岐は `dataR` が nullptr でも `FloatVectorOperations::copy(dataR, dataL, n)` を呼ぶ。ただし `AudioEngine` は常に 2 チャンネルで EQ を呼ぶため、出荷アプリからは到達しない。

## 2. CI の確認

Sanitizer CI は、マージ後に実行された `4641fe3b` を含めて success している（`42496269441` など）。Part 8 の「HEAD 9bb8515 に sanitizer run が無い」は事実だが、HEAD は文書のみのマージであり、コードは `4641fe3b` と同一である。

ISR Verification は HEAD で 2 回のスケジュール実行があり、`42507791714` が success、`42513179942` が failure だった。失敗ログを取得すると、原因は次の 1 行だった。

```
[ERROR] compile_commands.json missing: D:\a\ConvoPeq\ConvoPeq\build\compile_commands.json
```

clang-tidy 監査ゲートが `compile_commands.json` を要求するが、スケジュール実行ではビルド構成が作られていなかった。コードの欠陥ではなく、CI の実行状態に依存する失敗である。Part 8 が「原因未調査」とした点は、これで説明できる。

## 3. 旧 Part との整合

Part 1〜5 が根拠にした「型のみ・production 未接続」「Phase-4 frozen world concept」「B-6 実装待ち」は、現行コードでは解消済みである。

- `ISRLifetimeProof.h` は production 接続済みと明記され、呼出チェーン（`~AudioEngine` → `releaseResources` → `tryShutdownQuiescentReclaim`）が実在する。
- `FrozenRuntimeWorld` は publish 経路で構築され、release / seal / commit の順序が契約化されている。
- `LoaderThread.cpp` の B-6 コメントは、ロードループ内に世代チェックを置かず commit 直前ガードで正当性を担保する現行設計に更新されている。

ただし、微細な残留がある。`ISRShutdown.h:299` の「type only では未使用」と `:365` の「Phase A1 — type only」は、現在の実装（Permit は production で使用）と合わない。動作への影響は無いが、次に読む人を誤らせる。

## 4. 推奨対応

本検証ではコードを変更していない。優先度順の提案は次のとおり。

| 優先 | 対象 | 内容 |
|---|---|---|
| P2 | `ConvolverProcessor.LoaderThread.cpp` | 診断出力を `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` 等でゲートする |
| P2 | `convertToMinimumPhase` のスキップ | ログだけでなく、UI 状態またはロード結果に「未変換」を残す |
| P2 | 群遅延 120 サンプル | マニュアルと該当 UI に制約を明記する |
| P3 | `[IR_RATE_GEN]` | `requestGenerationId` を表示する |
| P3 | README | 385 → 386 に修正する |
| P3 | `ISRShutdown.h` の 2 箇所 | 「type only」の残留文言を現行に合わせる |
| 判断待ち | N1（Bypass 切断）、ポリフェーズ | いずれも既知の HOLD 事項。仕様判断が必要 |

## 5. 検証コマンド

```
grep -n 'fprintf(stderr' src/convolver/ConvolverProcessor.LoaderThread.cpp
grep -n 'MAX_MINPHASE_FFT_SIZE' src/convolver/ConvolverProcessor.ResampleAndFallback.cpp
grep -n 'maxAllowedGD' src/convolver/ConvolverProcessor.MixedPhase.cpp
git ls-files 'src/*' | wc -l
grep -n 'CONVOPEQ_CORRECT_POLYPHASE_GAIN' CMakeLists.txt | head
grep -n 'kL0PartsFloor\|computeL0MaxParts' src/MKLNonUniformConvolver.h
gh run view 42513179942 --log | Select-String '\[ERROR\]'
```

## 6. 限界

静的読解と API 取得のみで、実行時挙動は確認していない。ポリフェーズの損失量はコード上の解析値（0.75 の段数乗）であり、実測していない。N1 の切断も幾何計算による値で、波形での確認は行っていない。Part 8 が挙げた実行テスト（ポリフェーズ実測、N1 波形確認、リミッタ +Inf、破損 state、最小位相境界、FTZ 網羅）は未実施である。
