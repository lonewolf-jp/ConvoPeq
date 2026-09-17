# WORK102-IMPL — Bounded IR Load Admission + Streaming Hash 実装報告

- **作成日**: 2026-09-17
- **種別**: Implementation（R2 + IG で凍結した契約の production 実装）
- **前工程**: `big18_implementation_gate_20260917.md`（IMPLEMENTATION-READY）
- **契約の正本**: `big18_failure_contract_arbitration_20260917.md`（R2 凍結）
- **検証台帳**: `evidence/WORK102_IMPL_VERIFICATION.txt`

```text
判定: IMPLEMENTED
```

**契約値の変更 0 / scope drift 0 / テスト失敗 0。**
commit / push は未実施（指示どおり Final Gate 待ち）。

---

## 1. Baseline と変更面

```text
HEAD / origin/main = 35461e2e
ConvoPeq.md        = 再生成 2026-09-17 09:13:37 → FRESH（NEWER_SRC_COUNT=0）
```

| ファイル | 差分 | 内容 |
|---|---|---|
| `src/convolver/IRLoadAdmission.h` | **新規** | 契約定数 4 値 / 純 predicate / 診断の単一情報源 |
| `src/convolver/ConvolverProcessor.LoaderThread.cpp` | +61 | admission 順序 / hash 呼出移動 / FC-FORM-5 |
| `src/AllpassDesigner.cpp` | +186 −72 | `Xxh64Stream` 導入 / `computeIRHash` の O(1) 化 |
| `src/tests/AudioEngineHarness/IRLoadAdmissionTests.cpp` | **新規** | A〜L |
| `src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp` | +8 | harness への登録のみ |
| `CMakeLists.txt` | +1 | 既存 target への TU 追加（新規 CTest target なし） |
| `ConvoPeq.md` | 再生成 | 上記を反映 |

**RT / authority 面への波及は 0**: atomic 追加 0 行、`MAX_IR_LATENCY` 差分 0 件、
Runtime / DSPCore / RuntimeWorld / ISRShutdown / ISRRetire / Epoch / Crossfade /
LifetimeBudget / Deferred* の変更 0 件。`doc/Practical Stable ISR Bridge Runtime.md` の
「危険操作は NonRT へ隔離」原則に対し、本変更は `LoaderThread`（NonRT）と
`AllpassDesigner`（同 Thread から呼ばれる純計算）のみに閉じている。

---

## 2. 実装した admission 順序（実コード）

`ConvolverProcessor::LoaderThread::doLoadIRStep` の実測行番号:

| 順序 | 内容 | 行 |
|---|---|---|
| 1 | `createReaderFor` | 377-379 |
| 2 | `fileLength` / `rawChannels`（unsigned のまま） | 388-389 |
| 3 | **FC-FORM-4** degenerate（channel ≠ 0 / length ≥ 1） | 395-405 |
| 4 | **FC-FORM-3** INT32 representation | 407-411 |
| 5 | **FC-FORM-2** channel ≤ 8 | 414-418 |
| 6 | **FC-FORM-1** byte ≤ 1 GiB（除算形） | 421-426 |
| 7 | `numChannels` の narrowing（ここで初めて） | 428 |
| 8 | **FC-FORM-6** `computeIRHash`（O(1)） | 430-431 |
| 9 | chunk temporary（`N × 262,144`）+ `loadedIR` | 442-451 |
| 10 | chunked read | 454-479 |
| 11 | 末尾無音トリム → `L_trim` | 494-546 |
| 12 | **FC-FORM-5**（trim 後・resampler 構築前） | 548-565 |
| 13 | resampler 構築 | 572 |
| 14 | build | — |

**「admission より前に O(fileLength)/O(file bytes) の確保を置かない」を満たす**:
scan で admission 最終行 422 < allocation 最小行 431 を機械確認済み。

### 2.1 指示書の ordering 図からの意図的な逸脱（IG で確定済み）

指示書 §2 は「trim 後 length に対する FC-FORM-5」を **byte admission より前**に置いていたが、
`L_trim` は `loadedIR` を確保して trim した後でしか確定しない。IG §2.1 で
**§3 / R2-D4 を正**として確定しており、本実装はそれに従った（契約値の変更はない）。

---

## 3. 算術（overflow gate）

`admitByteBudget`（`IRLoadAdmission.h:86-93`）は**除算形**:

```cpp
const int64_t denom = static_cast<int64_t>(numChannels) * static_cast<int64_t>(sizeof(double));
return fileLength <= kMaxIRLoadBytes / denom;      // denom ∈ [8, 64]
```

- 被除算は定数 `kMaxIRLoadBytes` のみ。**未検証値の乗算が 1 回も現れない。**
- 先行する FC-FORM-2（N ≤ 8）と FC-FORM-3（L ≤ INT32_MAX）により、
  仮に乗算形を用いても積の上限は `8 × 2,147,483,647 × 8 = 137,438,953,472`（38 bit）で
  int64 に収まる（int32 では 17.2e9 で溢れるため 64-bit 必須）。
- `numChannels` の判定は **narrowing 前の unsigned 値**で行う（`reader->numChannels` は
  `unsigned int`。INT_MAX 超が負値に化ける経路を型レベルで排除）。

FC-FORM-5 は **double 比較**（`admitResampleOutput`）:
`(double)L_trim / fileSR ≤ (double)kMaxIRResampleOutputSamples / sr`。
r8b `getMaxOutLen` の `(int) ceil(...) + 1` に到達する前に拒否されるため、
N=1 極長（2,337,397,169 > INT32_MAX）の int 変換溢れは発生しない。

---

## 4. FC-8（streaming hash）

`AllpassDesigner.cpp:154-262` に `Xxh64Stream`（O(1) 補助メモリ = 32 B バッファ +
アキュムレータ）を追加し、`computeIRHash`（`:667-719`）を逐次処理へ置換。
`juce::HeapBlock<uint8_t> fileData` とその `malloc(fileSize)` は**完全撤去**。

| 要件 | 確認 |
|---|---|
| O(1) auxiliary memory | 確保はスタックの定数のみ。scan1 で fileSize 比例確保 0 件 |
| digest 不変 | 標準 XXH64 と同一規格（prime / rotate / merge / avalanche / `h += totalLen` 一致）。**13 種の長さ**（0,1,7,31,32,33,63,64,65,4095,4096,4097,100000）で独立 one-shot オラクルと全一致 |
| TOCTOU 整合 | before/after の size+mtime 検証を維持（`:712-715`）。読込バイト不一致は `totalRead != declaredSize` として同値に検出 |
| cancellation / failure semantics 不変 | return 0 条件は同一集合。**cancellation は追加していない**（現行も持たないため） |
| admission 迂回なし | 呼出を `:363-364` → `:430-431`（admission 後）へ移動。`stepFileHash` の唯一の消費者は `:651` で同一ステップ内のため観測可能な変化なし |

副次効果: 旧実装にあった「`HeapBlock` の `throwOnFailure=false` により失敗時 nullptr のまま
`memcpy` して AV に至る」経路が消滅（FC-INV-1 / FC-INV-5 の現行違反を閉塞）。

---

## 5. 診断（FC-INV-8 の単一情報源）

`IRLoadAdmission.h` の `diagnosticChannelLimit` / `diagnosticLengthLimit` /
`diagnosticByteLimit` / `diagnosticResampleLimit` に集約し、
dimension + actual + limit を開示する。旧文言
`"IR file is too large (exceeds 2GB samples limit)."`（sample と誤認させる）は撤去。

| 発火 | 文言（抜粋） |
|---|---|
| FC-FORM-2 | `IR file has too many channels (9 channels; limit is 8).` |
| FC-FORM-3/4 | `IR file is too long (2200000000 samples; limit is 2147483647 samples).` |
| FC-FORM-1 | `IR file is too large for the load memory limit (2 channels x … bytes = … bytes; limit is 1073741824 bytes).` |
| FC-FORM-5 | `IR is longer than the DSP can use at this sample rate (3.000 s at 44100.0 Hz; limit is 2.731 s at 768000.0 Hz = 2097153 samples).` |

`"IR too large (Out of Memory)"`（allocation 失敗・`:128`）は別 dimension のため**変更していない**。

---

## 6. Preview / UI の非変更

- `ConvolverProcessor.ResampleAndFallback.cpp` の `loadImpulseResponsePreviewFile` は
  `maxFileLength = 2147483647` ガードのまま**無変更**（scan7）。
- `ConvolverControlPanel.cpp` も**無変更**（scan9）。
- preview の admission parity と graceful failure は **WORK102-PREV-01（OPEN）** として
  引き続き追跡する。本実装は「main loader の安全性」を主張するものであり、
  preview の未解決を main loader の安全性の根拠にはしない。

---

## 7. テスト結果（A〜L）

| Case | 内容 | 結果 |
|---|---|---|
| A | 1ch accept | **PASS** |
| B | 2ch accept | **PASS** |
| C | 8ch accept | **PASS**（SR-01B「>2ch→先頭2」の回帰を兼ねる） |
| D | 9ch deterministic reject | **PASS**（actual=9 / limit=8 を文言で確認） |
| E | 1 GiB 境界 predicate | **PASS**（N=1/2/3/4/8 で limit ちょうど=真、+1=偽、INT32_MAX=偽） |
| F | INT32_MAX 境界 predicate | **PASS**（+1 偽 / FC-FORM-5 境界 / vacuous / 溢れ回帰） |
| G | raw > hardMax だが trim 後 accept | **PASS**（48 kHz 50 s → trim → accept。raw 適用なら拒否される長さ） |
| H | trim 後 FC-FORM-5 reject | **PASS**（44.1 kHz 3 s を 768 kHz で拒否） |
| I | streaming hash digest 一致 | **PASS**（13 種の長さで独立 one-shot と一致） |
| J | allocation failure の graceful path | **構造検証**（§8 参照） |
| K | cancellation semantics 不変 | **構造検証**（§8 参照） |
| L | resample path accept | **PASS**（44.1 kHz → 48 kHz） |

**Debug / Release 双方で同一結果**。CTest は両構成 **40/40 PASS**。

### 7.1 実装中に検出した事象（対処済み）

**Debug harness の初回実行が `testCallerDestroyTerminalDisposition` 内で SEGFAULT（exit 139）。**
原因は新規テストが `ConvolverProcessor::releaseResources()` を呼ばず LoaderThread を滞留させたこと
（work98 §8-13 が同種の前例を記録済み）。`ConvReleaser` RAII guard で全 exit 経路に
片付けを導入し、再走で Debug harness exit 0 / CTest 40/40 に復帰。production 側の変更は不要。

---

## 8. J / K / I(ii) の構造検証（決定論的テストが不能な項目）

| 項目 | 検証方法 | 結果 |
|---|---|---|
| **J** allocation failure → graceful | `performLoad` の catch 群（`:128` bad_alloc / `:133` std::exception / `:138` …）が**無変更**であることを差分で確認。選択的 fault injection は harness に存在しないため実行テストは行わない | 維持 |
| **K** cancellation semantics | `:427-431` のチャンク毎 cancellation と文言 `"IR loading cancelled."` が**無変更**であることを差分で確認。`externalCancellationCheck` は loader 内部からのみ設定されテストから注入不可（既存前例 0 件） | 維持 |
| **I(ii)** hash の O(1) メモリ | scan1/scan2（コメント除去後）で `computeIRHash` 内および production 全体に fileSize 比例確保が **0 件**であることを確認 | PASS |

**残置の観察（WORK102 scope 外）**: `MixedPhasePersistentCache.cpp:253` が
キャッシュファイル長で `std::vector<uint8_t>` を `resize` する（read-modify-write による
タイムスタンプ更新）。IR ロード経路ではなくアプリ生成ファイルが対象のため
FC-FORM-6 の対象外だが、同種の bounded-allocation 関心として記録する。

---

## 9. ゲート結果一覧

```text
build (Release / Debug)           : EXIT=0 / EXIT=0（変更 TU の警告 0）
targeted tests (WORK102 A〜L)     : Debug / Release 双方 PASS
CTest Release                     : 40/40 PASS（63.40 s）
CTest Debug                       : 40/40 PASS（89.80 s）
clang-tidy（変更 3 TU）           : warning 0 / error 0
cppcheck（変更 3 ファイル）       : 変更コードへの指摘 0
                                    （uninitMemberVar 1 件を検出→修正済み。
                                      残存 2 件は未変更 TU でも再現する pre-existing 設定要因）
source scan（必須 4 項目）        : すべて 0 件
RT / architectural scan           : atomic 追加 0 / MAX_IR_LATENCY 差分 0 / RT 経路変更 0
ConvoPeq.md 再生成                : 2026-09-17 09:13:37
ConvoPeq freshness check          : NEWER_SRC_COUNT=0 / FRESH
commit / push                     : 0（Final Gate 待ち）
```

---

## 10. 契約適合の確認（R2 との差分 0）

| R2 の凍結項目 | 実装値 | 一致 |
|---|---|---|
| `kMaxIRLoadBytes` = 1 GiB | 1073741824 | ✓ |
| `kMaxIRLoadChannels` = 8 | 8u | ✓ |
| IR file SR envelope 44.1–768 kHz | 変更なし（ドキュメント契約） | ✓ |
| `MAX_FILE_LENGTH` = INT32_MAX | 2147483647 | ✓ |
| `kMaxIRResampleOutputSamples` = 2^21+1 | 2097153 | ✓ |
| FC-FORM-5 は trim 後適用 | `:548-565`（`:544` shrinkToFit の後、`:572` resampler の前） | ✓ |
| simultaneous residency ≤ 2 GiB | **最適化せず**（`shrinkToFit` は無変更・IG IS-7 として別候補に残置） | ✓ |
| FC-8 = streaming hash | `Xxh64Stream` | ✓ |
| Preview = WORK102-PREV-01 として分離 | preview / UI 無変更 | ✓ |
| 追加 policy value なし | 定数は R2 の 4 値のみ | ✓ |

---

## 11. 次工程

```text
WORK102-IMPL  →  IMPLEMENTED
        ↓
WORK102-POST（Final Gate 前の最終確認）
        ↓
Final Gate → commit → push → post-push integrity
```

**持ち越し（R2 の governance、IMPL とは独立）**:
- IG-1〜IG-3 の countersign（envelope 承認 / SR-01B supersede 承認 / CC 承認）
- WORK102-PREV-01（preview の admission parity + graceful completion、OPEN）
- IG IS-7（trim の copy-on-resize 回避による 2 GiB → ~1.5 GiB は**未実施**・任意候補）
