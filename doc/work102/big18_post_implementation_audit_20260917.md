# WORK102-POST — Post-Implementation Contract / Scope / Freshness Audit（read-only）

- **作成日**: 2026-09-17
- **種別**: read-only post-implementation audit（**Final Gate 前の独立確認**）
- **一次ソース**: 再生成済み `ConvoPeq.md`（Generated 2026-09-17 09:13:37）+ 現行 source tree
- **方針**: IMPL の報告内容や記憶を根拠にせず、**bundle と source から順序・値・意味を再導出**した
- **変更**: production / tests / CMake / UI = 0（本工程は監査のみ）

```text
判定: POST-PASS
```

---

## 0. Baseline

```text
HEAD        = 35461e2e
origin/main = 35461e2e   (ahead/behind = 0/0)

ConvoPeq.md Generated : 2026-09-17 09:13:37
NEWER_SRC_COUNT       : 0
STATUS                : FRESH — snapshot は現行ソースを反映しています

working-tree (src / CMake / tests):
  M  CMakeLists.txt
  M  src/AllpassDesigner.cpp
  M  src/convolver/ConvolverProcessor.LoaderThread.cpp
  M  src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp
  ?? src/convolver/IRLoadAdmission.h
  ?? src/tests/AudioEngineHarness/IRLoadAdmissionTests.cpp
  M  ConvoPeq.md（再生成）

= WORK102-IMPL の変更面と完全一致（unrelated change なし）
```

---

## 1. 契約追跡（R2 → IG → IMPL）

**bundle 内の定数宣言から直接抽出**（`ConvoPeq.md:77393-77406`）:

| Contract | R2 凍結値 | bundle 実測 | 意味の保持 |
|---|---|---|---|
| `kMaxIRLoadBytes` | 1 GiB | `1073741824` | **不変**。除算形の被除算としてのみ使用 |
| `kMaxIRLoadChannels` | 8 | `8u` | **不変**。`unsigned` domain で判定 |
| file SR envelope 44.1–768 kHz | — | 変更対象外（ドキュメント契約） | **不変**（コードに SR 包絡の追加制約なし） |
| `MAX_FILE_LENGTH` | INT32_MAX | `2147483647` | **不変**。narrowing の型契約として保持 |
| `kMaxIRResampleOutputSamples` | 2,097,153 | `2097153` | **不変**。double 比較の分子 |
| FC-FORM-5 適用点 | trim 後 | trim 後（§2.3） | **不変** |
| simultaneous residency | 2 GiB | `shrinkToFit` 未変更 | **不変**（最適化していない） |
| FC-8 | streaming XXH64 | `Xxh64Stream` | **不変**（§4） |
| Preview | PREV-01 分離 | 両ファイル無変更 | **不変**（§6） |

### 1.1 `MAX_IR_LATENCY` と loader allocation bound の非混同（重点確認）

```text
確認方法: convo::irload 名前空間のコード行からコメントを除去して MAX_IR_LATENCY を検索
結果    : コード上の出現 0 件（コメント内の説明 1 件のみ）

  → kMaxIRLoadBytes は MAX_IR_LATENCY から構成されていない（独立の定数）。
  → FC-FORM-5 の値 2097153 は「U + 1」を literal で保持しており、
     loader 側で MAX_IR_LATENCY を再計算・参照していない。
  → FC-INV-3（非統合）は実装後も保持されている。
```

---

## 2. Admission ordering の post-audit（bundle から再構成）

bundle の `doLoadIRStep` を正規表現で走査し、**行番号順序を機械的に再導出**した:

| 順序 | 要素 | bundle 行 |
|---|---|---|
| 1 | `createReaderFor` | 72930 |
| 2 | `fileLength` / `rawChannels` | 72941-72942 |
| 3 | **FC-FORM-4**（channel ≠ 0） | 72945 |
| 4 | **FC-FORM-4**（length ≥ 1） | 72950 |
| 5 | **FC-FORM-3**（INT32） | 72957 |
| 6 | **FC-FORM-2**（channel ≤ 8） | 72964 |
| 7 | **FC-FORM-1**（byte ≤ 1 GiB） | 72971 |
| 8 | narrowing `int(rawChannels)` | 72977 |
| 9 | **FC-FORM-6** `computeIRHash` | 72980 |
| 10 | chunk temporary | 72992 |
| 11 | `loadedIR.setSize` | 73000 |
| 12 | chunk read | 73003 |

**strictly increasing: OK。**「最後の admission（72971）より前に置かれた hash / temp / loadedIR は 0 件」。

### 2.1 Allocation-before-admission

| 対象 | FC-FORM-1（72971）との関係 | 判定 |
|---|---|---|
| `loadedIR`（73000） | 後 | **OK** |
| file-size proportional buffer | **存在しない**（§4 で 0 件を確認） | **OK** |
| full-size temporary | **存在しない**（`tempFloatBuffer` は `N × 262,144` の chunk サイズ・72992） | **OK** |
| hash buffer | **存在しない**（O(1)・72980） | **OK** |

### 2.2 Hash placement

`computeIRHash`（72980）は **FC-FORM-1（72971）より後**。要求を満たす。

> **観察（drift ではない）**: 指示書 §2 の期待図は FC-FORM-1 と FC-FORM-6 の間に
> 「O(file bytes) / O(fileLength) allocation」を挟んでいるが、実装にはその区間に確保が
> **1 件も存在しない**（hash が O(1) 化され、chunk temp と `loadedIR` は FC-FORM-6 の後）。
> 図より**厳しい側**に倒れており、§2.1 / §2.2 の両方を満たす。

### 2.3 FC-FORM-5（raw `fileLength` に戻っていないことの重点確認）

```text
shrinkToFit（末尾無音トリムの縮小） : 73093
FC-FORM-5（admitResampleOutput）    : 73104
resampler 構築（resampleIR 呼出）    : 73121

  → trim → L_trim 確定 → FC-FORM-5 → resampler の順序を満たす（OK）

operand の実測:
  73103  const int64 trimmedLength = stepResult.loadedIR.getNumSamples();
  73104  if (!convo::irload::admitResampleOutput(trimmedLength, stepResult.loadedSR, sampleRate))

  → **trim 後の長さ**を渡している。raw fileLength ではない（R2-D4 の要求を保持）。
```

---

## 3. Arithmetic / type safety audit

### 3.1 FC-FORM-1（除算形）

bundle の `admitByteBudget` 実測:

```cpp
[[nodiscard]] inline bool admitByteBudget(unsigned numChannels, int64_t fileLength) noexcept
{
    if (!admitChannelCount(numChannels) || !admitFileLengthNonZero(fileLength))
        return false;

    const int64_t denom = static_cast<int64_t>(numChannels) * static_cast<int64_t>(sizeof(double));
    return fileLength <= kMaxIRLoadBytes / denom;
}
```

| 要求 | 実測 | 判定 |
|---|---|---|
| `numChannels` admission 後 | 冒頭で `admitChannelCount` を再評価 | **OK** |
| 64-bit arithmetic | `int64_t denom` / `kMaxIRLoadBytes` は `int64_t` | **OK** |
| zero divisor 不可 | `admitChannelCount` が `numChannels ≥ 1` を保証 → `denom ∈ [8, 64]` | **OK** |
| narrowing 前の channel 値 | 引数が `unsigned numChannels` | **OK** |
| 未検証値の乗算なし | `fileLength ×` が本文に存在しない（除算形） | **OK** |

### 3.2 FC-FORM-3 と narrowing の順序

```text
FC-FORM-3（admitFileLengthRepresentable） : 72957
int(fileLength) の最初の出現            : 73000（loadedIR.setSize）/ 73012（chunk 長）
  → INT32_MAX admission **より前に narrowing は発生していない**（OK）

channel narrowing: 72977（FC-FORM-2 の 72964 より後）→ OK
```

### 3.3 FC-FORM-5 と r8b の int conversion

```text
FC-FORM-5（reject 判定）           : 73104
r8b getMaxOutLen の (int) conversion: 74615（resampleIR 内）
  → FC-FORM-5 reject → r8b getMaxOutLen → (int) conversion の順序を満たす（OK）
  → N=1 極長（2,337,397,169 > INT32_MAX）は 73104 で先に拒否され、
    74615 の int 変換に到達しない。
```

`admitResampleOutput` は **double domain** のみで比較する:

```cpp
if (fileSampleRate <= 0.0 || processingSampleRate <= 0.0) return true;   // vacuous
if (trimmedLength <= 0) return false;
const double required = static_cast<double>(trimmedLength) / fileSampleRate;
const double allowed  = static_cast<double>(kMaxIRResampleOutputSamples) / processingSampleRate;
return required <= allowed;
```

int 中間値を生成しないため、fileSR が極小でも桁溢れしない。

---

## 4. Streaming XXH64 post-audit（意味の同一性）

「vector が無い」ではなく、**旧 hash と意味的に同一か**を bundle から確認した。

| # | 確認項目 | 実測 | 判定 |
|---|---|---|---|
| 1 | seed 不変 | `0x434f4e564f504551ull`（"CONVOPEQ"） | **OK** |
| 2 | XXH64 prime 不変 | prime1〜5 が既知の XXH64 定数と完全一致 | **OK** |
| 3 | round / merge / avalanche 不変 | `rotl64(v1_,1)+rotl64(v2_,7)+rotl64(v3_,12)+rotl64(v4_,18)` / `xxh64MergeRound ×4` / `xxh64Avalanche` | **OK** |
| 4 | `totalLen` の扱い不変 | `h += total_;`（one-shot の `h += len` と同義） | **OK** |
| 5 | tail 1/4/8-byte 不変 | 8-byte ループ → 4-byte 単発 → 1-byte ループ | **OK** |
| 6 | `readLE32/64` の endian / unaligned 不変 | `std::memcpy(&v, p, sizeof(v))`（整列非依存・リトルエンディアン維持） | **OK** |
| 7 | before/after size+mtime 不変 | `sizeBefore != sizeAfter \|\| mtimeBefore != mtimeAfter` | **OK** |
| 8 | 読込バイト不一致検出の維持 | `totalRead != declaredSize`（旧 `writePos != fileSize` と同値）＋ `totalRead > declaredSize` の早期終了 | **OK** |
| 9 | file-size proportional allocation = 0 | production 全体（コメント除去後）で `HeapBlock<uint8_t>` / fileSize 長 vector が **0 件** | **OK** |
| 10 | cancellation semantics を追加していない | `kHashCancel` / `hashShouldExit` 等の新規 API **0 件** | **OK** |

**digest 等価テストの残存**: `checkStreamingHashMatchesReference` が
**13 種の長さ**（0, 1, 7, 31, 32, 33, 63, 64, 65, 4095, 4096, 4097, 100000）で
テスト内の**独立 one-shot 実装**（`refXxh64`）と照合し、**Debug / Release 双方 PASS**（§8）。
テストは seed を `kExpectedHashSalt = 0x434f4e564f504551ull` として独立に固定している。

---

## 5. Error / failure semantics

### J — allocation failure の graceful handling

| 項目 | bundle 実測 | 判定 |
|---|---|---|
| `catch (const std::bad_alloc&)` | 存在（`performLoad`） | **維持** |
| `catch (const std::exception& e)` | 存在 | **維持** |
| `catch (...)` | 存在 | **維持** |
| `"IR too large (Out of Memory)"` | 存在（文言変更なし） | **維持** |
| 旧 hash の `HeapBlock` allocation failure 経路 | `HeapBlock<uint8_t>` が production から消滅 | **消滅（意図どおり）** |

旧経路は `ThrowOnFail<false>` により失敗時 nullptr のまま `memcpy` する AV 経路だった。
streaming 化により**確保そのものが無くなった**ため、この failure mode は構造的に消滅した。

### K — cancellation semantics

```text
chunk read（73003）→ cancellation check（73005）→ "IR loading cancelled."（73007）
  → 文言・位置・頻度（チャンク毎）すべて不変
新規 cancellation API の追加 : 0 件（§4-#10）
```

---

## 6. Preview boundary audit（OPEN 維持）

| 対象 | 実測 | 判定 |
|---|---|---|
| `src/convolver/ConvolverProcessor.ResampleAndFallback.cpp` | `git status` = **無変更** | **OK** |
| `src/ConvolverControlPanel.cpp` | `git status` = **無変更** | **OK** |
| preview が新 admission を使用 | `irload::` の出現 **0 件** | **横展開なし** |
| preview の既存ガード | `maxFileLength = 2147483647` 保持 | **OK** |
| preview の全長 temp | `tempFloatBuffer(numChannels, fileLength)` 保持（未改修） | **OK** |
| UI の preview reject | `exceedsHardLimit` 保持 | **OK** |

```text
main loader = WORK102-IMPL の対象
preview     = WORK102-PREV-01 = OPEN（admission parity + graceful completion を未解決のまま追跡）
```

**「Preview が未解決だから main loader は安全」という論理は採用していない。**
main loader の安全性の根拠は §2〜§5 の順序・算術・hash 設計であり、
preview の状態とは独立である。逆に preview の admission 不在・graceful failure 不在は
PREV-01 の未解決項目として残る。

---

## 7. Scope / architecture audit（追加コード行に対して）

| 項目 | 件数 | 判定 |
|---|---|---|
| atomic additions（`std::atomic` / `fetchAdd` / `memory_order` 等） | **0** | OK |
| RT process-path changes（`process(` / `isAudioThread` / `ScopedNoDenormals`） | **0** | OK |
| Publish changes | **0** | OK |
| Crossfade changes | **0** | OK |
| Retire changes | **0** | OK |
| Epoch changes | **0** | OK |
| RuntimeWorld changes | **0** | OK |
| LifetimeBudget changes | **0** | OK |
| ISRShutdown changes | **0** | OK |
| `MAX_IR_LATENCY` modifications | **0** | OK |
| `DELAY_BUFFER_SIZE` changes | **0** | OK |
| UI changes（`ConvolverControlPanel.cpp` / `ConvolverSettingsComponent.cpp`） | **0** | OK |

**RT = no new allocation / lock / wait / decision**（追加コード行の実測）:

```text
allocation : 0 件
lock       : 0 件
wait       : 0 件
追加コードは LoaderThread（NonRT）/ そこから呼ばれる AllpassDesigner / テストのみ
```

`doc/Practical Stable ISR Bridge Runtime.md:3`（RT は待たない・解放しない・判断しない）および
`:704`（危険操作を NonRT へ隔離）に対し、本変更は既に NonRT である経路のみを変更している。

---

## 8. Test integrity audit

「PASS の再掲」ではなく、**凍結契約を検査しているか**を確認した。

| 検査対象 | 実測 | 判定 |
|---|---|---|
| チェック関数 | 6 件（`checkEFPredicateBoundaries` / `checkChannelAdmissionRuntime` / `checkResampleBoundUsesTrimmedLength` / `checkResampleBoundRejectsOversized` / `checkResamplePathAccepted` / `checkStreamingHashMatchesReference`） | OK |
| A–D | 実ファイル（1 / 2 / 8 / 9ch）で admission を検証。FAIL パスに `A-D` ラベルあり | OK |
| E–F | 純関数 predicate。境界値は N=1/2/3/4/8 の **limit ちょうど**（134217728 / 67108864 / 44739242 / 33554432 / 16777216）と **limit+1**、加えて **INT32_MAX** / **INT32_MAX+1**、FC-FORM-5 の `2097153` ちょうど と `+1` | OK |
| G | 「raw に FC-FORM-5 を適用する実装なら拒否される」ことをコメントで明示した **FC-FORM-5 位置の回帰**として機能（48 kHz 50 s → trim → accept） | OK |
| H | trim 後 resample output 超過を reject（44.1 kHz 3 s を 768 kHz で） | OK |
| I | `refXxh64`（独立 one-shot）との digest 等価。13 種の長さ | OK |
| J / K / I(ii) | ファイル冒頭と各所で **構造検証** として明示。PASS を偽装していない | OK |
| 片付け契約 | `ConvReleaser` guard により全 exit 経路で `releaseResources()` | OK |

### 8.1 Debug / Release 双方（POST での再実行）

```text
CTest Release : 100% tests passed out of 40  (Total 53.15 s / AudioEngineHarness 39.87 s)
CTest Debug   : 100% tests passed out of 40  (Total 100.97 s / AudioEngineHarness 60.24 s)

直接実行での WORK102 出力（両構成で同一）:
  [WORK102] checkEFPredicateBoundaries: PASS                                    … E / F
  [WORK102] checkStreamingHashMatchesReference: PASS (13 sizes, one-shot oracle) … I
  [WORK102] checkChannelAdmissionRuntime: PASS (1/2/8ch accept, 9ch reject)      … A/B/C/D
  [WORK102] checkResampleBoundUsesTrimmedLength: PASS (raw 50s -> trim -> accept) … G
  [WORK102] checkResampleBoundRejectsOversized: PASS (3s@44.1k -> 768k reject)    … H
  [WORK102] checkResamplePathAccepted: PASS (44.1k -> 48k resample accept)        … L
  IRLoadAdmissionTests: PASS (WORK102 A/B/C/D/E/F/G/H/I/L)
  Debug harness exit code = 0
```

---

## 9. 新規 `IRLoadAdmission.h` の独立監査

bundle 内の span（`ConvoPeq.md:77384-77527`）から再確認。

| 確認対象 | 実測 | 判定 |
|---|---|---|
| 契約定数が R2 と一致 | 4 件すべて一致（§1） | OK |
| predicate が I/O 非依存 | `juce::File` / `FileInputStream` / `createReaderFor` の出現 **0 件** | OK |
| diagnostic source が単一 | `diagnostic*` 4 件を本 header に集約（loader は呼ぶだけ） | OK |
| 新しい policy value なし | `inline constexpr` は **4 件のみ**（R2 の 4 値） | OK |
| channel predicate が unsigned domain を壊さない | `bool admitChannelCount(unsigned numChannels)`。narrowing は呼出側で 1 箇所のみ | OK |
| byte predicate が overflow-safe | 除算形（§3.1） | OK |
| resample predicate が double domain | `static_cast<double>` 比較のみ（§3.3） | OK |
| 確保を一切行わない | `new` / `malloc` / `std::vector` / `setSize` の出現 **0 件** | OK |
| **production と test が同じ predicate を使う** | loader 側 `convo::irload::` 参照 12 箇所 / test 側 24 箇所。test は `using namespace convo::irload;` で**同一実体**を参照（predicate の再実装なし） | OK |

**形式的性質**: 全 predicate が `[[nodiscard]] inline bool ... noexcept`。
診断は `[[nodiscard]] inline juce::String`。header-only で ODR 問題なし（`inline`）。

---

## 10. `ConvoPeq.md` freshness

```text
baseline Generated : 2026-09-17 09:13:37  (ConvoPeq.md)
NEWER_SRC_COUNT    : 0
STATUS             : FRESH — snapshot は現行ソースを反映しています
```

**必須条件を満たす。** 本 POST は read-only であり、この状態を変化させていない。

---

## 11. POST 判定

```text
POST-PASS
```

| POST-PASS 条件 | 実測 | 判定 |
|---|---|---|
| R2 契約値不変 | 4 定数すべて一致（§1） | **PASS** |
| IG ordering 不変 | bundle 行順が strictly increasing（§2） | **PASS** |
| admission-before-allocation 不変 | 最後の admission より前の確保 0 件（§2.1） | **PASS** |
| FC-FORM-5 = trim 後 | trim(73093) < FC-FORM-5(73104) < resampler(73121)、operand は `trimmedLength`（§2.3） | **PASS** |
| overflow safety 維持 | 除算形 / int64 / zero-divisor guard / double 比較 / r8b 前評価（§3） | **PASS** |
| XXH64 digest equivalence | 10 項目すべて不変 + 13 長の独立 oracle 照合が両構成で PASS（§4, §8） | **PASS** |
| failure / cancellation semantics 維持 | catch 群・文言不変、新規 cancellation なし（§5） | **PASS** |
| preview OPEN 維持 | 2 ファイル無変更・横展開 0・OPEN のまま（§6） | **PASS** |
| RT / architecture scope drift = 0 | 11 項目すべて 0、RT alloc/lock/wait 0（§7） | **PASS** |
| A–L の期待結果維持 | 両構成で全 PASS、J/K/I(ii) は構造検証として明示（§8） | **PASS** |
| Debug / Release 両方 green | CTest 40/40 × 2、harness exit 0（§8.1） | **PASS** |
| `ConvoPeq.md` FRESH | NEWER_SRC_COUNT=0（§10） | **PASS** |
| unrelated changes なし | 変更面 6 ファイル = IMPL 集合と完全一致（§0） | **PASS** |

**contract drift は検出されなかった。**

### 11.1 判定に影響しない残置事項（明示）

```text
IS-7   trim の copy-on-resize による simultaneous residency 2 GiB → 約 1.5 GiB の最適化
       → **未実施**。R2 の契約値は 2 GiB のままであり、本 POST の条件ではない。
WORK102-PREV-01  preview の admission parity + graceful completion
       → **OPEN 維持**。本 POST の条件ではない（PREV-01 の受入基準として残る）。
governance  IG-1〜IG-3 の countersign（envelope / SR-01B supersede / CC 承認）
       → R2 の governance 手続きとして独立。
```

### 11.2 監査上の観察（drift ではない）

```text
指示書 §2 の期待図は FC-FORM-1 と FC-FORM-6 の間に確保を描いているが、
実装はその区間に確保を 1 件も持たない（hash が O(1) 化されたため）。
図より厳しい側であり、§2.1 / §2.2 の双方を満たす。契約の変更ではない。
```

---

## 12. POST 後の扱い

```text
POST-PASS では commit / push しない（本工程は監査のみ）。
次工程: FINAL GATE → commit → push → post-push integrity → WORK102 CLOSED
```

---

## 13. 監査メタ

```text
検証方法    : 再生成 ConvoPeq.md（一次ソース）からの行番号・定数・意味の再導出
              + 現行 source tree の直接実測 + git diff / git status による scope 差分
              + 独立 one-shot XXH64 oracle（テスト内）+ CTest 再実行（両構成）

Production 変更     : 0
tests 変更          : 0
CMake 変更          : 0
UI 変更             : 0
ConvoPeq.md 再生成  : 0（FRESH 維持）
commit / push       : 0
README E-G3-1 差分  : 維持

使用ツール: rg / python（bundle 走査・差分解析）/ ctest（両構成）/
            clang-tidy・cppcheck（IMPL で実施済み・本 POST は結果を再掲せず実測のみ）
```
