# WORK102 — big 1-8 Pre-Audit: Bounded-Allocation Contract Closure（read-only）

- **作成日**: 2026-09-17
- **種別**: Pre-Audit（read-only）
- **位置付け**: **「未修正バグの修正」ではない**。現行実装が既に持っている実効上限を
  **契約として固定するための bounded-allocation contract closure**。
- **実装**: 行わない（上限値そのものも本 work では決めない）

## 0. Baseline

```text
HEAD        = 35461e2e3a5fd42fd009416a20adaf457b9b4031
origin/main = 35461e2e3a5fd42fd009416a20adaf457b9b4031   (ahead/behind = 0/0)

Source（authoritative）:
  ConvoPeq.md  Generated = 2026-09-17 00:07:34
  --check      NEWER_SRC_COUNT = 0 / STATUS = FRESH

production source = 0 / CMake = 0 / UI = 0 / tests = 0
ConvoPeq regeneration = 0 / commit = 0 / push = 0
```

**scope 外（本 work では触らない）**: M-03 D3 / M-03 latency・PDC / Direct Head HC/LC / H-01 / H-02 /
SR-01〜03 / Publish authority / Crossfade authority / Retire authority / Epoch / RuntimeWorld /
Lifetime Budget / RT path。

## 1. 調査対象と前提の固定

目的は「ストリーミング化」ではない。**現行の（チャンク読込ベースの）ロード方式を維持したまま、
許容最大 IR サイズを明示的な契約として固定できるか**を検証する。

### 1.1 本 Pre-Audit の最重要分離

```text
Load capacity            ≠      DSP usable IR length
```

- `MAX_IR_LATENCY`（= 2,097,152 samples）は **DSP が使用する IR length の上限**。
- Loader の `fileLength` bound は **loader が確保可能な入力ファイル長の上限**。
- **両者を同一の capacity constant に統合することは本 scope では禁止。**

WORK92-RT の時点で記録した「`computeTargetIRLength` が使用長を `MAX_IR_LATENCY` で有界化している」は
**使用長の話**であり、**loader の確保量の話ではない**。本 Pre-Audit はここを明確に分離する。

### 1.2 事前に固定する既知事実（WORK92-RT の記録からの引き継ぎ）

```text
fileLength > INT32_MAX          → reject（allocation 前）
fileLength <= INT32_MAX         → loadedIR.setSize(numChannels, fileLength)
computeTargetIRLength()         → MAX_IR_LATENCY = 2,097,152 samples で「使用長」を制限
```

## 2. A. Loader allocation の実際の上限（source trace）

### 2.1 読込ステップの全容（`ConvolverProcessor.LoaderThread.cpp::doLoadIRStep`）

```text
:389  const int64 fileLength  = reader->lengthInSamples;
:390  const int   numChannels = static_cast<int>(reader->numChannels);
:391  static constexpr int64 MAX_FILE_LENGTH = 2147483647;
:393  if (fileLength > MAX_FILE_LENGTH) → errorMessage "IR file is too large (exceeds 2GB samples limit)." / return false
:398  if (numChannels <= 0)             → errorMessage "Invalid channel count in IR file." / return false
:413  constexpr int64 kStreamChunk = 256 * 1024;
:414  juce::AudioBuffer<float> tempFloatBuffer(numChannels, (int)kStreamChunk);
:415  auto tempAligned = convo::makeAlignedArray<double>((size_t)kStreamChunk);   ← null 検査あり（:416-420）
:422  stepResult.loadedIR.setSize(numChannels, (int)fileLength);                 ← ★ ピーク確保
:423  stepResult.loadedIR.clear();
:425  for (offset = 0; offset < fileLength; offset += kStreamChunk) {            ← チャンク読込ループ
:427      cancellation check（chunk 毎）
:434      chunk = min(kStreamChunk, remaining)
:435      jassert(offset + chunk <= 2147483647)                                  ← G4 narrowing 証明の belt-and-braces
:437      reader->read(&tempFloatBuffer, 0, chunk, offset, true, true)
:443-449  ch 毎に convertFloatToDoubleHighQuality → loadedIR.copyFrom(ch, offset, tempAligned, chunk)
:451  stepResult.loadedSR = reader->sampleRate;
```

**重要（WORK92-RT の記述の精密化）**: 旧 work92 の主張「fileLength 分を一括確保（ステレオ float
~16GB / double ~32GB）」が指す**転送用バッファの一括確保は既に解消済み**である。
`:404-413` の `★ work92 B-5 (big 1-8 / R-新規C)` コメントのとおり **チャンク読込へ変更済み**で、
transient は `kStreamChunk = 262,144` サンプル分に固定されている。

**残置するのは destination（`loadedIR`）の fileLength 比例確保のみ**（`:422`）。

### 2.2 allocation の全量（chain 全体）

| # | 位置 | 確保量 | 規模 |
|---|---|---|---|
| 1 | `:414` | `numChannels × 262,144 × sizeof(float)` | N × 1.05 MB |
| 2 | `:415` | `262,144 × sizeof(double)` | 2.10 MB（固定） |
| 3 | **`:422`** | **`numChannels × fileLength × sizeof(double)`** | **★ ピーク（fileLength 比例）** |
| 4 | `:514` | `numChannels × max(1,newLength) × 8` + `shrinkToFit`（:515） | ≤ #3（縮小のみ） |
| 5 | `:586` | `numChannels × targetLength × 8` | N × targetLength × 8（≤ N × 2,097,152 × 8 = N × 16.8 MB） |
| 6 | `:163-164` | `targetLength × 8` × 2（irL / irR） | 2 × 16.8 MB（固定） |
| 7 | `:244-245`, `:317-318` | `AudioBuffer(std::move(...))` | move のみ（追加確保なし） |

→ **ピークは #3 の `loadedIR` が支配する。**

### 2.3 reachable な最大 channel 数

- `numChannels` は **`reader->numChannels`（ファイルが持つチャンネル数）そのまま**。
- ロード経路に **channel 上限チェックは存在しない**（`:398` の `numChannels <= 0` のみ）。
  全 src を走査した結果、IR 読込経路に `numChannels` の上限を課す記述は 0 件。
- したがって **`numChannels` は形式的には非有界**（ファイル形式が許す範囲）。

**ただし意味論的な上限がある**: build 段（`:166-169`）は **ch0 / ch1 のみ**使用する。

```text
:166  const double* srcL = trimmed.getReadPointer(0);
:167  const double* srcR = (trimmed.getNumChannels() > 1) ? trimmed.getReadPointer(1) : srcL;
```

一方 `:586 stepTrimmed.setSize(stepResult.loadedIR.getNumChannels(), result.targetLength)` は
**全チャンネルを保持**する。→ **N ≥ 3 のチャンネルは build 以降 dead weight**。

### 2.4 reachable な最大 fileLength

- `reader->lengthInSamples`（`int64`）だが `:393` で `MAX_FILE_LENGTH = 2,147,483,647` に制限。
- `:422` の `static_cast<int>(fileLength)` は `fileLength ≤ INT32_MAX` のため値域証明済み
  （G4 narrowing proof。`:435` の jassert が belt-and-braces）。

### 2.5 worst-case resident allocation（実測ベースの算出）

ピーク = `numChannels × fileLength × sizeof(double)`

| channels | fileLength = INT32_MAX | 備考 |
|---|---|---|
| **2（ステレオ）** | **34.36 GB** | WAV 2ch = 実運用の実質上限 |
| 8 | 137.4 GB | マルチチャンネル WAV |
| 64 | 1.10 TB | 形式上は受入可能（上限チェックなし） |

transient（#1+#2）は N × 1.05 MB + 2.10 MB で、ピークに比して無視できる。

**算出に用いた到達可能な範囲**: `fileLength` は `INT32_MAX` まで到達可能（`:393` が通す）、
`numChannels` は上限チェックが無いため**ファイル依存で非有界**。したがって **worst-case bytes は
形式的には非有界**であり、`INT32_MAX` は **samples 上限であって byte 上限ではない**。

### 2.6 allocation failure の扱い（実測）

| 層 | 挙動 |
|---|---|
| JUCE `AudioBuffer::setSize`（`:392-395`） | **`(size_t)` キャスト後に乗算** → `(size_t)newNumChannels × (size_t)allocatedSamplesPerChannel × sizeof(Type)`。**int 溢れなし** |
| JUCE 確保（`:1267` `HeapBlock<char, true>`） | 第 2 引数 `true` = **throwOnFailure**。`:442 allocatedData.allocate(...)` が失敗時 **`std::bad_alloc` を送出** |
| ConvoProq `performLoad`（`:120-142`） | `try { while(true) stepOnce(); }` を **`catch (const std::bad_alloc&)`** が捕捉 → `errorMessage = "IR too large (Out of Memory)"` + Logger 出力で **graceful 失敗**（`std::terminate` なし）。`catch(const std::exception&)` / `catch(...)` も併設 |

→ **巨大確保の失敗は制御されている**。クラッシュ経路は確認できない。

### 2.7 cancellation / failure cleanup（実測）

- **cancellation**: `:427-431` が **chunk 毎**に `externalCancellationCheck` を評価し、
  中断時は `errorMessage = "IR loading cancelled."` で return false。`doTrimStep` / `doTransformStep`
  も `checkCancellation` を各所で実施。
- **failure cleanup**: `stepOnce`（`:329-357`）は失敗時に `StepState::Error` へ遷移して終端。
  `~LoaderThread`（`:28-34`）が `retireStereoConvolver` で `stepResult.newConv` を回収。
  `FlagResetter`（`:51-`）が `isLoading` / `isRebuilding` を復帰。
- **resample 前後**: trim (`:512-516`) で `newLength` に縮小 → `setSize(..., true)` + `shrinkToFit`。
  transform（`:610-677`）は `stepTrimmed` を move で受け渡し（追加の巨大確保なし）、失敗時は
  `validateBuffer` が `false` を返し旧 `stepTrimmed` を維持。

## 3. B. `MAX_IR_LATENCY` との責務分離（証明）

```text
[1] computeTargetIRLength(sampleRate, originalLength)
      = min(sampleRate × targetIRLengthSec, MAX_IR_LATENCY = 2,097,152)   ← originalLength は ignoreUnused
    用途:
      :163-164  irL / irR                       （targetLength サイズ）
      :586      stepTrimmed.setSize(N, targetLength)
    → これは「DSP が使用する IR length」の上限。

[2] doLoadIRStep
      fileLength      : :393 で MAX_FILE_LENGTH = INT32_MAX に制限
      loadedIR        : :422 で (numChannels × fileLength) を確保
    → これは「loader が確保する入力ファイル長」の上限。

[3] 両者の関係
      targetLength は fileLength の関数ではない（originalLength は未使用）。
      よって MAX_IR_LATENCY は loadedIR (:422) のサイズを上界しない。
```

**結論**: `MAX_IR_LATENCY` と Loader file-length bound は **独立した 2 つの capacity constant** であり、
統合してはならない。WORK92-RT に記録した「使用側は既に有界」という記述は [1] についてのみ成立し、
[2]（`:422` の確保）には及ばない。**この分離を本 work の凍結事項とする。**

## 4. C. Option A の検証

```text
Option A:
  現行一括ロードを維持
  ↓
  Loader maximum sample count を明示
  ↓
  上限超過は allocation 前に reject
  ↓
  documentation / diagnostic に明記
```

### 4.1 成立可否

| Option A の要件 | 現行実装 | 判定 |
|---|---|---|
| 現行一括ロード（destination）を維持 | `:422` がそのまま存在 | 成立 |
| 上限超過を **allocation 前** に reject | `:393-397` は `:422` より前に位置し、既に reject する | **構造は既に存在** |
| 上限値の明示 | `MAX_FILE_LENGTH = INT32_MAX` が唯一の値 | **値が契約として不適切（下記）** |
| documentation / diagnostic | `errorMessage = "IR file is too large (exceeds 2GB samples limit)."` のみ | 不十分（後述） |

**→ Option A の構造は現行実装に既にある。不足しているのは「契約値」と「その明文化」だけ。**

### 4.2 「INT32_MAX をそのまま製品契約値に採用する」ことの検証

**採用できない**。理由は 3 点。

1. **sample 上限であって byte 上限ではない**
   `MAX_FILE_LENGTH` は `fileLength`（サンプル数）のみを制限する。`numChannels` に上限が無いため、
   **byte 上限は非有界**のまま残る。
2. **ステレオ double で 34.36 GB**
   `INT32_MAX × 2ch × sizeof(double)` = 34.36 GB。64-bit Windows の実用コミット限界を超えることが多く、
   実際には `std::bad_alloc` → graceful 失敗となる（= 正常動作だが「上限として無意味」）。
3. **N ≥ 3 は build 段で使われない dead weight**
   `:166-169` が ch0/ch1 のみ使用するため、ch3 以降の確保は**意味論的に無駄**である。
   上限を設けることに設計上の損失がない。

### 4.3 Option A を成立させるために必要な契約形（選択肢の提示のみ・決定はしない）

```text
(A-1) byte 上限方式
      numChannels × fileLength × sizeof(double) ≤ kMaxIRLoadBytes
      → channel 数に依存せず常に同一のメモリ上限を保証できる（最も単純・最も強い）

(A-2) channel 上限の併設
      numChannels ≤ kN  ∧  fileLength ≤ MAX_FILE_LENGTH
      → 既存ガードの追加で済む。ただし byte 上限は kN × INT32_MAX × 8 となり依然巨大

(A-3) sample 上限の引き下げ
      MAX_FILE_LENGTH を製品上の意味のある値へ下げる（channel 上限と併用）
      → 「IR として現実的な長さ」を契約に反映できるが、既存ユーザーファイルの互換に注意
```

**本 Pre-Audit ではいずれも決定しない**（Contract Decision の論点）。

### 4.4 診断メッセージの現状（契約明文化の不足点）

現行 `:395` のメッセージは `"IR file is too large (exceeds 2GB samples limit)."` であり、
**(i) byte ではなく sample の話であること (ii) channel 上限が無いこと** を開示していない。
Option A を採る場合、この文言と documentation の双方が契約値と整合している必要がある。

## 5. D. Option B（streaming / chunked loading）の比較

### 5.1 既に実装されている部分

**入力ファイルの読み込み自体は既にチャンク化されている**（`:404-450`、`★ work92 B-5`）。
したがって「Option B = 読込のストリーミング化」は**部分的に実現済み**である。

### 5.2 未実装の部分と波及範囲

残るのは **destination（`loadedIR`）の一括確保をやめる**ことであり、これは以下へ波及する:

| 波及先 | 内容 |
|---|---|
| Loader architecture | 段階的（incremental）partition build が必要 |
| cancellation | チャンク単位の中間状態を破棄できる設計が要る |
| resampling | `stepTrimmed` を全長前提で扱っている（`:586`）ため逐次化が要る |
| state restoration | `IRState`（`ConvolverProcessor.h:1185-1192`）が全長バッファを前提 |
| IR ownership | `irOwner`（unique_ptr）+ `ir`（raw）の単一所有前提が崩れる |
| downstream snapshot publication | 公開単位が「完成 IR」前提 |
| memory lifetime | `applyNewState` / retire 経路（`:319`）の寿命管理が変わる |

`Rebuild.cpp:177` の `IncrementalRebuildJob::incrementalLoader` は **到達不能**
（`rebuildJob` の代入箇所が src 全体で 0 件、`StateAndUI.cpp:988` の `reset()` のみ）であり、
増分 build の受け皿は現状 dormant である。

### 5.3 Option B の判定

**必要か／不要か**: 本 Pre-Audit では **不要** と判定する。理由:

- 入力読込は既にチャンク化されており、旧主張の「一括確保で OOM」は転送バッファについては解消済み。
- 残る #3 は「IR 全体をメモリに保持する」という**畳み込みの本質的要請**に由来し、
  streaming 化は DSP 側（incremental partition build）の設計変更を伴う。
- Option A で契約化できる余地が残っている（§4）ため、まず A で閉じるのが費用対効果で優位。

## 6. Pre-Audit 最終判定

```text
判定: A — CONTRACT-READY
```

**根拠**:

1. **allocation 前 reject の構造が既に存在する**（`:393-397` は `:422` より前）。
2. **allocation failure は制御されている**（`:1267 HeapBlock<char,true>` → `std::bad_alloc` →
   `performLoad:128` が捕捉 → graceful 失敗）。**未制御クラッシュ経路は確認できない**。
3. **int 溢れは無い**（JUCE `:392-395` が `(size_t)` キャスト後に乗算）。
4. **足りないのは契約値と明文化だけ**であり、実装構造の変更を要しない。

**ただし A と決める際の限定（Option A の構造が存在すること ≠ 現行値が契約として妥当であること）**:

```text
MAX_FILE_LENGTH = INT32_MAX を、そのまま製品契約値として採用することはできない。
理由:
  (i) sample 上限であって byte 上限ではない（channel 上限が無いため byte 上限は非有界）
  (ii) ステレオ double で 34.36 GB（実用コミット限界を超え、上限として無意味）
  (iii) N ≥ 3 は build 段で不使用（dead weight）であるため上限設定に設計上の損失がない
→ 契約値は §4.3 の (A-1)/(A-2)/(A-3) いずれかの形で Contract Decision において導出する。
```

**他判定の棄却理由**:

- **B — DESIGN-DEFERRED**: 読込の chunked 化は既に実装済み。残る #3 は DSP 側の設計変更を伴うため、
  「上限値を決められない」ことではなく「上限値の選択」の問題である。→ 棄却。
- **C — ACTUAL BUG**: allocation failure は graceful に処理され（§2.6）、int 溢れも無い（§2.6）。
  **到達可能な未制御ケースは確認できない**。→ 棄却。
- **D — CLOSED**: 既存 guard だけでは **byte 上限を規定できていない**（channel 上限が皆無）。
  → 棄却。

## 7. Failure Contract へ引き継ぐ未決事項

```text
FC-1  契約形の選択: (A-1) byte 上限 / (A-2) channel 上限併設 / (A-3) sample 上限引下げ
FC-2  契約値の決定: 具体値（kMaxIRLoadBytes / kN / 新 MAX_FILE_LENGTH）
FC-3  診断の整合:    errorMessage 文言（現行 "exceeds 2GB samples limit." は不正確）
FC-4  後方互換:      既存ユーザー IR ファイルが新上限で拒否されないことの確認方法
FC-5  明文化の範囲:  documentation のどこに記載するか（UI 変更は本件の scope 外）
FC-6  死重の扱い:    N ≥ 3 のチャンネルを (a) 上限で弾く / (b) ch0-ch1 のみ保持 のいずれか
FC-7  MAX_IR_LATENCY との非統合の明文化（§3 の分離を契約文書へ固定）
```

**本 Pre-Audit では上限値を実装しない。** 次工程は Failure Contract。

## 8. 監査メタ

```text
検証方法            : LoaderThread.cpp / LoadPipeline.cpp / Rebuild.cpp / ConvolverProcessor.h /
                      juce_AudioSampleBuffer.h の行単位実測
                      （allocation chain・gurad 位置・例外経路・narrowing 証明・到達性）
Production 変更     : 0
CMake 変更          : 0
UI 変更             : 0
tests 変更          : 0
ConvoPeq.md 再生成  : 0
commit / push       : 0

禁止事項の遵守:
  M-03 D3 実装なし / M-03 latency・PDC 変更なし / Direct Head HC/LC 変更なし /
  H-01・H-02 変更なし / SR-01〜03 変更なし / Publish・Crossfade・Retire authority 追加なし /
  Epoch・RuntimeWorld・Lifetime Budget 変更なし / RT path 変更なし /
  上限値の実装なし（Pre-Audit の範囲に留めた）
```

big 1-7（`emitRetireIntentNonRT` の自動 enforce）は本 work に含めない。
現状は「実呼び出し元 = 非 RT 1 系統 / 実欠陥 = なし / 自動 enforce = なし」であり、
**hardening backlog として記録維持**する（責務もリスクも big 1-8 と異なるため統合しない）。
