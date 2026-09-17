# WORK102-IG — Implementation Gate（read-only / design freeze）

- **作成日**: 2026-09-17
- **種別**: Implementation Gate（**コードを書かない**。順序・算術・全 loader path・テスト境界の凍結のみ）
- **目的**: R2 で凍結した契約を**変更せずに**現行ソースへ安全に実装できることを read-only で最終確認する
- **被監査対象**: `big18_pre_audit_*` → `big18_failure_contract_*` → `..._reconciliation_*` → `..._arbitration_*`（R2）
- **前例フォーマット**: `doc/work98/m02_pre_audit_failure_contract_freeze_20260915.md`

> **IG の結論（先出し）**
>
> - **判定: IMPLEMENTATION-READY**（§8）。R2 の契約値は一切変更しない。
> - **★ 指示書 §2 の ordering 図に実装不能な矛盾があるため訂正した**（§2.2）。
>   §2 は「trim 後 length に対する FC-FORM-5」を **byte admission より前** に置いているが、
>   `L_trim` は `loadedIR` を確保して trim した後にしか確定しない。
>   同一指示書 §3 と R2-D4（凍結）は「load → trim → FC-FORM-5 → resampler」であり、
>   **§3 / R2-D4 を正**として順序を確定した。契約値の変更はない。
> - §2 の絶対条件「byte admission より前に loadedIR / full-size temp / file-byte hash buffer を確保しない」は
>   **満たせることを確認**（§2.3）。
> - **R2 で未解決だった 2 件を IG で確定**: (i) 算術は除算形で溢れを構造的に排除（§4）、
>   (ii) FC-8 は標準 XXH64 streaming 化で digest 完全一致（§5）。
> - **テスト境界の実測上の制約を 1 件確定**: E/F（1 GiB 境界）は現実的な fixture では再現不可能。
>   → **admission predicate を純関数として抽出**し、境界は単体テストで凍結する（§7.4）。
> - Preview は §6 のとおり **WORK102-PREV-01 に固定**。IG の実装対象は main loader のみ。

---

## 0. 基本条件と Baseline

```text
read-only / production source = 0 / tests = 0 / CMake = 0 / UI = 0
ConvoPeq.md 再生成 = 0 / commit = 0 / push = 0
README.md E-G3-1 差分 = 維持

R2 の契約値（本 IG で変更しない）:
  kMaxIRLoadBytes             = 1,073,741,824 (1 GiB)
  kMaxIRLoadChannels          = 8
  IR file SR envelope         = 44.1 kHz – 768 kHz
  MAX_FILE_LENGTH             = 2,147,483,647 (INT32_MAX)
  kMaxIRResampleOutputSamples = 2,097,153 (U + 1)
  simultaneous residency      = ≤ 2 GiB (2 × kMaxIRLoadBytes)
  FC-8                        = (b) streaming hash
  Preview                     = WORK102-PREV-01（分離維持）

Baseline:
  HEAD = origin/main = 35461e2e
  ConvoPeq.md mtime = 2026-09-17 00:07:39 JST（FRESH。再生成しない）
```

**参照した凍結文書**: R2 `big18_failure_contract_arbitration_20260917.md` /
SR-01B `sr01b_implementation_gate_contract_freeze_20260914.md` /
`doc/Practical Stable ISR Bridge Runtime.md`（RT/NonRT 分離原則）

---

## 1. 一次ソースの再確認（実コード行まで追跡）

`src/convolver/ConvolverProcessor.LoaderThread.cpp::doLoadIRStep` を一次ソースとして追跡した。

| # | 対象 | 実測位置 | 実測内容 |
|---|---|---|---|
| 1 | `doLoadIRStep()` | `:360-455` | `LoadIR` ステップ本体。`Trim → Transform → Build` は `stepOnce():329-358` |
| 2 | `computeIRHash()` | 呼出 `:363-364` / 実体 `AllpassDesigner.cpp:613-660` | `:364` で **admission より前**に実行。`fileData.malloc(fileSize)`（`:629`）が O(file bytes) |
| 3 | `AudioFormatReader` 生成 | `:380-382` | `formatManager.registerBasicFormats()` + `createReaderFor(file)`。WAV/AIFF/FLAC（`juce_AudioFormatManager.cpp:63-71`）はいずれもストリーミング reader で、**ヘッダ読込のみ**＝ O(file) 確保なし |
| 4 | `fileLength / numChannels` | `:389-390` | `fileLength` は `int64`（`juce_AudioFormatReader.h:240`）、`numChannels` は **`unsigned int`**（同 `:243`）→ `static_cast<int>` で narrowing |
| 5 | existing admission | `:391-397` / `:398-402` | `MAX_FILE_LENGTH = 2147483647` / `numChannels <= 0` |
| 6 | trim | `:465-517` | 末尾から `threshold = 1.0e-15` で走査。**判定に使うのは ch0 と ch1 のみ**（`:474-475`）。`newLength = 0` なら `max(1, newLength)` により 1 サンプルへ縮小 |
| 7 | `setSize(..., true)` | `:514` | `stepResult.loadedIR.setSize(numChannels, std::max(1, newLength), true)`。JUCE は **新規確保 → copy → swapWith**（`juce_AudioSampleBuffer.h:397-431`）→ 旧+新が同居 |
| 8 | `shrinkToFit()` | `:515` → `ConvolverProcessor.Internal.h:51-61` | `juce::AudioBuffer<double> newBuffer(...)` を**新規作成して copy**（`:56-60`）→ 同一サイズの複製が一時発生 |
| 9 | resample | `:519-553`（呼出 `:527-532`） | `loadedSR != sampleRate` のときのみ。入力は **trim 後の `loadedIR`** |
| 10 | build | `doBuildStep():679-682` → `buildConvolverFromTrimmed():149-195` | ch0/ch1 のみ使用（`:166-167`） |
| 11 | preview path | `ResampleAndFallback.cpp:271-331` ← `StateAndUI.cpp:455` ← `ConvolverControlPanel.cpp:1135-1168` | `:293` `maxFileLength` / `:301` `numChannels <= 0` のみ。`:307` に**非チャンクの全長 float バッファ**。**§6 のとおり IG 対象外** |

**確認結果**: `hash → admission → allocation` の現行順序（R1/R2 の指摘）は一次ソースで再確認された。
**FC-8 の実装対象であることは明白**である（`fileData` の全長確保とその呼出位置）。

---

## 2. Admission ordering の実装固定

### 2.1 ★ 指示書 §2 の ordering 図の矛盾と訂正

指示書 §2 に示された順序は次のとおり:

```text
（指示書 §2 の記載）
  ... channel <= 8
      ↓
  trim後 length に対する FC-FORM-5     ← ★
      ↓
  byte admission <= 1 GiB
      ↓
  O(file bytes) / O(fileLength) allocation
      ↓
  loadedIR
```

**これは実装不能である。** 理由:

```text
L_trim（trim 後 length）は、loadedIR を確保してチャンク読込を完了し、
末尾無音トリム（:465-517）を実行した後にしか確定しない。
したがって FC-FORM-5 を loadedIR の確保より前に置くことはできない。
```

同一指示書の §3 は正しい順序を示している:

```text
（指示書 §3 の記載）
  load → trim → FC-FORM-5 → resampler construction
```

R2-D4（凍結）も「trim 後・resampler 構築前」と確定している。

**→ IG は §3 と R2-D4 を正とする。§2 の図はドラフト上の記載位置の誤りであり、
契約値の変更ではない。**（R2 の契約値は一切動かさない）

### 2.2 凍結する最終 ordering（実装順序）

```text
 1. formatManager.registerBasicFormats() / createReaderFor(file)     :380-382
 2. fileLength = reader->lengthInSamples                             :389
    numChannels = reader->numChannels                                :390
 3. FC-FORM-4  degenerate check                                      :398-402（既存維持）
    ※ unsigned 値で判定する（§4.3）
 4. FC-FORM-3  INT32_MAX representation check                        :391-397（既存維持）
 5. FC-FORM-2  channel admission  (numChannels <= 8)                ★新規
 6. FC-FORM-1  byte admission     (<= 1 GiB)                        ★新規
 7. ★ ここまで O(fileLength) / O(file bytes) の確保は 0
 8. FC-FORM-6  computeIRHash（O(1) streaming 化済）                  ★呼出を :364 からここへ移動
 9. tempFloatBuffer / tempAligned（チャンクサイズ。全長ではない）    :414-415
10. loadedIR.setSize(numChannels, (int)fileLength)                  :422
11. チャンク読込ループ（kStreamChunk = 262,144）                     :425-450
12. 末尾無音トリム → L_trim = loadedIR.getNumSamples()               :465-517
13. ★ FC-FORM-5  resample output admission（L_trim に対して）        :517 と :519 の間
14. resampler 構築 / r8b getMaxOutLen / (int) 変換                   :527
15. build                                                            :679-682
```

**順序の根拠**: FC-FORM-5 だけが「loadedIR を必要とする」ため後置される。他の 4 つは
`reader` のメタデータのみで判定でき、**確保より前に置ける**。

### 2.3 §2 の絶対条件の検証

```text
絶対条件: 「byte admission より前に loadedIR、full-size temp、file-byte hash buffer を確保しない」
```

| 対象 | 現行 | IG 後 | 判定 |
|---|---|---|---|
| `loadedIR`（O(fileLength)） | `:422`。admission は `:391-402` のみ（byte なし） | ⑩ で確保。⑦（FC-FORM-1）が先行 | **満たす** |
| full-size temp（O(fileLength)） | main loader には**存在しない**。`tempFloatBuffer` は `N × 262,144`（`:414`）、`tempAligned` は `262,144 × 8`（`:415`）＝ チャンクサイズ | 同じ | **満たす**（現行で既に満たしている） |
| file-byte hash buffer | `AllpassDesigner.cpp:629` の `fileData.malloc(fileSize)` が `:364` で **確保される** | FC-8 (b) で**消滅**（O(1)） | **満たす**（§5 で設計確定） |

**補足**: 「full-size temp」が存在するのは **preview 経路のみ**（`ResampleAndFallback.cpp:307`）。
当該経路は WORK102-PREV-01（§6）に固定されるため、main loader の絶対条件には影響しない。

### 2.4 実装対象ファイル（IG の変更面）

```text
IS-1  src/convolver/ConvolverProcessor.LoaderThread.cpp
        - admission ブロック追加（FC-FORM-2 → FC-FORM-1）
        - computeIRHash 呼出の移動（:363-364 → admission 後）
        - FC-FORM-5 の追加（:517 と :519 の間）
IS-2  src/AllpassDesigner.cpp
        - xxh64Digest の streaming 化（computeIRHash の本体）
        - 全長 staging buffer（HeapBlock<uint8_t> fileData）の除去
        ※ 公開シグネチャ computeIRHash(const File&, bool) は不変
IS-3  ★新規 src/convolver/IRLoadAdmission.h（純関数・契約定数の単一情報源）
        - kMaxIRLoadBytes / kMaxIRLoadChannels / kMaxIRResampleOutputSamples
        - FC-FORM-1/2/3/5 の predicate（テスト可能・I/O 非依存）
        - 診断文言の生成（FC-INV-8 の単一情報源）
IS-4  tools/check-loader-admission-order.py（任意）
        - doLoadIRStep 内の admission 呼出行 < allocation 行 を AST/行順で機械検証
        - 既存の tools/check-*.py 前例に倣う
IS-5  src/tests/AudioEngineHarness/ に新規 TU + CMakeLists 登録（§7）
```

**禁止**: IS-3 で新しい契約値を導入しないこと（定数は R2 の 4 値のみ）。
IS-3 は**テスト可能性のための抽出**であり、値の追加ではない。

---

## 3. FC-FORM-5 の trim 後適用をコード位置まで確定

### 3.1 適用点

```text
:465-517  if (stepResult.loadedIR.getNumSamples() > 0) { ... newLength 算出 ... }
:512-516      if (newLength < numSamples)
:514              stepResult.loadedIR.setSize(numChannels, std::max(1, newLength), true);
:515              ConvolverProcessorInternal::shrinkToFit(stepResult.loadedIR);
:517  }        ← ★ FC-FORM-5 はこの直後に置く
:519  if (stepResult.loadedSR > 0.0 && sampleRate > 0.0 && std::abs(...) > 1e-6)
:527      auto resampleOut = ConvolverProcessorInternal::resampleIR(...)
```

**`L_trim` の定義**:

```text
L_trim = stepResult.loadedIR.getNumSamples()  （:517 時点）
       = min(numSamples, max(1, newLength))
  - newLength = 「ch0 または ch1 の |値| > 1.0e-15 を満たす最後の index」+ 1
  - 該当サンプルが無い場合 newLength = 0 → max(1,0) = 1
  → L_trim ∈ [1, fileLength] が常に成立（trim は単調減少のみ）
```

**判定式（FC-INV-10 の overflow-safe 形）**:

```text
skip 条件: loadedSR <= 0.0 ∨ sampleRate <= 0.0     （:519-520 の resample ガードと同一）
判定:      (double)L_trim / loadedSR  >  (double)kMaxIRResampleOutputSamples / sampleRate
           → reject（FC-FORM-5）
```

**`raw fileLength` では評価しない**（R2-D4）。raw に適用すると
「長い無音テールを持つファイル」を UI が受理するのに loader が拒否する回帰が生じる（R2 §6.2）。

**リサンプルが実行されない場合（`loadedSR == sampleRate`）も FC-FORM-5 は評価する。**
上の式はその場合 `L_trim > kMaxIRResampleOutputSamples` に簡約される。
これは UI の `exceedsHardLimit`（`StateAndUI.cpp:540`）が処理レートのサンプル数で判定するのと同義であり、
`hardMaxSec(sr)` の意味と一致する。skip するのは `loadedSR <= 0.0 ∨ sampleRate <= 0.0` のときのみ。

### 3.2 trim phase の 2 GiB と矛盾しないことの確認

R2 §9.3 の位相 B ピーク **2 × C_src = 2 GiB** は次の 2 つの複製に由来する:

| 複製 | 実装 | 実測 |
|---|---|---|
| `setSize(..., keepExistingContent=true)` | 新規 `HeapBlock` を確保 → copy → `allocatedData.swapWith(newData)`（`juce_AudioSampleBuffer.h:397-431`） | 旧 + 新 が同居 |
| `shrinkToFit` | `juce::AudioBuffer<double> newBuffer(N, numSamples)` を新規作成 → copy → move 代入（`Internal.h:56-60`） | 新バッファ（同一サイズ）が同居 |

**FC-FORM-5 を `:517` に置いても、この 2 つの複製は変化しない。**
FC-FORM-5 は**読取のみの判定**であり、確保・解放を追加しない。
したがって **IG 後も trim phase のピークは 2 × C_src = 2 GiB のまま**で、
R2 §9.3 の結論と矛盾しない。✓

（位相 B を 1 GiB 側へ下げる `avoidReallocating=true` の検討は R2 IG-5 / IS-7 の任意候補であり、
**本 IG の実装対象に含めない**。R2 の simultaneous residency 値 2 GiB は不変。）

---

## 4. Arithmetic / overflow gate

### 4.1 無条件の乗算実装を禁止する理由（実測）

```text
NG: const int64_t bytes = numChannels * fileLength * sizeof(double);   // ← 型が int なら溢れる
    （numChannels が int、fileLength が int64 の混在式では int64 へ昇格するが、
      途中で int の中間値を作る書き方は禁止）
実測: int32 では 8 × 2,147,483,647 = 17,179,869,176 > INT32_MAX (2,147,483,647) → **溢れる**
```

### 4.2 推奨形（除算・溢れを構造的に排除）

```text
★ FC-FORM-2（numChannels <= 8）を FC-FORM-1 より先に評価する。

  const int64_t denom = static_cast<int64_t>(numChannels) * static_cast<int64_t>(sizeof(double));
      // numChannels ∈ [1, 8] → denom ∈ [8, 64]。乗算は 64 以下で自明に安全
  const int64_t maxSamplesPerChannel = static_cast<int64_t>(kMaxIRLoadBytes) / denom;
  if (static_cast<int64_t>(fileLength) > maxSamplesPerChannel)  → reject (FC-FORM-1)

  → 被除算は定数 kMaxIRLoadBytes（1 GiB）のみ。**未検証値の乗算が 1 回も現れない。**
```

### 4.3 先行 admission による安全域の証明（ユーザー要求）

| 前段の admission | 保証 | 後段の積への効果 |
|---|---|---|
| FC-FORM-4: `numChannels ≥ 1` | N ∈ [1, …] | 除算の分母 ≥ 8（0 除算なし） |
| FC-FORM-3: `fileLength ≤ INT32_MAX` | L ≤ 2,147,483,647 | — |
| FC-FORM-2: `numChannels ≤ 8` | N ≤ 8 | **積の上限 = 8 × 2,147,483,647 × 8 = 137,438,953,472（38 bit）** |

```text
→ 8 × INT32_MAX × 8 = 137,438,953,472 < 2^38 ≪ INT64_MAX (9,223,372,036,854,775,807)
→ 64-bit 型（int64/uint64/size_t）であれば中間値も含めて溢れは発生しない。
→ 32-bit int は不可（上記のとおり 17.2e9 で溢れる）。**型は 64-bit 必須。**
```

**等価な代替形**（乗算形を使う場合）:

```text
  const int64_t bytes = static_cast<int64_t>(numChannels)
                      * static_cast<int64_t>(fileLength)
                      * static_cast<int64_t>(sizeof(double));
  if (bytes > static_cast<int64_t>(kMaxIRLoadBytes)) → reject
  ※ この形が安全なのは FC-FORM-2（N ≤ 8）と FC-FORM-3（L ≤ INT32_MAX）が
     **必ず先行する**場合に限る。順序を入れ替えてはならない。
```

### 4.4 `numChannels` の narrowing（IG で扱う）

```text
実測: juce::AudioFormatReader::numChannels は unsigned int（juce_AudioFormatReader.h:243）。
      現行 :390 は static_cast<int>(reader->numChannels)。

リスク: unsigned で INT_MAX 超の値（2,147,483,648–4,294,967,295）は
        int へ narrowing すると実装定義（実務上は負値）→ 既存の `numChannels <= 0` が捕捉する。

IG の扱い: FC-FORM-2 の判定は **narrowing 前の unsigned 値**で行う
           （`static_cast<unsigned>(reader->numChannels) > kMaxIRLoadChannels` → reject）。
           これにより「負値に化けた値が偶然 K 以下になる」経路を型レベルで排除する。
           既存の FC-FORM-4（`<= 0`）は維持し、判定材料として二重化しない。
```

### 4.5 FC-FORM-5 の算術（再掲）

```text
dividend は (double)L_trim、divisor は (double)loadedSR。
比較対象は (double)kMaxIRResampleOutputSamples / (double)sampleRate。
→ すべて double。int 中間値を作らない。
→ fileSR が極小（例 1e-6）でも double の範囲で巨大値になり、確実に reject される。
→ r8b の (int) ceil(MaxInLen × DstSR / SrcSR) + 1（CDSPFracInterpolator.h:831）に
   到達する前に拒否されるため、N=1 極長で 2,337,397,169 > INT32_MAX となる溢れは発生しない。
```

---

## 5. FC-8（streaming hash）の最終設計

### 5.1 O(1) auxiliary memory の確認（要求 1）

現行実装（`AllpassDesigner.cpp:613-660`）は
`HeapBlock<uint8_t> fileData; fileData.malloc(fileSize);`（`:626-629`）で
**物理ファイルサイズ比例の確保**を行う。

```text
IG 後の state:
  uint64_t v1, v2, v3, v4;   // 32 B
  uint64_t totalLen;         //  8 B
  uint64_t seed;             //  8 B
  uint8_t  buf[32];          // 32 B
  size_t   bufLen;           //  8 B
  → 合計 ≒ 88 B（スタック）。**fileSize に依存しない = O(1)。✓**
```

### 5.2 既存 hash 値が変わらないこと（要求 2）

現行 `xxh64Digest`（`AllpassDesigner.cpp:142-202`）は **標準 XXH64** の構造をとっている:

```text
len >= 32 : v1..v4 を seed ± prime で初期化 → 32 B ストライプで xxh64Round
            → h = rotl64(v1,1)+rotl64(v2,7)+rotl64(v3,12)+rotl64(v4,18)
            → xxh64MergeRound × 4
len <  32 : h = seed + kPrime5
h += len
tail: 8 B ループ → 4 B（1 回）→ 1 B ループ
最後に xxh64Avalanche
```

**これは XXH64 の規格そのもの**（prime 定数・rotate 量・merge 順序・avalanche が一致）であるため、
標準のストリーミング state machine は**全入力について同一 digest を返す**。

**注意点（実装時に厳守）**:
- `readLE64` / `readLE32` は `std::memcpy` ベース（`AllpassDesigner.cpp:97-109`）＝**整列非依存**。
  streaming の 32 B staging buffer は任意整列で問題ない。✓
- seed は `kHashVersionSalt = 0x434f4e564f504551`（`："CONVOPEQ"`）を維持する。**変更禁止**。
- `h += totalLen` を維持する（`totalLen` は読込済み総バイト数）。
- `len < 32` の分岐は **`totalLen < 32`** で判定する（one-shot の `len` と同義）。

### 5.3 `size + mtime` による既存 TOCTOU 検証との整合（要求 3）

```text
現行: sizeBefore / mtimeBefore を取得（:617-618）→ 読込 → :655 で sizeAfter / mtimeAfter を再取得し
      「sizeBefore != sizeAfter ∨ mtimeBefore != mtimeAfter」なら return 0。
      buffer は「hash 対象のスナップショット」を作るためだけに存在し、TOCTOU 保証は size+mtime 検証が担う。

IG 後: 同じ before/after 検証をそのまま維持する。
      ストリーミング hash は読みながら逐次 hash するため buffer は不要。
      読込中にファイルが変化した場合、size+mtime 検証が同じ条件で 0 を返す（保証同一）。
      ※ 現行が writePos != fileSize で検出している「読込バイト数不一致」は、
        totalLen != sizeBefore として同値に検出する。
```

### 5.4 cancellation / failure semantics を変えないこと（要求 4）

```text
現行の return 0 条件（すべて維持）:
  - !irFile.existsAsFile()
  - stream == nullptr（createInputStream 失敗）
  - stream->getStatus().failed()
  - writePos != fileSize          → totalLen != sizeBefore
  - size/mtime 変化（TOCTOU）

現行は **cancellation を一切サポートしていない**（threadShouldExit を見ない）。
→ IG では cancellation を**追加しない**。追加すると semantics 変更になるため。
  （逐次 hash は無停止で走る。時間は file size に比例するが、メモリは O(1)。）
```

### 5.5 hash が admission を迂回しないこと（要求 5）

```text
現行: :363-364 で computeIRHash が admission（:391-402）より前に実行される → FC-INV-1 FAIL。

IG 後: 呼出を admission ブロック（FC-FORM-1〜4）の**後**、:414 より前に移す（§2.2 の ⑦→⑧）。
       stepFileHash の唯一の消費者は :651（convertToMixedPhase の cache key）であり、
       同一ステップ内で後置可能。**観測可能な挙動の変化はない。**
       副次効果: admission で拒否されるファイルに対して hash を計算しなくなる（時間削減、観測不能）。
```

---

## 6. Preview の分離維持

### 6.1 境界（R2-D6 を維持しつつ、IG の実装対象を確定）

```text
本 IG の実装対象: main loader（ConvolverProcessor::loadImpulseResponse / LoaderThread）**のみ**

  - preview の問題を修正しない
  - preview のために UI を変更しない
  - main loader の契約を preview の**実装対象**へ拡張しない
```

**R2-D6 との整合（明示）**: R2-D6 は「preview 経路の admission（FC-FORM-1/2/3/4/5）は
main contract と同一に適用」と記述した。本 IG はその**要求を WORK102-PREV-01 の受入基準へ移管**する。
すなわち:

```text
WORK102-PREV-01   Preview allocation failure graceful completion
                  ＋（IG で受入基準に追加）preview admission parity
                  = (i)  preview 経路に FC-FORM-1/2/3/4/5 を適用
                    (ii) job 例外時に preview が必ず終端状態へ到達し、
                         irPreviewInProgress が確実に false へ戻る

  → R2 の契約記述は変更しない。IG の**実装面と受入面**から preview を外すだけである。
  → FAIL は消さない。WORK102-PREV-01 は OPEN のまま IG に引き継がれる。
```

### 6.2 明示的に否定する扱い

```text
「Preview は未解決なので main loader の実装は安全」という扱いを**しない**。
  - main loader の安全性は §2〜§5 の順序・算術・FC-8 によって独立に成立する。
  - preview の OPEN は main loader の安全性の根拠でも、その欠落の免罪でもない。
  - preview の admission 不在（`:293`/`:301` のみ）と graceful failure 不在は
    WORK102-PREV-01 の受入基準として残る。
```

---

## 7. テスト設計の凍結

### 7.1 harness の実測（実装可能性の確認）

| 能力 | 実測 | 出典 |
|---|---|---|
| 任意 SR / 任意 ch / 任意長の WAV を書ける | `writeSR01TempIr(tag, sampleRate, numCh, totalSamples, peakPos)` | `ConvolverStateRoundTripTests.cpp:953-986` |
| processing SR を設定できる | `conv.prepareToPlay(768000.0, 512)` の前例あり | 同 `:1064, :1116, :1144` |
| ロード完了を待てる | `pollH01Peak` / `!conv.isLoadingIR()` ポーリング | 同 `:710, :754, :1332` |
| errorMessage を検証できる | `conv.getLastError()` | `ConvolverProcessor.h:452` |
| テスト登録 | `runConvolverStateRoundTripTests()` を宣言し main から呼ぶ | `PublishPipelineIntegrationTests.cpp:59, :1282` |

### 7.2 テストケース（A〜L）

| ID | 条件 | fixture | 期待 | 検証方法 |
|---|---|---|---|---|
| **A** | 1ch / byte limit 以下 | `writeSR01TempIr(48000, 1, 4800, 100)` | **accept** | `getLastError()` 空 ∧ `isIRLoaded()` |
| **B** | 2ch / byte limit 以下 | `writeSR01TempIr(48000, 2, 4800, 100)` | **accept** | 同上 |
| **C** | 8ch / byte limit 以下 | `writeSR01TempIr(48000, 8, 4800, 100)` | **accept** | 同上。**同時に SR-01B「>2ch→先頭2」の回帰**（ch0/ch1 のみが build に使われること） |
| **D** | 9ch | `writeSR01TempIr(48000, 9, 4800, 100)` | **deterministic reject** | `getLastError()` が channel 文言（actual=9 / limit=8）を含む。**fixture は小さい** |
| **E** | byte limit + 1 | **(N, L) の合成入力**（fixture 不可。§7.4） | **reject before allocation** | 純関数 predicate の単体テスト |
| **F** | `INT32_MAX + 1` | **(N, L) の合成入力** | **reject before allocation** | 同上 |
| **G** | raw は巨大、trim 後は許容 | `48000` SR・2ch・**2,400,000 サンプル（50 s）**、ピークを index 1000 に置き以降は無音 | **accept** | §7.3 参照。**raw 適用なら拒否される長さ**であることが本質 |
| **H** | trim 後に FC-FORM-5 超過 | `writeSR01TempIr(44100, 2, 132300, 100)`（3 s、末尾無音なし）＋ `prepareToPlay(768000.0, 512)` | **reject（FC-FORM-5）** | `getLastError()` に resample 文言。L_res = 2,304,001 > U+1 |
| **I** | hash 対象が巨大 | (i) 固定 fixture に対する golden digest | (i) **既存値と一致** (ii) **O(1) memory** | (i) は golden 値の凍結。(ii) は構造検証（§7.5） |
| **J** | allocation failure | 決定論的に発生させられない | **graceful error** | 構造検証（§7.5） |
| **K** | cancellation during chunk read | — | **existing semantics** | 構造検証（§7.5） |
| **L** | resample required | `writeSR01TempIr(44100, 2, 44100, 100)` ＋ `prepareToPlay(48000.0, 512)` | **contract-compatible** | accept ∧ `L_res = ceil(44100 × 48000/44100) + 1 = 48,001 ≤ U+1` |

### 7.3 ケース G の設計根拠（R2-D4 の核心）

```text
sr = 48000 → hardMaxSec = 2,097,152 / 48,000 = 43.69 s
fixture の duration = 2,400,000 / 48,000 = 50.0 s  > 43.69 s

  raw fileLength に FC-FORM-5 を適用 → duration 50 s > 43.69 s → **拒否（誤り）**
  L_trim に適用                     → L_trim ≒ 1,000 → duration 0.021 s → **受理（正しい）**

かつ FC-FORM-1: 2 × 2,400,000 × 8 = 38,400,000 B ≪ 1 GiB → PASS
→ 本ケースは「raw 適用の regression を検出する回帰テスト」として機能する。
```

### 7.4 ケース E / F の実装可能性と predicate 抽出（IG の設計判断）

```text
実測上の制約:
  E（byte limit + 1）を実ファイルで作るには
    N=8 → L = 16,777,217 サンプル（8ch float32 WAV ≒ 537 MB）
    N=2 → L = 67,108,865 サンプル（2ch float32 WAV ≒ 537 MB）
  いずれも AudioBuffer をメモリに構築して書き出す必要があり、
  既定の ctest に常設する fixture としては不適切（時間・容量・CI 安定性）。

IG の決定:
  FC-FORM-1/3 の境界判定を **純関数** として抽出する（IS-3: IRLoadAdmission.h）。
    bool irLoadAdmitBytes(int numChannels, int64_t fileLength);   // overflow-free（§4.2 の除算形）
    bool irLoadAdmitChannels(unsigned numChannels);
  単体テストで (N, L) を合成注入し、境界（limit ちょうど / limit+1 / INT32_MAX / INT32_MAX+1）を固定する。
  → 実ファイル fixture を必要とせず、E/F を決定論的に凍結できる。
  → 併せて §4 の overflow-safe 算術そのものを単体テストで固定する。

  ※ 統合テストとしての E/F は本 IG の受入に含めない（fixture 非現実性のため）。
     必要なら opt-in（既定 ctest 外）の重量テストとして別途起票する。
```

### 7.5 構造検証（J / K / I(ii)）— 決定論的テストが不能な項目

```text
J（allocation failure → graceful）
  決定論的に bad_alloc を発生させる手段（fault injection）が現行 harness に無い。
  → 検証: ConvolverProcessor::LoaderThread::performLoad の catch 群
          （:128 bad_alloc / :133 std::exception / :138 ...）が変更されていないこと、
          および errorMessage の割当が維持されていることを**差分スコープ**で確認する。
          （現行仕様の維持が要件であり、新規の分岐は追加しない）

K（cancellation during chunk read）
  externalCancellationCheck（LoaderThreadInline.h:14）は loader 内部からのみ設定され、
  loadImpulseResponse 経由ではテストから注入できない（既存 test に前例 0 件）。
  → 検証: :427-431 のチャンク毎 cancellation と errorMessage "IR loading cancelled." が
          変更されていないことを差分スコープで確認する。

I(ii)（hash の O(1) メモリ）
  → 検証: computeIRHash 内に物理ファイルサイズ比例の確保が残っていないことを
          ソース検査で確認する（HeapBlock<uint8_t> / std::vector<uint8_t> の fileSize 依存確保が 0 件）。
          tools/check-loader-admission-order.py（IS-4）に併せて機械検証を追加してよい。
```

### 7.6 Debug / Release 差の扱い

```text
work98 §8-6 の前例（Debug/Release で発火経路が異なる）を踏まえ、
本 IG の A〜L は **両構成で同一の判定**となるよう設計する:
  - A〜D / G / H / L は admission と errorMessage のみを検証 → 構成非依存
  - E / F は純関数単体テスト → 構成非依存
  - I(i) は golden digest → 構成非依存（xxh64Digest は noexcept な純計算）
  → 片方の構成でのみ green になるケースを作らない。
```

---

## 8. IG 最終判定

```text
判定: IMPLEMENTATION-READY
```

### 8.1 ユーザー提示の IMPLEMENTATION-READY 条件

| 条件 | 状態 | 根拠 |
|---|---|---|
| R2-D2/D3/D4/D5 に矛盾なし | **✓** | §2.2 の順序は R2-D4（trim 後適用）と一致。契約値は 1 GiB / 8 / U+1 / envelope 44.1k–768k をそのまま使用。指示書 §2 の図を §3 に合わせて訂正 |
| admission ordering 完全固定 | **✓** | §2.2 の 15 段。FC-FORM-4→3→2→1→(allocate)→6→5 の順序と行位置を確定 |
| FC-FORM-5 の適用点固定 | **✓** | §3.1: `:517` と `:519` の間、`L_trim = loadedIR.getNumSamples()`。skip 条件も確定 |
| overflow-safe arithmetic 固定 | **✓** | §4.2 の除算形を主形として確定。§4.3 で N ≤ 8 + L ≤ INT32_MAX ⇒ 積 ≤ 2^38 を証明。§4.4 で unsigned narrowing を確定 |
| FC-8 の実装方式固定 | **✓** | §5: XXH64 streaming state、digest 完全一致の根拠、TOCTOU 維持、cancellation 非追加、呼出位置移動 |
| Preview の分離維持 | **✓** | §6: 実装面・受入面から除外。R2-D6 の要求は WORK102-PREV-01 の受入基準へ移管。FAIL は OPEN 維持 |
| compatibility change が R2 契約と一致 | **✓** | CC-1（N>8）/ CC-2（N≥4 高 SR）/ CC-4（duration > hardMax、CLI・state 復元のみ）を §7 のテスト D/H/G が直接検証 |
| テストケースを実装可能な粒度で固定 | **✓** | §7.2 の A〜L。fixture・期待・検証方法・実装可能性を確定。E/F は純関数化（§7.4）、J/K/I(ii) は構造検証（§7.5）として明示分離 |

### 8.2 他判定の棄却

- **HOLD**: 実装不能な契約は 0。唯一の実装上の障害（指示書 §2 の ordering 矛盾）は
  §2.1 で §3 / R2-D4 を正として解決済みであり、契約値は動かしていない。
  E/F の fixture 非現実性は §7.4 の predicate 抽出で解消済み。→ 棄却。

### 8.3 IG 完了時に持ち越す項目

```text
carry-1  IG-1〜IG-3（R2 の countersign）: R2-D2 envelope / R2-D3 SR-01B supersede /
         CC-1・CC-2・CC-4 の承認。**R2 の governance 前提であり、IG 判定とは独立。**
carry-2  WORK102-PREV-01（OPEN）: preview の admission parity + graceful completion。
carry-3  IS-7（任意）: trim の copy-on-resize を avoidReallocating で回避し
         whole-path ピークを 2 GiB → ~1.5 GiB。R2 の契約値は変更しない。
carry-4  E/F の重量統合テスト（opt-in）を起票するかどうかは IG 後の判断。
```

**次工程は WORK102-IMPL（実装）。本 IG はここで停止する（コードを書かない）。**

---

## 9. 監査メタ

```text
検証方法    : ConvoPeq.md（一次ソース）+ src の行単位実測
              LoaderThread.cpp / LoaderThreadInline.h / ResampleAndFallback.cpp /
              AllpassDesigner.cpp / ConvolverProcessor.Internal.h / ConvolverProcessor.h /
              LoadPipeline.cpp / StateAndUI.cpp / ConvolverControlPanel.cpp /
              juce_AudioFormatReader.h / juce_AudioSampleBuffer.h /
              r8brain-free-src/CDSPFracInterpolator.h /
              src/tests/AudioEngineHarness/（harness 能力の実測）
              + 算術の上限証明 + fixture 実現可能性の評価

Production 変更     : 0
CMake 変更          : 0
UI 変更             : 0
tests 変更          : 0
ConvoPeq.md 再生成  : 0（FRESH 維持）
commit / push       : 0
README E-G3-1 差分  : 維持

R2 契約値の変更     : 0（kMaxIRLoadBytes / kMaxIRLoadChannels / envelope /
                      MAX_FILE_LENGTH / kMaxIRResampleOutputSamples / simultaneous residency /
                      FC-8 方式 / Preview 分離 — すべて不変）
```

### 本 IG で確定した新規事項（要約）

| # | 事項 | 種別 |
|---|---|---|
| 1 | 指示書 §2 の ordering 図は実装不能（`L_trim` は loadedIR 後）。§3 / R2-D4 を正とする | 矛盾の解決（契約値の変更なし） |
| 2 | admission 順序を 15 段で固定（FC-FORM-4→3→2→1→allocate→6→5） | 実装設計 |
| 3 | FC-FORM-1 は除算形で実装（未検証値の乗算を 0 回にする） | 算術設計 |
| 4 | `numChannels` の channel 判定は narrowing 前の unsigned 値で行う | 算術設計 |
| 5 | XXH64 streaming は digest 完全一致（標準規格と一致しているため） | FC-8 設計 |
| 6 | E/F は fixture 非現実性のため純関数 predicate の単体テストで凍結 | テスト設計 |
| 7 | J / K / I(ii) は決定論的テスト不能 → 構造検証として明示分離 | テスト設計 |
| 8 | IS-3（IRLoadAdmission.h）を新設しテスト可能性と単一情報源を両立 | 実装面 |
