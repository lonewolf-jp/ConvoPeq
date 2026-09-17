# WORK102-FC — big 1-8 Failure Contract Freeze（read-only / contract only）

- **作成日**: 2026-09-17
- **種別**: Failure Contract（契約凍結。**実装は行わない**）
- **前工程**: `doc/work102/big18_pre_audit_bounded_allocation_contract_20260917.md`
  （判定 **A — CONTRACT-READY**。本 work はその §7 未決事項 FC-1〜FC-7 を閉じる）
- **前例フォーマット**: `doc/work98/m02_pre_audit_failure_contract_freeze_20260915.md`
  （Source Trace → 隣接欠陥の分離 → 契約凍結 → 出口判定 → Implementation Gate 条件）
- **位置付け**: 「未修正バグの修正」ではない。**現行 Loader を維持したまま、
  何を上限とし、どの時点で reject し、何を診断するかを凍結する**契約作業。

> **1 件の Pre-Audit 是正を含む**（§2.3 / FC-8）。
> Pre-Audit §2.6 の「クラッシュ経路は確認できない」は **不成立** である。
> `computeIRHash` に admission 前の O(file bytes) 確保と、失敗時の null 経路がある（実測）。
> ただし Pre-Audit の **判定 A（CONTRACT-READY）自体は維持** する（後述 §2.3 の理由）。

---

## 1. Baseline

```text
HEAD        = 35461e2e3a5fd42fd009416a20adaf457b9b4031
origin/main = 35461e2e3a5fd42fd009416a20adaf457b9b4031   (ahead/behind = 0/0)

Source（authoritative）:
  ConvoPeq.md  Generated = 2026-09-17 00:07:34
  --check      NEWER_SRC_COUNT = 0 / STATUS = FRESH   ★ 本 work で再実測（下記）

再実測（本 work）:
  ConvoPeq.md mtime = 2026-09-17 00:07:39 JST
  src/**/*.{c,cpp,h,hpp,txt} のうち ConvoPeq.md より新しいもの = 0 件
  => FRESH を維持。巻き戻しなし（ユーザー指示どおり現状維持）

README.md E-G3-1 差分: 維持（本 work で触らない）

本 work の変更:
  doc/work102/big18_failure_contract_20260917.md の新規作成のみ
  production source = 0 / tests = 0 / CMake = 0 / UI = 0
  ConvoPeq.md 再生成 = 0 / commit = 0 / push = 0
```

**scope 外（本 work では触らない）**: M-03 D3 / M-03 latency・PDC / Direct Head HC/LC /
H-01 / H-02 / SR-01〜03 / Publish authority / Crossfade authority / Retire authority /
Epoch / RuntimeWorld / Lifetime Budget / RT path。big 1-7（`emitRetireIntentNonRT` 自動 enforce）も
引き続き本 work に含めない（hardening backlog として維持）。

---

## 2. Pre-Audit（A — CONTRACT-READY）の引継ぎ

### 2.1 引継ぎ事項（確定として受け取る）

| # | 内容 | 出典 |
|---|---|---|
| P1 | allocation 前 reject の**構造は既に存在**する（`:393-397` は `:422` より前） | Pre-Audit §4.1 |
| P2 | int 溢れなし（JUCE `AudioBuffer::setSize` は `(size_t)` キャスト後に乗算） | Pre-Audit §2.6 / §4.2 |
| P3 | ピークは `:422 loadedIR = N × fileLength × 8` が支配する | Pre-Audit §2.2 |
| P4 | `MAX_FILE_LENGTH = INT32_MAX` は **sample 上限であって byte 上限ではない** | Pre-Audit §4.2 |
| P5 | `numChannels` 上限チェックは読込経路に **存在しない**（`:398` の `<= 0` のみ） | Pre-Audit §2.3 |
| P6 | `MAX_IR_LATENCY` は DSP usable length の上限であり、loader bound ではない | Pre-Audit §3 |
| P7 | N ≥ 3 は build 段（`:166-169`）で不使用 = dead weight | Pre-Audit §2.3 |
| P8 | streaming / incremental partition build は不要（入力読込は既にチャンク化済） | Pre-Audit §5.3 |
| P9 | `MAX_FILE_LENGTH = INT32_MAX` を製品契約値に採ることはできない | Pre-Audit §4.2 / §6 |

### 2.2 本 work で追加実測した確定事実（F 系）

Pre-Audit が扱わなかった経路・量を実測で確定した。FC-1/FC-2 の導出は本表に依拠する。

| ID | 事実 | 実測位置 |
|---|---|---|
| **F1** | Convolver が実際に使用するのは **ch0 / ch1 の 2ch のみ**（mono は ch0 を両側へ複製） | `LoaderThread.cpp:166-167` |
| **F2** | N ≥ 3 は trim / resample / DC block / Tukey / `setSize` を通るが build で未使用 | `:514,:559,:573,:586` vs `:166-169` |
| **F3** | DSP usable length 上限 = `MAX_IR_LATENCY = 2,097,152`。`computeTargetIRLength` は `min(sr × irLenSec, 2,097,152)`（`originalLength` は `ignoreUnused`） | `ConvolverProcessor.h:250`, `StateAndUI.cpp:945-967` |
| **F4** | `targetIRLengthSec` の権威クランプ = `[IR_LENGTH_MIN_SEC=0.5, hardMaxSec(sr)=2,097,152/sr]`。`hardMax` は 3.0 と min を取らない（48k "Load as-is" >3s 互換） | `Runtime.cpp:929-969`, `StateAndUI.cpp:155-175` |
| **F5** | 対応処理 SR = **44.1 kHz – 768 kHz**（`SAFE_MAX_SAMPLE_RATE = 768000`） | `doc/audit/ConvoPeq_SampleRate_Support_Audit_2026-09-13.md:41,175` |
| **F6** | state 直列化は `irPath`（**ファイルパス**）+ `irLength`(sec) + パラメータのみ。**IR 音声サンプルは保存しない** | `StateAndUI.cpp:212-264` |
| **F7** | `loadedIR` は `juce::AudioBuffer<double>` = **8 B/sample/ch** | `LoaderThread.cpp:422`, `LoaderThreadInline.h` |
| **F8** | `fileLength > INT32_MAX` は allocation 前に reject（既存・維持） | `LoaderThread.cpp:391-397` |
| **F9** | `makeAlignedArray` は失敗時 **`std::bad_alloc` を throw**（nullptr 返しではない）。`:416` の null 検査は **dead code** | `AlignedAllocation.h:143-146` |
| **F10** | JUCE `ThreadPool::runNextJob` は job 例外を `catch(...) { jassertfalse; }` で握る（terminate しない） | `juce_ThreadPool.cpp:389-393` |
| **F11** | **第 2 の IR ファイル読込経路**が実在する: `loadImpulseResponsePreviewFile`（**非チャンク・全長一時確保**）→ `analyzeImpulseResponseFile` ← `ConvolverControlPanel::startAsyncIRLoadPreview`（thread pool job） | `ResampleAndFallback.cpp:271-331`, `StateAndUI.cpp:455-546`, `ConvolverControlPanel.cpp:1135-1168` |
| **F12** | **`computeIRHash` が admission 前に O(file bytes) を確保する**（→ §2.3） | `LoaderThread.cpp:363-364` → `AllpassDesigner.cpp:626-643` |
| **F13** | 例外モデルは `/EHsc`（`/EHa` はリポジトリに 0 件）→ **SEH は `catch(...)` に翻訳されない** | `CMakeLists.txt:1522,1527,1617` |

### 2.3 ★ Pre-Audit 是正 — `computeIRHash` の admission 前確保と null 経路（FC-8）

**Pre-Audit §2.6 の「クラッシュ経路は確認できない」は、`computeIRHash` を調査対象に含めていなかったため成立しない。**

実測（`AllpassDesigner.cpp:626-643`、TOCTOU 検証は `:655`）:

```cpp
juce::HeapBlock<uint8_t> fileData;                        // throwOnFailure = false（既定）
const size_t fileSize = (size_t) jmax<int64>(0, sizeBefore);
if (fileSize > 0)
    fileData.malloc(fileSize);                            // ← ① O(file bytes) を admission 前に確保
...
for (;;) {
    const int bytesRead = stream->read(tempBuffer, 4096);
    if (bytesRead <= 0) break;
    const size_t bytes = (size_t) bytesRead;
    if (writePos + bytes > fileSize) return 0;
    if (bytes > 0)
        std::memcpy(fileData.getData() + writePos, tempBuffer, bytes);   // ← ② nullptr + 0 へ書込
    writePos += bytes;
}
```

根拠（4 点すべて実測）:

1. `juce::HeapBlock<uint8_t>` は既定 `throwOnFailure = false` → `ThrowOnFail<false>::checkPointer` は
   **no-op**（`juce_HeapBlock.h:43`）。確保失敗時は `std::malloc` の nullptr がそのまま残る。
2. `fileData.getData()` が nullptr のまま ② に到達する。初回反復は `writePos = 0` なので
   `nullptr + 0` への memcpy = **アクセス違反**。`writePos + bytes > fileSize` は
   ファイルサイズ既知（`sizeBefore`）のため通常成立せず、ガードにならない。
3. 例外モデルは `/EHsc`（F13）→ SEH は `catch(...)` に翻訳されず **プロセス終了**。
4. 呼び出し位置 `LoaderThread.cpp:363-364` は **admission 判定（`:391-402`）より前**であり、
   `stepFileHash` の唯一の消費者は `:651`（mixed-phase cache key）である。

**到達条件**: `malloc(fileSize)` が失敗すること。すなわち「メモリ制約下で大きな IR ファイルを選ぶ」
（例: 誤って数 GB の音声ファイルを IR として指定）。float64 stereo WAV では
file bytes ≒ N × fileLength × 8（= loadedIR と同規模）。

**判定**: これは **FC-INV-1（allocation は admission より前でない）の現行違反**であり、
**FC-INV-5（allocation failure は graceful）の現行違反**でもある。
両者は本 work が凍結しようとしている不変条件そのものなので、契約に **FC-8 として取り込む**（§11）。

**ただし Pre-Audit 判定 A は維持する**。理由:
- Option A の骨格（admission 前 reject の構造・int 溢れなし・`:422` の bounded 化）は §2.1 P1-P9 のとおり成立しており、
  FC-8 は **同じ admission 順序契約の欠落 1 箇所**であって契約形の変更ではない。
- FC-8 の閉じ方は「hash を admission 後に移す」「hash を streaming 化する」「null 検査 + graceful 失敗」
  のいずれも **小改修**であり、Pre-Audit §4.1 の結論（構造は既にある／足りないのは契約と明文化）を覆さない。
- したがって判定は **A のまま**、Pre-Audit §2.6 の文言のみ訂正する。

---

## 3. FC-1 — 契約形の比較と採択

候補（ユーザー提示）:

```text
A-1  byte 上限      N × fileLength × sizeof(double) ≤ kMaxIRLoadBytes
A-2  channel + sample 上限   N ≤ K  ∧  fileLength ≤ L
A-3  sample 上限引下げ  fileLength ≤ L（+ 必要なら channel 上限）
```

### 3.1 各候補の評価（実測に基づく）

| 候補 | byte を有界化できるか | transient を有界化できるか | 意味のある上限になるか | 判定 |
|---|---|---|---|---|
| **A-3**（sample のみ） | **不可**。N が非有界（P5）なので `N × L × 8` は非有界 | **不可**。`N × kStreamChunk × 4` は N 比例・fileLength 非依存 | — | **単独では不成立** |
| **A-2**（channel + sample） | **可**（`K × L × 8` は有限） | **可**（`K × 1 MiB`） | **L を小さくしない限り不可**。互換のために L=INT32_MAX を残すと byte 上限は `K × INT32_MAX × 8` = 8ch で 137 GB となり「有限だが無意味」（P9 と同じ欠陥） | **L を下げる＝A-3 併用が必須** |
| **A-1**（byte） | **可**。N に依存せず常に同一の byte 保証 | **不可**。`fileLength = 1, N = 10^6` でも条件を満たすため、`N × 262,144 × 4` が無界 | **可** | **channel 上限の併設が必須** |

**結論**: どの単一案も単独では「bounded allocation」を成立させない。
A-1 と A-2 は **同じ契約の直交する 2 面**であり、両方を採る必要がある。

### 3.2 採択形（凍結）

```text
FC-FORM-1  byte admission（主契約・A-1）
    numChannels × fileLength × sizeof(double) ≤ kMaxIRLoadBytes

FC-FORM-2  channel admission（A-2 の channel 面）
    numChannels ≤ kMaxIRLoadChannels

FC-FORM-3  representation precondition（既存維持・A-3 の sample 面は「引下げ」しない）
    fileLength ≤ MAX_FILE_LENGTH = 2,147,483,647

FC-FORM-4  degenerate precondition（既存維持）
    numChannels ≥ 1  ∧  fileLength ≥ 1
```

**A-3 を「sample 上限の引下げ」として採用しない理由**:
- FC-FORM-1 が成立するとき、実効 sample 上限は `kMaxIRLoadBytes / (N × 8)` であり、
  **常に INT32_MAX より小さい**（§4.3 の表）。したがって sample 上限は
  **FC-FORM-1 に吸収される**（独立した policy 値を持たない）。
- 独立した sample 上限を policy として置くことは **truncation policy の変更**であり、
  互換性変更（§6）を不必要に広げる。本 work は **allocation の有界化** を契約対象とし、
  truncation policy は SR-01 の既存ログに委ねる。

**FC-FORM-3 を残す理由**: `static_cast<int>(fileLength)`（`:422`）の **narrowing 証明**に必要。
`juce::AudioBuffer::setSize` は `int` 引数であり、`fileLength ≤ INT32_MAX` は
**型契約であって製品上限ではない**。この区別を §9 FC-7 で凍結する。

**FC-FORM-4 を残す理由**: 既存 `:398`（`numChannels <= 0`）と `:454`（`> 0` 検査）の維持。
削除・改変しない。

---

## 4. FC-2 — 契約値の導出

### 4.1 導出に用いる実測量（ユーザー指定 8 項目との対応）

| # | 指定項目 | 実測値 | 出典 |
|---|---|---|---|
| 1 | Convolver が実際に使用する channel 数 | **2**（ch0/ch1。mono は複製） | F1 |
| 2 | N ≥ 3 が dead weight | **真**（build は ch0/ch1 のみ） | F2 |
| 3 | `MAX_IR_LATENCY` の責務 | DSP usable length 上限（2,097,152 samples）。**loader bound ではない** | F3 / P6 |
| 4 | 現行ユーザーが利用可能な IR 長 | `irLenSec ∈ [0.5, 2,097,152/sr]` → targetLength ≤ 2,097,152 samples | F4 |
| 5 | sample rate による IR length | 下表（44.1k: 47.55 s 〜 768k: 2.73 s） | F4/F5 |
| 6 | 既存 state / file compatibility | state は `irPath` のみ保存（音声非保存）→ **互換はオンディスク IR ファイルの問題に限定** | F6 |
| 7 | `loadedIR` の double representation | **8 B/sample/ch** | F7 |
| 8 | peak と transient | peak = `:422`（`N × fileLength × 8`）／transient = `:414` `N × 262,144 × 4`（**N 比例・fileLength 非依存**）+ `:415` 2 MiB | P3, F7 |

#### (5) sample rate ごとの利用可能 IR 長（`hardMaxSec(sr) = 2,097,152 / sr`）

| processing SR | hardMaxSec | targetLength 上限 |
|---|---|---|
| 44,100 Hz | 47.55 s | 2,097,152 |
| 48,000 Hz | 43.69 s | |
| 88,200 Hz | 23.78 s | |
| 96,000 Hz | 21.85 s | |
| 176,400 Hz | 11.89 s | |
| 192,000 Hz | 10.92 s | |
| 352,800 Hz | 5.94 s | |
| 384,000 Hz | 5.46 s | |
| 705,600 Hz | 2.97 s | |
| 768,000 Hz | 2.73 s | |

**どの SR でも targetLength ≤ 2,097,152 samples**（サンプル数では SR 不変）。

### 4.2 `kMaxIRLoadBytes` の導出

**導出規則（外部仮定を置かない）**:

```text
loader は「DSP が完全に使い切れる最大のソース」を必ず受け入れなければならない。

  完全に使い切れるソース [samples/ch] = MAX_IR_LATENCY × (fileSR / sr)
                                      ↑ リサンプル後に DSP 窓 (2,097,152) に収まる上限
  fileSR の上限・sr の下限は、いずれも製品自身が宣言する SR 範囲 (F5) を用いる:
      fileSR_max = 768,000 Hz
      sr_min     =  44,100 Hz

required = N_used × ceil(MAX_IR_LATENCY × fileSR_max / sr_min) × sizeof(double)
         = 2        × ceil(2,097,152   × 768,000 / 44,100)   × 8
         = 2        × 36,521,831                             × 8
         = 584,349,296 B
         = 557.28 MiB

kMaxIRLoadBytes = required 以上の最小 2 冪
                = 2^30 = 1,073,741,824 B = 1,024 MiB (1 GiB)
```

**マージン**: `1 GiB / 557.28 MiB = 1.84×`。
この倍数は「アロケータの内部オーバーヘッド + リサンプラのテール数サンプル
（`getMaxOutLen` の `+1`、`CDSPFracInterpolator.h:831`）+ ceil 切り上げ」を吸収する目的の
最小 2 冪であり、恣意的な丸めではない。

**「2 冪」を規則にする理由**: `DELAY_BUFFER_SIZE` と同じく 2 冪は比較・記述・回帰テストが
決定的になる（`ConvolverProcessor.h:257-259` の既存流儀）。

> **採用しなかった案**: `fileSR_max = 384,000`（IR ライブラリの実勢上限）とすると required = 278.64 MiB →
> 256 MiB 以上最小 2 冪 = **512 MiB**。しかしこの場合 FC-4 で **768 kHz の stereo が newly rejected** となり、
> FC-4 が明示的に分離を要求する「stereo」「768 kHz」の 2 カテゴリを同時に汚す。
> 製品自身の宣言範囲（44.1–768 kHz）をそのまま使えば **外部仮定が 0 になり、かつ stereo が全 SR で OK** になるため、
> **1 GiB を採る**（§6 の表参照）。

### 4.3 `kMaxIRLoadChannels` の導出

**導出規則**: `k` は「(i) N 比例 transient を byte 予算の数 % に収め、(ii) 実在する IR ファイルの
チャンネル数を拒否しない」最小値とする。

```text
(i)  transient  T1 = N × kStreamChunk × 4 = N × 1 MiB    （fileLength 非依存・N 比例。P3/8）
     T1 ≤ byte 予算の 1 % に収める:  K ≤ 0.01 × 1,024 MiB / 1 MiB = 10.24  → K ≤ 10
     参考: K = 8 のとき T1 = 8 MiB = 予算の 0.78 %

(ii) 実在する IR ファイルのチャンネル数
     1 (mono) / 2 (stereo) / 4 (B-format・quad) / 8 (7.1)。N > 8 の IR WAV は製品入力として存在しない。

→ 両条件を満たす上限値: kMaxIRLoadChannels = 8
```

**N ≥ 3 が dead weight であることは K を 2 にする根拠にはしない。**
これは **互換性の判断**（§6）であり、§4.2 の導出とは分離する（ユーザー指示）。

### 4.4 凍結値

```text
kMaxIRLoadBytes    = 1,073,741,824   (1 GiB, 2^30)
kMaxIRLoadChannels = 8
MAX_FILE_LENGTH    = 2,147,483,647   （既存維持・representation precondition）
```

### 4.5 各 N に対する実効 sample 上限（`kMaxIRLoadBytes / (N × 8)`）

| N | max fileLength [samples/ch] | INT32_MAX との関係 |
|---|---|---|
| 1 | 134,217,728 | byte 上限が先に効く |
| 2 | 67,108,864 | 同上 |
| 3 | 44,739,242 | 同上 |
| 4 | 33,554,432 | 同上 |
| 5 | 26,843,545 | 同上 |
| 6 | 22,369,621 | 同上 |
| 7 | 19,173,961 | 同上 |
| 8 | 16,777,216 | 同上 |

**全 N ≥ 1 で byte 上限が INT32_MAX より先に効く** → Pre-Audit P9 の要求
（INT32_MAX を既定値として採用しない）を満たす。INT32_MAX は
**型契約（narrowing 証明）専用** に降格する（§9）。

### 4.6 導出される有界量（K = 8, kStreamChunk = 262,144）

| 記号 | 実体 | 位置 | 上限 |
|---|---|---|---|
| — | `loadedIR`（**admission 対象**） | `:422` | **1,024 MiB** |
| T1 | `tempFloatBuffer`（transient, N 比例） | `:414` | 8 MiB |
| T2 | `tempAligned`（transient, 固定） | `:415` | 2 MiB |
| T3 | `stepTrimmed`（N × targetLength × 8） | `:586` | 128 MiB |
| T4 | `irL` / `irR`（2 × targetLength × 8） | `:163-164` | 32 MiB |
| T5 | persisted `IRState`（= `commit->loadedIR`） | `LoadPipeline.cpp:801` | ≤ 1,024 MiB |

**worst resident 合計 ≒ 1,194 MiB** = 1,024（loadedIR = T5 と同一実体）+ 128（stepTrimmed）
+ 32（irL/irR）+ 8 + 2。
`IRState` は `loadedIR` そのものを保持するため、T5 は loadedIR の行と**同一の実体**である
（二重計上していない）。`stepTrimmed` は `doTrimStep` で生成され、
`buildConvolverFromTrimmed` 完了まで `loadedIR` と同居するため加算している。

**対比（Pre-Audit worst case）**:

| 構成 | 現行（無制限） | 凍結後 |
|---|---|---|
| 2ch × INT32_MAX | 34.36 GB | ≤ 1 GiB（**32× 削減**） |
| 64ch × INT32_MAX | 1.10 TB | ≤ 1 GiB（**1,024× 削減**） |

---

## 5. FC-3 — 診断メッセージ契約

### 5.1 現行の不正確さ

```text
LoaderThread.cpp:395        "IR file is too large (exceeds 2GB samples limit)."
ResampleAndFallback.cpp:297 "IR file is too large (exceeds 2GB samples limit)."   ← 同一文字列の重複
LoaderThread.cpp:130        "IR too large (Out of Memory)"
```

問題:
1. 「2GB」は **サンプル数** であって byte ではない（P4）。ユーザーは byte と誤読する。
2. channel 上限が存在しないことを開示していない。
3. **実際に拒否された量**と**契約値**を出していない。
4. 同一ガードの文字列が 2 箇所に重複している（drift リスク）。

### 5.2 凍結する診断仕様

**契約面**: `LoadResult::errorMessage`（`juce::String`）。
経路: `doLoadIRStep` → `stepResult.errorMessage` → `LoaderThread::run():88-100` →
`callAsync` → `ConvolverProcessor::handleLoadError` → `setLastError` → UI。
**本 work はこの文字列契約のみを対象とし、UI 表示形式・レイアウトは変更しない。**

**開示レベル（決定）**: **dimension + actual + limit** の 3 要素を出す。
数式・計算過程は出さない（可読性と i18n 非依存の維持）。
単位は **byte は "bytes"、sample は "samples"、channel は "channels"** と明示する。

| 発火条件 | 凍結文言（`errorMessage`） |
|---|---|
| FC-FORM-1 違反 | `"IR file is too large for the load memory limit (2 channels x 20000000 samples x 8 bytes exceeds 1073741824 bytes)."` |
| FC-FORM-2 違反 | `"IR file has too many channels (16 channels; limit is 8)."` |
| FC-FORM-3 違反 | `"IR file is too long (2200000000 samples; limit is 2147483647)."` |

上記の太字部は `juce::String(...)` による実数値の埋め込み（`numChannels`, `fileLength`,
`numChannels * fileLength * 8`, `kMaxIRLoadBytes`, `kMaxIRLoadChannels`）。
**4 要素（sample limit / channel limit / byte limit / actual）をすべて出し、contract limit も同時に出す。**

**単一情報源**: FC-FORM-1/2/3 のガードと文言は **1 箇所（loader 契約ブロック）で定義**し、
`LoaderThread.cpp` と `ResampleAndFallback.cpp` の 2 経路はそこを参照する（重複文字列を残さない）。

**既存文言の扱い**: `"IR too large (Out of Memory)"`（`:130`）は **allocation 失敗**（FC-INV-5）に対応する
別事象であり、FC-FORM-1/2/3 とは dimension が異なるため **変更しない**。

---

## 6. FC-4 — 後方互換性（独立判定）

### 6.1 判定の枠組み

```text
「旧契約（fileLength ≤ INT32_MAX, N ≥ 1, byte 無制限）で合法だった IR」
        ↓
「新契約（FC-FORM-1/2/3/4）でも合法か」
```

**F6 により、互換性は state ファイルではなくオンディスク IR ファイルの問題に限定される。**
（state は `irPath` のみ保存し IR 音声を保存しない。既存プロジェクトの state は
新契約で一切影響を受けない。**互換性判定は IR ファイルの (N, fileLength, fileSR) の 3 変数のみ**。）

### 6.2 判定式

ファイルが **newly rejected** になるのは次のいずれか:

```text
(1) N > 8                                     （channel bound）
(2) N × fileLength × 8 > 1,073,741,824        （byte bound）
```

「DSP が完全に使い切れる」条件は `fileLength ≥ MAX_IR_LATENCY × fileSR / sr`。
これが受理される条件は `N × MAX_IR_LATENCY × (fileSR/sr) × 8 ≤ kMaxIRLoadBytes`。

### 6.3 SR × channel 互換マトリクス（sr = 44.1 kHz が最悪条件）

「完全に使い切れる最大ファイル」が受理されるか（OK / NG）:

| N | 実効上限 | 44.1k | 48k | 96k | 192k | 384k | 705.6k | 768k |
|---|---|---|---|---|---|---|---|---|
| **1 (mono)** | 134,217,728 | OK | OK | OK | OK | OK | OK | **OK** |
| **2 (stereo)** | 67,108,864 | OK | OK | OK | OK | OK | OK | **OK** |
| 3 | 44,739,242 | OK | OK | OK | OK | OK | OK | OK |
| 4 | 33,554,432 | OK | OK | OK | OK | OK | OK | **NG** |
| 6 | 22,369,621 | OK | OK | OK | OK | OK | NG | NG |
| 8 | 16,777,216 | OK | OK | OK | OK | NG | NG | NG |

### 6.4 カテゴリ別判定（ユーザー指定項目）

| カテゴリ | 判定 | 根拠 |
|---|---|---|
| **48 / 96 / 192 / 384 / 705.6 / 768 kHz（processing SR）** | **互換** | FC-FORM-1/2 は**処理 SR に依存しない**（判定式 §6.2 に sr が現れない） |
| **既存最大 IR length** | **互換** | 最大利用長 = `targetLength ≤ 2,097,152 samples`（F3）。stereo の実効上限 67,108,864 samples はこれを **32×** 上回る |
| **stereo (N=2)** | **互換（全 SR）** | §6.3 のとおり 44.1k–768k すべて OK。**完全に使い切れる stereo ファイルが拒否されることはない** |
| **mono (N=1)** | **互換（全 SR）** | 実効上限 134,217,728 samples。完全に使い切れる mono ファイルは拒否されない |
| **N = 3** | **互換（全 SR）** | §6.3 |
| **N = 4〜8** | **条件付き互換（宣言付き）** | 「高 SR × 極長」でのみ newly rejected（4ch の 768 kHz、8ch の 384 kHz 以上）。実ファイル存在性は無視できるが、**hardening ではなく compatibility change として明示する**（§6.5） |
| **N > 8** | **compatibility change（宣言付き）** | 新規拒否。ただし N > 8 の IR ファイルは製品入力として存在しない |

### 6.5 compatibility change の明示（重要）

ユーザー指定の原則に従い、次を **hardening ではなく compatibility change として記録する**:

```text
CC-1  N > 8 の IR ファイルは新規に拒否される。
CC-2  4 〜 8ch かつ高 SR (≥ 384 kHz) かつ完全利用長に近いファイルは新規に拒否されうる。
      （4ch: 768 kHz のみ / 6ch: 384 kHz 以上 / 8ch: 192 kHz 以上）
CC-3  mono / stereo / 3ch は全対応 SR で互換。処理 SR には一切依存しない。
```

**CC-1/CC-2 の実害評価**: 該当ファイルは「同時に巨大（≥ 数百 MB）」かつ「多チャンネル」かつ
「超高 SR」であり、いずれも製品の IR 入力として現実に流通していない。
一方、**CC-3 が成立することにより、通常の IR ワークフロー（mono/stereo, 44.1–768 kHz）は
一切影響を受けない**。この非対称性が本契約の互換性根拠である。

### 6.6 楽観的でない確認（ファイル内容の非依存性）

- `loadedIR` の byte 量は `(N, fileLength)` のみの関数であり、
  **音声内容・フォーマット・圧縮率に依存しない**（`:422` は常に全長を確保する）。
- したがって §6.3 の判定は **あらゆるファイルに対して厳密**である（確率的でない）。

---

## 7. FC-5 — 明文化の場所（documentation loci）

UI は本件の scope 外。**次の 3 箇所で意味が矛盾しないこと**を契約条件とする。

| # | 場所 | 記載する意味 | 現状 | 必要な変更 |
|---|---|---|---|---|
| **D-1** | `ConvolverProcessor.LoaderThread.cpp` の loader 契約ブロック（定数宣言と同一箇所） | **正本**。FC-FORM-1/2/3/4、`kMaxIRLoadBytes` / `kMaxIRLoadChannels` の値、reject 順序、診断仕様 | 無し（`MAX_FILE_LENGTH` のみ） | 新設 |
| **D-2** | `ConvolverProcessor.h:250` の `MAX_IR_LATENCY` 宣言コメント | **DSP usable length の上限であること**、**loader allocation bound ではないこと**（FC-INV-3） | 「DelayLine用定数」「IRの最大長(kMaxIRCap)と最大ブロックサイズをカバーする値」のみ | 責務境界の 1 文を追記 |
| **D-3** | `doc/work102/big18_failure_contract_20260917.md`（本文書） | 契約の完全定義・導出・互換判定・不変条件 | 本 work で作成 | — |

**矛盾解消の確認事項**:

1. D-2 の「IRの最大長」は `computeTargetIRLength` の `kMaxIRCap`（`StateAndUI.cpp:952`）を指す。
   D-1 の loader bound とは**別の量**であることを D-1/D-2 の双方に明記し、
   **`kMaxIRLoadBytes` を `MAX_IR_LATENCY × N × 8` と定義してはならない**。
2. SR-01（`doc/work95/sr01b_implementation_gate_contract_freeze_20260914.md`）が凍結した
   `hardMaxSec(sr) = MAX_IR_LATENCY / sr` は **DSP 使用長**の契約であり、D-1 と競合しない。
   D-1 はこれを変更しない。
3. 既存の preview 側診断（`IRLoadPreview::exceedsRecommended` / `exceedsHardLimit`、
   `StateAndUI.cpp:540-541`）は **長さ（秒）** の助言であり、D-1 の **ファイル byte** 契約とは
   dimension が異なる。両者を同一メッセージに統合しない。
4. `ResampleAndFallback.cpp:297` の重複文字列は D-1 を参照する形へ一本化する（§5.2）。

**README / ユーザー向け文書への追記は本 work の必須条件にしない**
（ユーザー指定により UI と外部文書は scope 外。D-1/D-2/D-3 で閉じる）。

---

## 8. FC-6 — channel handling の決定

### 8.1 選択肢の比較

```text
(a) loader admission で reject
(b) load 後に ch0/ch1 だけ保持
(c) 現状の全 channel 保持を維持
```

| 案 | 内容 | 本 work scope での妥当性 |
|---|---|---|
| **(b)** | `loadedIR` を最初から 2ch で確保し、ch2 以降を読み捨て | **不可（scope 外）**。確保量は減るが、`IRState` の所有前提（`irOwner` unique_ptr + `ir` raw、`ConvolverProcessor.h:1185-1192`）と data representation、`stepTrimmed` の全 ch 前提（`:586`）、`IRConverter::computeScaleFactor` / minimum-phase / mixed-phase（全 ch 反復、`MixedPhase.cpp:221,239`）が同時に変わる。**allocation bound とは別の ownership/data-representation change** |
| **(a)** | `numChannels > K` を admission で reject | **採用（K = 8 の範囲でのみ）** |
| **(c)** | 全 ch 保持を維持 | **採用（N ≤ K の範囲）** |

### 8.2 凍結

```text
FC-CH-1  N ≤ kMaxIRLoadChannels (=8) では (c) 現状維持。
         ch0/ch1 以外は引き続き load/trim/resample/Tukey される（dead weight だが bounded）。
FC-CH-2  N > kMaxIRLoadChannels でのみ (a) admission reject。
FC-CH-3  (b) ch0/ch1 のみ保持は本 work で行わない（ownership / data-representation change として分離）。
FC-CH-4  N ≥ 3 が dead weight である事実は D-1 にコメントとして記録する
         （K を縮める根拠には使わない。K の根拠は §4.3）。
```

**(b) を分離する理由（記録）**: (b) は「メモリを減らす」変更ではなく
「IR の表現を 2ch 固定へ変える」変更である。影響範囲は
`IRState`/`applyNewState`/`PendingCommit`/`IRConverter`/`MixedPhase`/visualization に及び、
§11 の Implementation scope（小改修）を明確に超える。将来 work として記録する。

---

## 9. FC-7 — `MAX_IR_LATENCY` 非統合の凍結

Pre-Audit §3 の責務分離を契約文書へ固定する。

```text
FC-INV-3（再掲・凍結）
  MAX_IR_LATENCY = 2,097,152 は DSP が使用する IR length の上限である。
  Loader allocation bound ではない。
  両者を同一の capacity constant に統合してはならない。

  [1] computeTargetIRLength(sampleRate, originalLength)
        = min(sampleRate × targetIRLengthSec, MAX_IR_LATENCY)      （originalLength は ignoreUnused）
        用途: :163-164 irL/irR、:586 stepTrimmed.setSize(N, targetLength)
        → DSP 使用長の上限

  [2] doLoadIRStep
        fileLength : :391-397  MAX_FILE_LENGTH
        loadedIR   : :422      numChannels × fileLength × 8
        → Loader が確保する入力の上限

  [3] targetLength は fileLength の関数ではない（originalLength 未使用）
        => MAX_IR_LATENCY は loadedIR(:422) のサイズを上界しない

禁止条項:
  kMaxIRLoadBytes を MAX_IR_LATENCY × N × sizeof(double) として定義すること。
  kMaxIRLoadChannels を「チャンネル数を 2 にすれば MAX_IR_LATENCY が byte 上限になる」と
  いう理由で決めること（K の根拠は §4.3 の transient 予算 + 実在 channel 数のみ）。
```

---

## 10. Frozen Invariants

ユーザー提示 FC-INV-1〜7 を **Contract Requirement**（外部から観測される契約）と
**Implementation Requirement**（実装が満たすべき内部条件）に分離して凍結する。

### 10.1 Contract Requirements

```text
FC-INV-1  [C] admission は allocation に先行する。
              観測可能な意味: 上限超過ファイルに対して、
              O(fileLength) / O(file bytes) の確保を試みた形跡なく reject が返ること。
              ★ 現行は computeIRHash（§2.3 FC-8）で違反している。実装で閉じる。

FC-INV-2  [C] 許容される最大 loadedIR allocation は有限である。
              凍結値: 1,073,741,824 B (= kMaxIRLoadBytes)。
              本不変条件の対象は loadedIR（:422）に限定する（後述 10.3 R-1 参照）。

FC-INV-4  [C] 上限超過は allocation 前に deterministic reject する。
              deterministic = 同じ (N, fileLength, fileSR, sr) に対し常に同じ判定。
              音声内容・フォーマットに依存しない（§6.6）。

FC-INV-5  [C] allocation failure は graceful failure とする。
              現行の performLoad の catch（bad_alloc / std::exception / ...）を維持し、
              errorMessage を設定して return false する（std::terminate しない）。
              ★ FC-8 の null memcpy はこの不変条件の違反であり、実装で閉じる。
```

### 10.2 Implementation Requirements

```text
FC-INV-3  [I] MAX_IR_LATENCY は DSP usable length の上限であり、
              Loader allocation bound ではない（§9 の禁止条項を含む）。

FC-INV-6  [I] 既存の chunked file read（:404-450、kStreamChunk = 262,144）を維持する。
              本 work はチャンク化を変更・撤去しない。

FC-INV-7  [I] streaming / incremental partition build は本 work の scope 外。
              `IncrementalRebuildJob`（Rebuild.cpp:177）は dormant のまま維持する。

FC-INV-8  [I] 診断は単一情報源とする（§5.2）。同一ガードを 2 箇所に重複定義しない。

FC-INV-9  [I] admission 順序:
              1. FC-FORM-4 (degenerate)
              2. FC-FORM-3 (representation, fileLength ≤ INT32_MAX)
              3. FC-FORM-2 (channel)
              4. FC-FORM-1 (byte)
              5. ★ ここより後に O(file bytes) / O(fileLength) の確保を開始する
              6. loadedIR 確保 / chunk read
              ※ FC-8 により、現行の computeIRHash は 5 より前に置かれている。
                 実装で 5 以降へ移すか、O(1) 化する。
```

### 10.3 scope 外として登録する有界性（R 系）

```text
R-1  リサンプル中間（`ResampleAndFallback.cpp:44-101`）
     peak ≈ loadedIR + 2 × N × ceil(fileLength × sr/fileSR) × 8
     FC-FORM-1 は loadedIR のみを有界化するため、この中間は
     SR 比 (sr/fileSR) が有界でない限り FC-FORM-1 では有界化されない。
     現状は FC-INV-5（bad_alloc → graceful）で受け止められている。
     → 本契約の対象外。§12 D-1 として判断点に登録。

R-2  フォーマットデコーダ内部のバッファ（WAV/AIFF/FLAC、`registerBasicFormats`）
     ストリーミング実装であり file size 比例の確保は確認できないが、
     デコーダ内部の挙動は本契約の対象外。将来の hardening 候補として記録。
```

---

## 11. Implementation Scope（実装 GO 時に触る範囲。本 work では実装しない）

```text
IS-1  定数の新設（D-1 の正本ブロック）
        kMaxIRLoadBytes    = 1,073,741,824
        kMaxIRLoadChannels = 8
        （MAX_FILE_LENGTH = 2147483647 は既存のまま残す）

IS-2  admission ブロックの追加（LoaderThread.cpp:391-402 の直後、:413 より前）
        FC-FORM-2 → FC-FORM-1 の順に判定し、超過時は errorMessage を設定して return false。
        ★ :414 の tempFloatBuffer は N 比例なので、必ず FC-FORM-2 の後であること。

IS-3  診断の一本化（§5.2）
        LoaderThread.cpp:395 と ResampleAndFallback.cpp:297 の重複文字列を 1 箇所に統合し、
        dimension + actual + limit を出す文言へ差し替える。
        "IR too large (Out of Memory)"（:130）は変更しない。

IS-4  FC-8 の閉塞（§2.3）
        (a) computeIRHash を admission ブロックの後へ移す、または
        (b) computeIRHash を O(1) メモリのストリーミング実装へ変更する、または
        (c) null 検査 + graceful error を追加する。
        推奨: (b)。TOCTOU 保護は既に before/after の size+mtime 検証
        （AllpassDesigner.cpp:655）が担っており、全ファイル buffer は冗長。
        (a) も可（stepFileHash の唯一の消費者は :651 で、同一ステップ内で後置可能）。
        ★ (a) は fileLength 基準、(b)/(c) は file bytes 基準であり、
          「巨大メタデータ chunk を持つ小さな data chunk の WAV」には
          (b)/(c) のみが有効。**(b) を第一候補とする。**

IS-5  D-2 のコメント追記（ConvolverProcessor.h:250）— 非統合の 1 文のみ。

IS-6  ResampleAndFallback.cpp:307 の全長 tempFloatBuffer（preview 経路）は
      IS-2 の contract を共有する限りで新規拒否が入る。
      ★ 非チャンク実装そのものの改修は本 work の scope 外（別 work）。
         契約（N ≤ 8, N × fileLength × 8 ≤ 1 GiB）は両経路に同一に適用する。

禁止（変更 0）:
  M-03 D3 / M-03 latency・PDC / Direct Head HC/LC / H-01 / H-02 / SR-01〜03 /
  Publish・Crossfade・Retire authority / Epoch / RuntimeWorld / Lifetime Budget / RT path /
  chunked read の撤去 / incremental partition build / IRState の表現変更 /
  ch0-ch1 のみ保持（FC-CH-3）/ UI / tests(target 追加) / CMake / ConvoPeq.md 再生成 /
  README E-G3-1 差分
```

**RT safety 監査（凍結案に対する事前評価）**: IS-2/IS-3/IS-4 はすべて
`LoaderThread`（NonRT・`:42` で HeavyBackground affinity を適用）または
purpose-built thread pool job 上で動く。RT callback への追加は **0**。
atomic 追加 **0**（`stepFileHash` / `errorMessage` は既存の非共有メンバ）。
したがって SR-03 / M-04 / M-02 が凍結した RT telemetry 語彙への影響はない。

---

## 12. Gate 判定

```text
判定: CONTRACT-FROZEN
```

**根拠（ユーザーの判定基準との対応）**:

| 基準 | 状態 |
|---|---|
| 契約形が確定 | **確定** — FC-FORM-1/2/3/4（§3.2）。A-1 + channel 併設を採択し、A-3 は FC-FORM-3 の representation precondition として吸収 |
| 値が確定 | **確定** — `kMaxIRLoadBytes = 1 GiB`、`kMaxIRLoadChannels = 8`（§4.4）。導出規則は製品自身の定数（`MAX_IR_LATENCY`, SR 範囲）のみを使用し外部仮定 0 |
| diagnostic が確定 | **確定** — dimension + actual + limit の 3 要素、単一情報源、UI 非変更（§5.2） |
| compatibility が確定 | **確定** — mono/stereo/3ch は全対応 SR で互換、N≥4 の高 SR 極長のみ CC-1/CC-2 として明示的に compatibility change（§6.5） |

**他判定の棄却**:

- **CONTRACT-HOLD**: 値の根拠は §4.2 の式で再現可能（外部仮定 0）。
  互換性も §6.3 で全カテゴリ判定済み。→ 棄却。
- **DESIGN-DEFERRED**: streaming / channel representation 変更は **不要**（P8）。
  必要なのは契約値と明文化のみ（Pre-Audit §4.1 の結論を維持）。→ 棄却。
- **NO-GO**: bounded contract は §4.6 のとおり定義可能。→ 棄却。

### 実装 GO 時の条件（Implementation Gate へ引き継ぐ決定点）

```text
cond-1  D-1（判断点）: リサンプル中間の有界化（R-1）を本 work に含めるか。
        含める場合の追加契約: `ceil(fileLength × sr / fileSR) ≤ kMaxIRResampleOutputSamples`
        （候補値は §4.1 の DSP usable 上限に整合させる）。
        推奨: **含めない**。理由:
          (i) FC-INV-2 は loadedIR に限定されており、R-1 は契約の外にある。
          (ii) FC-FORM-1 の外に第 2 の SR 依存上限を足すと、
               互換性サーフェスが増え CC の範囲が広がる。
          (iii) 現状 FC-INV-5 で graceful に受け止められている。
        → 別 work「IR resample intermediate bound」として起票候補に記録。

cond-2  IS-4 の閉塞方式（推奨: (b) ストリーミング hash）。
cond-3  CC-1 / CC-2 の compatibility change の承認（ユーザー裁定）。
cond-4  §11 の禁止条項の承認（特に ch0-ch1 のみ保持を混入しないこと）。
cond-5  Installation Scope の IS-6（preview 経路の非チャンク実装）を
        本 work で扱わないことの承認。
```

**本 work はここで停止する。Implementation Gate へは進まない。**
Pre-Audit の「次工程は Failure Contract」を本 work が閉じ、
次の工程（Implementation Gate）の起動はユーザー承認を待つ。

---

## 13. 監査メタ

```text
検証方法    : LoaderThread.cpp / LoadPipeline.cpp / StateAndUI.cpp / Runtime.cpp /
              ResampleAndFallback.cpp / MixedPhase.cpp / Rebuild.cpp / ConvolverProcessor.h /
              AllpassDesigner.cpp / AlignedAllocation.h / juce_AudioSampleBuffer.h /
              juce_HeapBlock.h / juce_ThreadPool.cpp / r8brain-free-src（getMaxOutLen）の
              行単位実測 + 数値導出の再計算

Production 変更     : 0
CMake 変更          : 0
UI 変更             : 0
tests 変更          : 0
ConvoPeq.md 再生成  : 0（FRESH / NEWER_SRC_COUNT = 0 を再実測。巻き戻しなし）
commit / push       : 0
README E-G3-1 差分  : 維持

禁止事項の遵守:
  M-03 D3 実装なし / M-03 latency・PDC 変更なし / Direct Head HC/LC 変更なし /
  H-01・H-02 変更なし / SR-01〜03 変更なし / Publish・Crossfade・Retire authority 追加なし /
  Epoch・RuntimeWorld・Lifetime Budget 変更なし / RT path 変更なし /
  上限値の実装なし（本 work は契約凍結まで）
```

### Pre-Audit からの差分（重要な変更点の要約）

1. **§2.3 新規**: `computeIRHash` の admission 前 O(file bytes) 確保と null memcpy。
   Pre-Audit §2.6 の「クラッシュ経路は確認できない」を **訂正**（判定 A は維持）。
2. **FC-1**: A-3 を policy として採用せず、FC-FORM-3 の representation precondition へ降格。
3. **FC-4**: 判定変数は `(N, fileLength, fileSR)` のみ（F6 により state 非依存）。
   全カテゴリの OK/NG を確定。
4. **FC-6**: (b) を scope 外として明示分離（Pre-Audit の選択肢提示を確定に変更）。
5. 第 2 の loader 経路（F11）を契約適用対象に含めた（Pre-Audit は 1 経路のみ追跡）。
