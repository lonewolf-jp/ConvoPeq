# WORK102-FC-R1 — Failure Contract Reconciliation / Mathematical Re-proof（read-only）

- **作成日**: 2026-09-17
- **種別**: Reconciliation / Mathematical Re-proof（**実装は行わない**）
- **被監査対象**: `doc/work102/big18_failure_contract_20260917.md`（以下 **FC 文書**）
- **前工程**: `doc/work102/big18_pre_audit_bounded_allocation_contract_20260917.md`（判定 A — CONTRACT-READY）
- **前例フォーマット**: `doc/work98/m02_pre_audit_failure_contract_freeze_20260915.md`
- **目的**: **実装 GO を出すことではない。** `1 GiB` と `8ch` を
  「数学的に導出された値」と「製品として選択した policy」に混同しない状態まで契約を再整理すること。

> **R1 の中心結論（先出し）**
>
> 1. FC 文書 §4.2 の「外部仮定を置かない」は **誤り**。SR 包絡の外部仮定は除去したが、
>    **より強い policy 前提 P1（「DSP が完全に使い切れるソースは必ず受け入れる」）を無ラベルで導入**していた。
>    P1 は **現行コードにも既存仕様にも存在しない**（§3）。
> 2. `kMaxIRLoadChannels = 8` の根拠「N > 8 は製品入力として存在しない」は
>    **外部情報でありソース証明ではない**（§4）。
>    さらに **SR-01B の既凍結境界条件「channel count: >2ch→先頭2」と衝突する**（§4.4）。
> 3. **R-1（リサンプル中間）は scope 外にできない。** 実測: 修正後契約の内側でも
>    **37.40 GB** に到達し、**big 1-8 の原欠陥 34.36 GB を上回る**（§5）。
>    FC 文書 §4.6 の「worst resident ≒ 1,194 MiB」はリサンプル経路を無視した過小評価。
> 4. `loadedIR` のみを有界化する契約は、**big 1-8 の目的（IR ロード時の巨大確保防止）を達成しない**。
>    → **Case B を要求として確定**（§5.4）。
> 5. したがって **R-1 を閉じる新たな admission 次元（FC-FORM-5）が必要**である。
>    閉じ方自体は admission チェックであり設計変更ではない（§10.3）。
>
> **最終判定: DESIGN-DEFERRED**（§12）。値の policy 裁定は併存する HOLD 要因として残る。

---

## 1. Baseline

```text
HEAD        = 35461e2e3a5fd42fd009416a20adaf457b9b4031
origin/main = 35461e2e3a5fd42fd009416a20adaf457b9b4031   (ahead/behind = 0/0)

ConvoPeq.md  mtime = 2026-09-17 00:07:39 JST（本 R1 でも不変。再生成なし）
src/ のうち ConvoPeq.md より新しいもの = 0 件（FRESH 維持）

本 R1 の変更:
  doc/work102/big18_failure_contract_reconciliation_20260917.md の新規作成のみ
  production source = 0 / tests = 0 / CMake = 0 / UI = 0
  ConvoPeq.md 再生成 = 0 / commit = 0 / push = 0
  README.md E-G3-1 差分 = 維持（不触）
```

### 1.1 再実測した現行構造（ユーザー指摘の確認）

`ConvolverProcessor.LoaderThread.cpp::doLoadIRStep` の順序（行番号実測）:

```text
:364      stepFileHash = computeIRHash(file)                     ← ① O(file bytes) 確保
:389      fileLength  = reader->lengthInSamples
:390      numChannels = reader->numChannels
:391-397  MAX_FILE_LENGTH = 2147483647 判定                       ← ② 既存 admission
:398-402  numChannels <= 0 判定                                   ← ③ 既存 admission
:414      tempFloatBuffer(numChannels, 256*1024)                  ← ④ N 比例 transient
:415      makeAlignedArray<double>(256*1024)                      ← ⑤ 固定 transient
:422      loadedIR.setSize(numChannels, (int)fileLength)          ← ⑥ ピーク確保
```

別経路 `ConvolverProcessor.ResampleAndFallback.cpp::loadImpulseResponsePreviewFile`:
`:293 maxFileLength = 2147483647` 判定 / `:301 numChannels <= 0` 判定 は**同一**だが、
`:307 tempFloatBuffer(numChannels, fileLength)`（**全長・非チャンク**）を持つ。

→ ユーザー指摘のとおり。**byte 次元の admission は現行 0 箇所**である。

---

## 2. FC-FORM re-check（契約形の再検証）

FC 文書 §3.2 の凍結形:

```text
FC-FORM-1  N × fileLength × 8 ≤ kMaxIRLoadBytes
FC-FORM-2  N ≤ kMaxIRLoadChannels
FC-FORM-3  fileLength ≤ INT32_MAX        （representation precondition）
FC-FORM-4  N ≥ 1 ∧ fileLength ≥ 1
```

### 2.1 形の必要性（これは導出できる）

「bounded allocation」という**目標**を採る限り、形は強制される:

| 目標 | 強制される形 | 理由 |
|---|---|---|
| `loadedIR` の byte を N に依存せず有界化 | **byte bound（FC-FORM-1）が必須** | sample bound では `N × L × 8` が非有界 |
| N 比例 transient（T1 = `N × 262,144 × 4`）の有界化 | **channel bound（FC-FORM-2）が必須** | T1 は fileLength 非依存なので byte bound では抑えられない |
| `static_cast<int>` narrowing の正当化 | **FC-FORM-3 が必須** | `AudioBuffer::setSize(int,int)` の型契約 |

→ **FC-FORM-1/2/3 は「目標 → 形」の導出として妥当。** これは R1 でも維持する。

### 2.2 形だけでは不足（R1 の新規指摘）

上記 3 つは `loadedIR` と load-step transient のみを対象とする。
**リサンプル中間（C_res）はこの 3 つのいずれでも有界化されない**（§5）。
したがって **FC-FORM は不完全**であり、§10.3 で **FC-FORM-5** を追加する。

### 2.3 FC-FORM の前提（無ラベルだった policy 目標）

FC-FORM-1 の **値** は「どのソースを受け入れるか」という policy に依存する。
FC 文書 §4.2 はそこに **P1** を無ラベルで置いていた:

```text
P1: 「DSP が完全に使い切れる最大のソースを、loader は必ず受け入れなければならない」
```

**P1 は目標ではなく前提である。** §3.1 で P1 がコード／既存仕様から証明できないことを示す。

---

## 3. `kMaxIRLoadBytes` の再証明

### 3.1 P1 の出典調査（P1 はソース証明されていない）

リポジトリ全体（src/ + doc/）を検索した結果:

```text
検索: "fully usable" / "full usable" / "使い切" / "完全に利用" / "accept all" / "must accept"
結果: src/ と doc/ に「完全に使い切れるソースを必ず受け入れる」を要求する記述は 0 件
      （唯一の一致は FC 文書自身 = doc/work102/big18_failure_contract_20260917.md）
```

現行コードが実際に要求しているのは次のみ:

```text
:391-397  fileLength ≤ INT32_MAX なら受け入れる        （完全利用可否を問わない）
:398-402  numChannels ≥ 1 なら受け入れる               （同上）
```

**→ P1 は「コードから証明できる事実」ではなく、R1 以前の FC 文書が導入した policy 前提である。**

### 3.2 P1 を外した場合に残る純粋な事実

```text
PF-1  C_src = N × fileLength × 8                          （:422 の式）
PF-2  U     = MAX_IR_LATENCY = 2,097,152                  （DSP usable length の上限）
PF-3  C_dsp = 2 × U × 8 = 33,554,432 B = 32 MiB           （build は ch0/ch1 のみ使用）
PF-4  L_res = ceil(fileLength × sr / fileSR) + 1           （r8b CDSPFracInterpolator.h:831）
PF-5  現行の上限は fileLength ≤ INT32_MAX のみ。N に上限は無い
      → 現行の worst-case byte は形式的に非有界
```

**PF-1〜5 から有限の byte 上限は導出できない。**
「非有界をどこで切るか」は必ず **policy** である。

### 3.3 FC 文書の 1 GiB を分解する

FC 文書 §4.2 の式:

```text
required = 2 × ceil(2,097,152 × 768,000 / 44,100) × 8 = 584,349,296 B = 557.28 MiB
         → 次の 2 冪 → 1,073,741,824 B = 1 GiB
```

この式が成立するには、次の **3 つの前提すべて**が必要:

| 前提 | 内容 | 区分 | 出典 |
|---|---|---|---|
| **P1** | 「完全に使い切れるソースは必ず受け入れる」 | **POLICY CHOICE** | 証明なし（§3.1） |
| **P2** | `N_used = 2`（ch0/ch1 のみ使用） | **PROVEN FACT** | `LoaderThread.cpp:166-167` |
| **P3** | fileSR の包絡 = 768,000 Hz（= 製品の**処理** SR 上限を IR **ファイル** SR に流用） | **POLICY CHOICE** | 処理 SR に 768k は証明されるが、IR **ファイル** が 768k である証明はない（§3.4） |

**→ `1 GiB` は「数学的に導出された唯一の契約値」ではなく、P1+P3 を前提とした
DERIVED REQUIREMENT（条件付き要求）である。P1/P3 自体が policy なので、最終的には POLICY CHOICE。**

### 3.4 P3（fileSR 包絡 = 768 kHz）の出典確認

```text
証明できる:  処理 SR の範囲 = 44.1 kHz – 768 kHz
             （doc/audit/ConvoPeq_SampleRate_Support_Audit_2026-09-13.md:41,175、
               SAFE_MAX_SAMPLE_RATE = 768000.0 / kMaxInternalRate = 768000）
証明できない: IR ファイルの SR が 768 kHz まで存在しうること
             現行コードは reader->sampleRate を一切検証しない（0 より大きければ受理）
             doc/ に IR ファイル SR の上限を述べた記述は 0 件
             （唯一の関係記述は work88 BUG-045 の「48000 Hz IR を 192000 Hz セッションへ」例）
```

**→ P3 は policy choice。** なお P3 を 384 kHz にすると required = 278.64 MiB → 512 MiB、
192 kHz にすると 139.3 MiB → 256 MiB となり、**値は前提だけで 4 倍動く**（この感度が policy である証拠）。

### 3.5 「2 冪に丸める」規則も policy

FC 文書は「required 以上の最小 2 冪」を規則とした。これは:
- 導出ではない（丸め規則の選択）。
- 実利は「記述・比較・回帰テストの決定性」であり、`DELAY_BUFFER_SIZE` の前例に倣った選択。

**→ POLICY CHOICE**（ただし無害であり、維持を推奨）。

### 3.6 代替前提（policy 選択肢）

| 前提 | 意味 | required | 値（最小 2 冪） | 備考 |
|---|---|---|---|---|
| **P1 + P3(768k)** ← FC 文書の採用 | 全 SR 包絡で完全利用を保証 | 557.28 MiB | **1 GiB** | 最も寛容 |
| P1 + P3(384k) | 384 kHz まで | 278.64 MiB | 512 MiB | 768k stereo を新規拒否 |
| P1 + P3(192k) | 192 kHz まで | 139.33 MiB | 256 MiB | 384k 以上を新規拒否 |
| P1 なし（1:1 のみ保証） | fileSR==sr の完全利用のみ | 32 MiB | 32 MiB | 48k IR @44.1k を拒否 |
| 目標を「メモリ安全」に限定 | 完全利用の保証を放棄 | — | 任意（例 512 MiB） | 互換は別途 CC で明示 |

**裁定が必要な点**: どの前提を採るか。R1 は P1+P3(768k) を棄却しないが、
**「導出値」ではなく「選択値」として扱う**ことを要求する。

---

## 4. `kMaxIRLoadChannels` の再証明

### 4.1 要求された 4 区分の分離

| 区分 | 内容 | 実測／出典 | 区分 |
|---|---|---|---|
| **A** | ConvoPeq が実際に**処理可能**な channel 数 | **2**。`srcL = getReadPointer(0)`、`srcR = (N>1) ? getReadPointer(1) : srcL`。N≥3 は build に到達しない | **PROVEN FACT** |
| **B** | loader が**現在受け入れている** channel 数 | **非有界**。`numChannels <= 0` のみ（`:398-402`）。上限チェックは全 src で 0 件 | **PROVEN FACT** |
| **C** | **製品仕様として**許容される channel 数 | **SR-01B の凍結境界条件が「>2ch→先頭2 現行不変」と明記** = 上限なしで受理し、先頭 2ch を使用する契約。上限値を定めた仕様は **0 件** | **部分的に PROVEN**（上限は未定義） |
| **D** | **一般的な IR ファイル**に存在する channel 数 | 1/2/4/8 等 | **EXTERNAL**（リポジトリ外情報。ソース証明ではない） |

### 4.2 FC 文書の表現の是正

FC 文書 §4.3 は次を根拠とした:

```text
「1ch / 2ch / 4ch / 8ch が実在する」「N > 8 は製品入力として存在しない」
→ K = 8 を「実在する IR ファイルを拒否しない最小値」と表現
```

**R1 判定: この表現は不適切。**
- 後半（N > 8 は存在しない）は区分 **D = 外部情報** であり、**ソース証明ではない**。
- 前半のみでは「最小値」は決まらない（1ch しか数えなければ K=1 が「最小」になる）。
- 実際に K を決めているのは FC 文書 §4.3 の (i)「T1 ≤ byte 予算の 1 %」という
  **新たに置いた予算基準**である。これも policy である。

**→ `kMaxIRLoadChannels = 8` は derived value ではなく POLICY VALUE。**

### 4.3 K の値が依存する policy

| 基準 | 内容 | 導かれる K |
|---|---|---|
| (i) T1 を byte 予算の 1 % 以下 | `K ≤ 0.01 × 1 GiB / 1 MiB = 10.24` | K ≤ 10 |
| (ii) 区分 D（外部情報） | 実在は ≤ 8ch | K = 8 |
| (iii) 区分 A のみ（ソース証明のみ） | 使用は 2ch → 上限 2 で足りる | K = 2（ただし CC 大） |
| (iv) 現行維持（区分 B/C） | 上限なし | K = 無制限（byte bound のみで拘束） |

**K = 2 は「ソース証明だけ」なら最も自然だが、区分 C（SR-01B 凍結）と衝突する。**
K = 8 は (i)(ii) の合成であり、**外部情報と新しい予算基準の合成**である。

### 4.4 ★ SR-01B 既凍結契約との衝突（R1 の重要指摘）

`doc/work95/sr01b_implementation_gate_contract_freeze_20260914.md`（2026-09-14 凍結）:

```text
### 境界条件（凍結）
| channel count | 2ch 正規化（mono→複製、>2ch→先頭2）現行不変（LoaderThread.cpp:166-169） |
```

これは「**>2ch は受理され、先頭 2ch を使う**」という**既に凍結された境界条件**である。
上限値は定めていないが、**「上限なしで受理する」ことが凍結されている**。

したがって `K = 8` は:

```text
N ≤ 8   : SR-01B 凍結条件と両立（>2ch を引き続き受理し、先頭2を使用）
N > 8   : SR-01B 凍結条件を変更する（新規拒否の導入）
          → これは「新しい hardening」ではなく **既凍結契約の改訂** である
```

**→ K = 8 を採る場合は、SR-01B §境界条件の `channel count` 行を
明示的に supersede（改訂）する必要がある。** 黙って追加してはならない。
FC 文書 §6.5 の CC-1（N > 8 は compatibility change）は正しく宣言しているが、
**「既凍結契約の改訂」という性質が未記載**だった点を R1 で補正する。

---

## 5. `loadedIR` vs whole-loader resource scope（R1 の中核）

### 5.1 3 段の capacity の再計算

```text
C_src = N × fileLength × 8                                  ← source-file capacity（:422）
L_res = ceil(fileLength × sr / fileSR) + 1                   ← r8b 契約（CDSPFracInterpolator.h:831）
C_res = N × L_res × 8                                        ← resampled capacity（中間）
C_dsp = 2 × U × 8 = 32 MiB                                   ← DSP usable capacity
```

**同一ファイル（N=2）で 3 段を並べる**（U = 2,097,152、`fileLength` は「完全利用の境界」）:

| 条件 | fileLength | C_src | r = sr/fileSR | L_res | C_res | C_dsp |
|---|---|---|---|---|---|---|
| fileSR == sr (44.1k/44.1k) | 2,097,152 | **32.0 MiB** | 1.000 | 2,097,152 | **32.0 MiB** | 32.0 MiB |
| 192k file @ 44.1k | 9,130,453 | **139.3 MiB** | 0.230 | 2,097,152 | **32.0 MiB** | 32.0 MiB |
| 768k file @ 44.1k | 36,521,813 | **557.3 MiB** | 0.057 | 2,097,152 | **32.0 MiB** | 32.0 MiB |
| 44.1k file @ 768k | 120,399 | **1.8 MiB** | 17.415 | 2,096,746 | **32.0 MiB** | 32.0 MiB |

**読み取れること（R1 の重要な観察）**:

1. **「完全利用」の境界では C_res が常に ≒ C_dsp（32 MiB）になる。** これは定義上当然である
   （完全利用 = リサンプル結果が DSP 窓にちょうど収まる）。
2. **C_src だけが前提に依存して 1.8 MiB 〜 557.3 MiB の 300 倍以上に振れる。**
   すなわち **C_src は物理要求ではなく「どの fileSR まで完全利用を保証するか」という policy の産物**である。
3. **リサンプル比の向きが逆になると、別の量が支配する**（次節）。

**→ 「完全利用可能」という表現は、loader の要求仕様ではなく、
「そのソースを採用した場合に DSP が全部使える」という**ソース側の性質**にすぎない。
loader がそれを受け入れる義務は、P1 を採用した場合にのみ発生する。**

### 5.2 C_res は C_src の 17.4 倍になりうる（R-1 の実体）

`C_res / C_src ≈ sr / fileSR` である。アップサンプリング（fileSR < sr）では
**リサンプル中間がソースの 17.4 倍**になる（44.1k → 768k）。

さらに `resampleIR`（`ResampleAndFallback.cpp:44-101`）は
**① チャンネル毎の `std::vector<double> chData[N]`（`buf.resize(maxOut)`）と
② 出力 `juce::AudioBuffer<double> result(numCh, maxLen)` を同時に保持**する
（`:46`, `:62`, `:93`）。したがって中間のピークは **≒ 2 × C_res**。

### 5.3 ★ 修正後契約の内側での R-1 worst case（決定的）

**FC-FORM-1 を満たす範囲で**、C_res の最大値を計算する:

```text
条件: N = 2, C_src = 1 GiB（= FC-FORM-1 の上限ちょうど）→ fileLength = 67,108,864
       fileSR = 44,100 Hz, sr = 768,000 Hz
L_res = ceil(67,108,864 × 768,000 / 44,100) + 1 = 1,168,698,585 samples
C_res = 2 × 1,168,698,585 × 8                    = 18,699,177,360 B
C_res ピーク（chData + result）                  = 37,398,354,720 B = 37.40 GB
```

| 構成（すべて FC-FORM-1 適合） | C_res ピーク |
|---|---|
| N=2, fileSR=768k, sr=44.1k | 0.12 GB |
| N=2, fileSR=44.1k, sr=44.1k | 2.15 GB |
| N=2, fileSR=48k, sr=768k | **34.36 GB** |
| **N=2, fileSR=44.1k, sr=768k** | **37.40 GB** |
| （比較）big 1-8 の原欠陥 | **34.36 GB** |

**→ FC-FORM-1 を完全に満たしていても、IR ロード経路は 37.40 GB の確保を試行しうる。
これは big 1-8 が除去しようとした原欠陥（34.36 GB）を上回る。**

**FC 文書 §4.6 の「worst resident 合計 ≒ 1,194 MiB」は、リサンプル中間を計上していないため
過小評価である。** 正しい whole-path worst は **≒ 38.4 GB**（1 GiB + 37.40 GB）。

### 5.4 Case A / Case B の確定

```text
Case A: big 1-8 の契約対象 = loadedIR allocation のみ
Case B: big 1-8 の目的     = IR loading pipeline 全体の bounded allocation
```

**記録上の scope（Case A を支持する証拠）**:

- work92 `CORRECT_INTEGRATED_BUG_LIST.md:44`: 「LoaderThread が `MAX_FILE_LENGTH = INT32_MAX` まで
  **一括確保**（OOM）」「2GB 超 IR で最大 ~16GB 確保試行」
- work92 `PLAN.md:345`: メモリ効果の内訳は
  `tempFloatBuffer + tempAligned + loadedIR` の 3 項のみ（**リサンプルは列挙されていない**）
- work101 §5-2: 「実装必要性は『上限の明文化』であり『ストリーミング実装』ではない」

**しかし Case A は自己矛盾する**:

```text
big 1-8 の欠陥記述は「IR ファイルを読み込むと巨大確保が起きる」である。
R-1 は同じ doTrimStep/Resampler 経路で 37.40 GB を試行する（§5.3）。
→ Case A を採ると、big 1-8 は「同種の欠陥を 1 箇所だけ閉じ、より大きい箇所を残す」ことになる。
```

**R1 判定: Case B を要求として確定する。**
すなわち **R-1 は scope 外にできない**（ユーザー指示の Case B 条件に該当）。
ただし §10.3 のとおり、閉塞手段は **admission チェックの追加**であり、
リサンプラの順序変更やストリーミング化といった**設計変更ではない**。

---

## 6. FC-8 `computeIRHash`

### 6.1 現行順序の再確認（FAIL 維持）

```text
:364  computeIRHash(file)                                  ← AllpassDesigner.cpp:626-643
      HeapBlock<uint8_t> fileData; fileData.malloc(fileSize);   // throwOnFailure=false → nullptr
      ...
      std::memcpy(fileData.getData() + writePos, tempBuffer, bytes);   // nullptr + 0 へ書込
:391  admission（MAX_FILE_LENGTH）
:422  loadedIR 確保
```

- `ThrowOnFail<false>::checkPointer` は no-op（`juce_HeapBlock.h:43`）。
- 例外モデルは `/EHsc`（`CMakeLists.txt:1522,1527,1617`。`/EHa` は 0 件）
  → SEH は `catch(...)` に翻訳されず **プロセス終了**。

```text
FC-INV-1（admission が allocation に先行）     : 現行 **FAIL**（①が②より前）
FC-INV-5（allocation failure は graceful）     : 現行 **FAIL**（null memcpy → AV）
```

**R1 では FC-8 を実装要求として維持する。**（ユーザー指示どおり）

### 6.2 IS-4 (a)/(b)/(c) の契約上の要否

**重要な追加事実**: ハッシュの確保量は `fileSize`（物理ファイルバイト）であり、
**`fileLength`（サンプル数）とは別次元** である。

```text
PCM では  fileSize ≒ N × fileLength × (2|3|4|8)  ≦ N × fileLength × 8
しかし WAV は任意の追加チャンクを持てるため、
「巨大なメタデータチャンク + 1 サンプルの data チャンク」のファイルでは
fileSize ≫ N × fileLength × 8 となり、**FC-FORM-1 では拘束されない**
```

| 案 | 内容 | FC-INV-1（順序） | FC-INV-5（graceful） | ハッシュ確保の**有界性** | 契約上の要否 |
|---|---|---|---|---|---|
| **(a)** admission 後に hash を移す | 順序は満たす | ○ | ✗（null 経路は残る） | **✗**（file bytes は FC-FORM-1 で拘束されない） | **単独では不十分** |
| **(b)** streaming hash 化（O(1) メモリ） | 確保そのものを除去 | ○ | ○（例外要因が消える） | **○（O(1)）** | **十分** |
| **(c)** null 検査 + graceful error | 失敗を graceful 化 | **✗**（確保が admission より前のまま） | ○ | ✗ | **単独では不十分** |
| **(d)** 物理ファイルサイズの admission 追加 | `fileSize ≤ kMaxIRFileBytes` を `:364` より前に置く | ○ | ○（null 経路も閉じる） | **○（バイト上限で拘束）** | **十分** |

**R1 判定（契約必要条件）**:

```text
必要条件: hash 経路は「O(1) メモリ」または「事前の byte admission で拘束」のいずれかであること。

  (b) 単独で十分。
  (d) 単独で十分（かつ「メタデータチャンク肥大」という fileLength が見ない次元も閉じる）。
  (a) 単独では不十分（順序のみ。有界性は得られない）。
  (c) 単独では不十分（graceful のみ。順序も有界性も得られない）。
  (a)+(c) でも有界性は得られない。

推奨: (b) を主、 (d) を等価代替。
  理由: (b) は TOCTOU 保護が既存の before/after size+mtime 検証
        （AllpassDesigner.cpp:655）で既に成立しており、全長 buffer は冗長。
        (d) は新しい policy 値を 1 つ増やすが、より強い（メタデータ次元も閉じる）。
```

### 6.3 RT 原則との整合

`doc/Practical Stable ISR Bridge Runtime.md` の原則（RT 側に allocation/free を持ち込まず
危険操作を NonRT 側へ隔離）に対し、`computeIRHash` は
`LoaderThread`（NonRT・`:41-42` で HeavyBackground affinity）上で動くため**原則違反ではない**。
ただし「NonRT だから無制限でよい」ではなく、**NonRT 側でも確保量は有界であるべき**という
本契約の趣旨により、§6.2 の必要条件が課される。

---

## 7. Preview loader の適用性

### 7.1 別 admission consumer であることの確認

```text
経路 1（main）  : ConvolverProcessor::loadIR / loadImpulseResponse
                  → LoaderThread::doLoadIRStep            （admission: :391,:398 のみ）
経路 2（preview）: ConvolverControlPanel::startAsyncIRLoadPreview  （:1135-1168）
                  → g_irPreviewThreadPool.addJob          （:16, 単一スレッド pool）
                  → ConvolverProcessor::analyzeImpulseResponseFile （StateAndUI.cpp:455）
                  → loadImpulseResponsePreviewFile        （ResampleAndFallback.cpp:271-331）
                                                          （admission: :293,:301 のみ）
```

**→ 最終契約は両経路で同一 admission semantics を持たねばならない。**

### 7.2 Preview 経路の追加実測（R1 新規）

| 項目 | 実測 | 帰結 |
|---|---|---|
| transient | `:307 tempFloatBuffer(numChannels, fileLength)` = **N × L × 4（全長・非チャンク）** | `:314 makeAlignedArray<double>(fileLength)` = L × 8 と合わせ、ピーク ≒ `L(12N+8)`。FC-FORM-1 適合時 `≤ kMaxIRLoadBytes × (1.5 + 1/N) ≤ 2.5 GiB`（N=1） |
| byte admission | **無し**（`:294` は INT32_MAX samples のみ） | FC-FORM-1/2 を追加する必要 |
| **graceful failure** | **無し**。`startAsyncIRLoadPreview` の job lambda に try/catch が無く、JUCE `ThreadPool::runNextJob` が `catch(...){ jassertfalse; }` で握る（`juce_ThreadPool.cpp:389-393`） | **bad_alloc 時は `finishAsyncIRLoadPreview` に到達せず、`irPreviewInProgress` が true のまま残る**（`:1143` set / `:1176` のみ clear。`:1437` の `updateIRInfo` が以後ずっと "Analyzing IR..." を表示し早期 return） |

**→ Preview 経路は FC-INV-5 に**現行 FAIL**。**（クラッシュではないが恒久的な UI 状態欠陥）
これは FC 文書が「§11 IS-6 は scope 外」として扱った範囲に含まれるが、
**FC-INV-5 を contracts として掲げる以上、preview 経路の graceful failure は
別 work として明示的に起票される必要がある**（黙って残さない）。

### 7.3 Preview の duration 検査は allocation 検査ではない（重要）

`finishAsyncIRLoadPreview`（`ConvolverControlPanel.cpp:1202-1220`）は
`preview.exceedsHardLimit` のとき警告を出して **`applyPreviewIRLengthAndLoad` を呼ばずに return** する。
すなわち **UI 経路は既に「hardMax 超はロードしない」を実装している**（SR-01B の「preview reject（既装）」）。

ただし `exceedsHardLimit` の比較対象は
`autoDetectedLengthSec = estimateEffectiveIRLengthSamples(...) / loadedSampleRate`
（`StateAndUI.cpp:535-541`）であり、**「有効長（末尾減衰を除いた実効長）」** である。

```text
→ 長大なファイルでも実効長が短ければ UI は受理する。
   ローダは依然として全長を確保する。
   したがって UI の duration 検査は **allocation bound の代替にならない**。
```

**→ FC-FORM-1 は UI 検査から導出できない。**（これも FC-FORM-1 の値が policy である理由）

### 7.4 経路別 admission の現状まとめ

| 経路 | LoaderThread | Preview | CLI (`MainWindow.cpp:796,838`) | state 復元 (`StateAndUI.cpp:417`) |
|---|---|---|---|---|
| byte admission | ✗ | ✗ | ✗ | ✗ |
| channel 上限 | ✗ | ✗ | ✗ | ✗ |
| duration/hardMax 検査 | ✗ | **○**（間接。§7.3） | ✗ | ✗ |
| graceful failure | ○（`performLoad` catch） | **✗**（§7.2） | （main 経路に同じ） | （main 経路に同じ） |

**→ CLI と state 復元は preview を経由しない**（`requestConvolverPreset` を直接呼ぶ）。
したがって「UI が既に拒否しているから loader は不要」とは言えない。

---

## 8. Compatibility の再証明

### 8.1 「K が policy として承認された」と「その K で matrix が成立する」の分離

FC 文書 §6.4 は次の順序で書かれていた:

```text
（暗黙に K = 8 を確定） → mono/stereo/3ch 互換 / 4-8ch conditional / >8ch CC
```

**R1 では順序を分離する:**

```text
Step 1（policy 裁定）: K の値を決める。K = 8 は §4 のとおり policy value。
                       併せて SR-01B §境界条件 channel count 行の supersede を承認する（§4.4）。
Step 2（条件付き証明）: Step 1 で決まった K に対してのみ、下記 matrix が成立する。
                       K を変えれば matrix は再計算が必要。
```

**Step 2 は Step 1 の結論に依存する。K が未裁定である限り、matrix は
「K = 8 と仮定した場合の帰結」であって compatibility proof ではない。**

### 8.2 「N > 8 は存在しない」を proof の根拠にしない

```text
FC 文書 §6.4 の「N > 8 の IR ファイルは製品入力として存在しない」は区分 D（外部情報）。
→ Step 2 の根拠として使用しない。
→ 代わりに「N > 8 は新規拒否 = 既凍結 SR-01B 境界条件の改訂」として §4.4 に計上する。
```

### 8.3 K = 8, B = 1 GiB を仮定した場合の matrix（条件付き帰結）

判定式: newly rejected ⟺ `N > 8` ∨ `N × fileLength × 8 > 1 GiB`
（「完全利用の境界」=`fileLength = (U−1) × fileSR / sr` の受理可否、sr = 44.1 kHz が最悪条件）

| N | 実効上限 | 44.1k | 48k | 96k | 192k | 384k | 705.6k | 768k |
|---|---|---|---|---|---|---|---|---|
| 1 (mono) | 134,217,728 | OK | OK | OK | OK | OK | OK | OK |
| 2 (stereo) | 67,108,864 | OK | OK | OK | OK | OK | OK | OK |
| 3 | 44,739,242 | OK | OK | OK | OK | OK | OK | OK |
| 4 | 33,554,432 | OK | OK | OK | OK | OK | OK | NG |
| 6 | 22,369,621 | OK | OK | OK | OK | OK | NG | NG |
| 8 | 16,777,216 | OK | OK | OK | OK | NG | NG | NG |

（数値は FC 文書 §6.3 と一致。R1 で再計算し一致を確認）

**処理 SR には依存しない**（判定式に sr が現れない）。FC 文書の「48/96/192/384/768 kHz 互換」は
**K と B を固定した条件下で**成立する。

### 8.4 FC-FORM-5 が新たに導入する互換差分

FC-FORM-5（§10.3）は `L_res ≤ kMaxIRResampleOutputSamples`、すなわち

```text
duration = fileLength / fileSR  ≤  U / sr = hardMaxSec(sr)
```

を要求する。これによる差分:

| 経路 | 差分 |
|---|---|
| interactive ControlPanel | **差分なし**。SR-01B の preview reject が既に hardMax 超を拒否している（`:1202-1220`） |
| CLI（`MainWindow.cpp:796,838`） | **新規拒否**。preview を経由しないため、duration > hardMaxSec(sr) のファイルが新たに拒否される |
| state 復元（`StateAndUI.cpp:417`） | **新規拒否**。同上 |

処理 SR ごとの最長受理 duration:

| sr | 44.1k | 48k | 96k | 192k | 384k | 705.6k | 768k |
|---|---|---|---|---|---|---|---|
| 最長 duration | 47.55 s | 43.69 s | 21.85 s | 10.92 s | 5.46 s | 2.97 s | **2.73 s** |

**注意**: 705.6 k / 768 kHz では **3 秒 IR が新規拒否**される（UI 経路では既に拒否済み）。
44.1k–384 kHz では 3 秒 IR はすべて受理され、実質的な差分は生じない。

### 8.5 R-1 を閉じる場合の追加互換（重要）

FC-FORM-5 は **R-1 を閉じる代わりに、新たな拒否を導入する**:

```text
現行:      fileLength ≤ INT32_MAX なら、duration に関わらずロードを試行（巨大確保 → OOM or graceful 失敗）
FC-FORM-5: duration > hardMaxSec(sr) は **確保前に deterministic reject**
```

これは「OOM で失敗していたものが、明示的なエラーになる」という**挙動の正規化**であり、
成功していたロードを失敗させるものではない（§5.3 のとおり、そのようなファイルは
リサンプル中間の確保に失敗するか、巨大メモリを消費して成功するかのいずれかだった）。

**→ ただし厳密には「巨大メモリを消費して成功していたケース」を失敗に変える。**
これは compatibility change として CC-4 に計上する（§10.5）。

---

## 9. Proven Fact / Derived Requirement / Policy Choice の分離

### 9.1 PROVEN FACT（コードまたは既存仕様から証明できる）

| ID | 内容 | 出典 |
|---|---|---|
| PF-1 | `loadedIR` の確保式 = `numChannels × fileLength × sizeof(double)` | `LoaderThread.cpp:422` |
| PF-2 | `sizeof(double)` = 8、`loadedIR` は `juce::AudioBuffer<double>` | `LoaderThreadInline.h` / `:422` |
| PF-3 | `MAX_IR_LATENCY = 2,097,152` は DSP 使用長の上限（`computeTargetIRLength` の cap） | `ConvolverProcessor.h:250`, `StateAndUI.cpp:945-967` |
| PF-4 | build が使用するのは **ch0/ch1 の 2ch のみ** | `LoaderThread.cpp:166-167` |
| PF-5 | `numChannels` に上限チェックが存在しない（`:398` は `<= 0` のみ） | `LoaderThread.cpp:398-402` |
| PF-6 | `fileLength > INT32_MAX` は allocation 前に reject される | `LoaderThread.cpp:391-397` |
| PF-7 | リサンプル出力長 = `ceil(inLen × DstSR/SrcSR) + 1` | `r8brain-free-src/CDSPFracInterpolator.h:831` |
| PF-8 | リサンプル中間は chData（N × maxOut × 8）+ result（N × maxLen × 8）を同時保持 | `ResampleAndFallback.cpp:46-93` |
| PF-9 | 処理 SR 範囲 = 44.1 kHz – 768 kHz | `SampleRate_Support_Audit:41,175` |
| PF-10 | `hardMaxSec(sr) = MAX_IR_LATENCY / sr`、かつ UI はこれを超える IR を拒否する | `Runtime.cpp:946-969`, `ConvolverControlPanel.cpp:1202-1220` |
| PF-11 | `computeIRHash` が admission 前に O(file bytes) を確保し、失敗時 null memcpy に至る | `AllpassDesigner.cpp:626-643`, `juce_HeapBlock.h:43` |
| PF-12 | CLI 経路と state 復元経路は preview（duration 検査）を経由しない | `MainWindow.cpp:796,838`, `StateAndUI.cpp:417` |
| PF-13 | preview 経路は job 例外を握るだけで graceful failure を持たない | `ConvolverControlPanel.cpp:1143-1176`, `juce_ThreadPool.cpp:389-393` |
| PF-14 | SR-01B が「>2ch→先頭2」を凍結境界条件として明記 | `sr01b_implementation_gate_contract_freeze_20260914.md` §境界条件 |

### 9.2 DERIVED REQUIREMENT（前提を認めたときに必然となる要求）

| ID | 内容 | 前提 |
|---|---|---|
| DR-1 | byte bound + channel bound + representation precondition の 3 点が必要 | 「bounded allocation」という目標 |
| DR-2 | `required = N_used × U × (fileSR_max/sr_min) × 8 = 557.28 MiB`（→ 1 GiB） | **P1 と P3** |
| DR-3 | `K ≤ 10`（T1 ≤ byte 予算の 1 %） | T1 を予算の 1 % に収めるという基準 + B = 1 GiB |
| DR-4 | FC-FORM-5（`L_res ≤ U`）を追加すれば R-1 は有界（C_res ≤ 256 MiB @N=8） | Case B + 閉塞手段としての admission |
| DR-5 | `L_res ≤ U` は `duration ≤ hardMaxSec(sr)` と同値 | PF-7, PF-10 |
| DR-6 | hash 経路は O(1) 化 (b) または byte admission (d) のいずれかが必要 | FC-INV-1 + FC-INV-5 + fileSize が fileLength と別次元であること |

### 9.3 POLICY CHOICE（コードからは決まらず、裁定が必要）

| ID | 内容 | 選択肢 |
|---|---|---|
| **PC-1** | **前提 P1**: 「完全に使い切れるソースは必ず受け入れる」を採用するか | 採用 / 不採用（§3.6） |
| **PC-2** | **前提 P3**: IR ファイル SR の包絡をどこまで認めるか | 768k / 384k / 192k / その他 |
| **PC-3** | 丸め規則（required 以上の最小 2 冪） | 維持 / 他の丸め |
| **PC-4** | **`kMaxIRLoadChannels` の値** | 2 / 8 / その他 / 無制限（byte bound のみ） |
| **PC-5** | SR-01B §境界条件 `channel count` 行を supersede するか | supersede する / 現行維持（K 無制限） |
| **PC-6** | **`kMaxIRResampleOutputSamples`（FC-FORM-5 の値とマージン）** | U / U+1 / 独自値 |
| **PC-7** | FC-FORM-5 を採るか（Case B を閉じるか） | 採る / 採らない（→ DESIGN-DEFERRED 相当） |
| **PC-8** | FC-8 の閉塞方式 | (b) streaming / (d) byte admission / 両方 |
| **PC-9** | preview 経路の graceful failure を本 work に含めるか | 含める / 別 work 起票 |

---

## 10. Revised Frozen Contract（裁定待ちの改訂案）

### 10.1 契約形（改訂）

```text
FC-FORM-1  byte admission
    numChannels × fileLength × sizeof(double) ≤ kMaxIRLoadBytes
FC-FORM-2  channel admission
    numChannels ≤ kMaxIRLoadChannels
FC-FORM-3  representation precondition
    fileLength ≤ MAX_FILE_LENGTH = 2,147,483,647
FC-FORM-4  degenerate precondition
    numChannels ≥ 1  ∧  fileLength ≥ 1
FC-FORM-5  ★新規 resample-output admission（R-1 閉塞）
    (double)fileLength / fileSR  ≤  (double)kMaxIRResampleOutputSamples / sr
    （fileSR ≤ 0 のとき resample は実行されないため vacuous）
FC-FORM-6  ★新規 hash-path precondition（FC-8）
    computeIRHash は O(1) メモリであるか、または物理ファイルサイズの
    admission 後にのみ実行されること
```

### 10.2 値（すべて裁定待ち。括弧内は FC 文書の暫定値）

```text
kMaxIRLoadBytes              = 1,073,741,824 B (1 GiB)   [PC-1, PC-2, PC-3 に依存]
kMaxIRLoadChannels           = 8                          [PC-4 に依存]
kMaxIRResampleOutputSamples  = 2,097,152 (= U) [+margin]  [PC-6 に依存]
MAX_FILE_LENGTH              = 2,147,483,647              （既存維持）
```

### 10.3 改訂後の有界量（すべて admission で拘束）

| 記号 | 実体 | 位置 | 上限 | 拘束する形 |
|---|---|---|---|---|
| C_src | `loadedIR` | `:422` | 1,024 MiB | FC-FORM-1 |
| T1 | `tempFloatBuffer` | `:414` | 8 MiB | FC-FORM-2 |
| T2 | `tempAligned` | `:415` | 2 MiB | 固定 |
| **C_res** | **resample chData + result** | `ResampleAndFallback.cpp:46-93` | **256 MiB (N=8)** | **FC-FORM-5** |
| stepTrimmed | N × targetLength × 8 | `:586` | 128 MiB | FC-FORM-2 + PF-3 |
| irL / irR | 2 × targetLength × 8 | `:163-164` | 32 MiB | PF-3 |
| hash buffer | `computeIRHash` | `AllpassDesigner.cpp:626` | O(1) or ≤ 1 GiB | FC-FORM-6 |

**改訂後の whole-path worst resident ≒ 1,448 MiB**（FC 文書の 1,194 MiB から、
R-1 を正しく計上して 254 MiB 増）。

**対比**:

| 構成 | 現行 | FC 文書（改訂前） | R1 改訂案 |
|---|---|---|---|
| 2ch × INT32_MAX（loadedIR） | 34.36 GB | ≤ 1 GiB | ≤ 1 GiB |
| リサンプル中間 worst | 無制限 | **37.40 GB（未拘束）** | **≤ 256 MiB** |
| whole-path worst | 非有界 | **≒ 38.4 GB** | **≒ 1.45 GiB** |

### 10.4 不変条件の改訂

```text
FC-INV-1  [Contract] admission が allocation に先行する。
          ★ 対象を「O(fileLength) および O(file bytes) の全確保」に**拡張**する。
            現行は computeIRHash で FAIL（PF-11）。
FC-INV-2  [Contract] 許容される最大 allocation は有限である。
          ★ 「loadedIR のみ」から **「IR ロード経路の全 allocation」に改訂**（Case B 採用）。
            FC 文書の "loadedIR に限定する" という限定は撤回する。
FC-INV-3  [Impl]     MAX_IR_LATENCY は DSP usable length の上限であり、allocation bound ではない。
          ※ FC-FORM-5 は **output-length（utility）bound** であり allocation bound ではない。
             FC-FORM-5 から導出される allocation（N × (U+1) × 8 ≤ 128 MiB）が
             FC-FORM-1 とは独立であることを明記することで非統合を維持する。
FC-INV-4  [Contract] 上限超過は allocation 前に deterministic reject する。
FC-INV-5  [Contract] allocation failure は graceful failure とする。
          ★ 対象に preview 経路を含める（現行 FAIL、PF-13）。
FC-INV-6  [Impl]     既存の chunked file read を維持する。
FC-INV-7  [Impl]     streaming / incremental partition build は本 work の scope 外。
FC-INV-8  [Impl]     診断は単一情報源とする。
FC-INV-9  [Impl]     admission 順序:
              1. FC-FORM-4  2. FC-FORM-3  3. FC-FORM-2  4. FC-FORM-1  5. FC-FORM-5
              6. ★ ここより後に O(file bytes) / O(fileLength) の確保を開始する（FC-FORM-6）
FC-INV-10 [Impl]     FC-FORM-5 は **resampler 構築より前**に、**64-bit 以上の算術で**評価する。
              理由: r8b の `getMaxOutLen` は `(int) ceil(MaxInLen × DstSR / SrcSR) + 1`
              （CDSPFracInterpolator.h:831）であり、N=1・fileLength=134,217,728・
              fileSR=44.1k・sr=768k では 2,337,397,169 > INT32_MAX となり **int 変換が溢れる**
              （当該ファイルの duration = 3,043.5 s に対し hardMaxSec(768k) = 2.73 s）。
              double 比較（duration 形式）で admission 時に除外すれば
              この溢れは r8b に到達しない。
```

### 10.5 compatibility change 一覧（改訂）

| ID | 内容 | 性質 | 実害評価 |
|---|---|---|---|
| **CC-1** | `N > 8` を新規拒否 | **既凍結 SR-01B 境界条件の改訂**（§4.4） | 外部情報上、実ファイル存在性は無視できるが、proof には使わない |
| **CC-2** | `N ≥ 4` かつ高 SR 極長ファイルを新規拒否 | 新規拒否 | §8.3。mono/stereo/3ch は全 SR で影響なし |
| **CC-3** | mono / stereo / 3ch は全対応 SR（44.1k–768k）で互換 | 互換（変更なし） | — |
| **CC-4** | **FC-FORM-5 により `duration > hardMaxSec(sr)` を新規拒否** | 新規拒否 | interactive 経路は変更なし。**CLI / state 復元のみ**新規拒否（§8.4）。705.6k/768k では 3 秒 IR が該当 |
| **CC-5** | 上記 CC-1/CC-2/CC-4 は **policy 裁定（PC-1〜PC-7）に依存**する | 条件付き | 裁定確定後に再確定する |

---

## 11. Implementation Gate の前提条件

```text
R1-D1 [必須] P1（「完全に使い切れるソースを必ず受け入れる」）を採用するか裁定する。
      不採用なら kMaxIRLoadBytes の導出式が変わる（§3.6）。
R1-D2 [必須] P3（IR ファイル SR 包絡）を裁定する。値が最大 4 倍動く。
R1-D3 [必須] K を裁定する（PC-4）。併せて SR-01B §境界条件 channel count 行の
      supersede を承認する（PC-5）。これがない限り §8.3 の matrix は proof にならない。
R1-D4 [必須] FC-FORM-5 を採用するか裁定する（PC-7）。
      不採用の場合、IR ロード経路は 37.40 GB の確保を試行しうる（§5.3）。
      → 判定は DESIGN-DEFERRED のままとなる。
R1-D5 [必須] FC-8 の閉塞方式を裁定する（PC-8）。必要条件は §6.2 のとおり
      (b) 単独 or (d) 単独。(a)/(c) 単独では不十分。
R1-D6 [推奨] preview 経路の graceful failure を本 work に含めるか、別 work 起票するか（PC-9）。
      現行は FC-INV-5 FAIL（§7.2）。黙って残さないこと。
R1-D7 [推奨] FC 文書 §4.6 の「worst resident ≒ 1,194 MiB」を
      「≒ 1,448 MiB（改訂後）」または「≒ 38.4 GB（改訂前）」へ訂正する。

禁止（変更 0）:
  M-03 D3 / M-03 latency・PDC / Direct Head HC/LC / H-01 / H-02 / SR-01〜03 /
  Publish・Crossfade・Retire authority / Epoch / RuntimeWorld / Lifetime Budget / RT path /
  chunked read 撤去 / incremental partition build / IRState 表現変更 /
  ch0-ch1 のみ保持（FC-CH-3）/ UI / CMake / ConvoPeq.md 再生成 / README E-G3-1 差分
```

---

## 12. Final classification

```text
最終判定: DESIGN-DEFERRED
```

**判定理由（ユーザーの判定表との対応）**:

| 条件 | 該当 | 本 R1 の結論 |
|---|---|---|
| 1 GiB と 8ch がコード/既存仕様から十分に正当化でき、R-1 scope も明確 → CONTRACT-FROZEN | **✗** | P1/P3/K はいずれもコード証明でなく policy（§3,§4）。R-1 は scope 外にできない（§5.4） |
| 値は policy choice だが、明示的な裁定だけ残る → CONTRACT-HOLD | **部分的に該当** | 値は確かに policy。ただし HOLD だけでは不十分（下記） |
| **loader 全体の bounded allocation を要求するなら resample bound 等が必要 → DESIGN-DEFERRED** | **✓ 該当** | §5.3 のとおり、改訂前契約の内側で **37.40 GB** に到達し、**原欠陥 34.36 GB を上回る**。Case B を要求として確定 → R-1 の閉塞（FC-FORM-5）が必要 |
| 現契約では安全性を保証できない → NO-GO | ✗ | FC-FORM-5 により有界化可能（§10.3）。安全性は保証可能 |

**結論**:

1. **FC-FORM-1/2/3 の「形」は妥当**（目標 → 形の導出として維持）。
2. **数値（`1 GiB` / `8ch`）は POLICY CHOICE** であり、導出値ではない。
   前提 P1・P3 を明示し、裁定を得る必要がある（PC-1〜PC-5）。
3. **R-1 は scope 外にできない。** これを閉じるには新たな admission 次元
   **FC-FORM-5** が必要であり、その追加は「契約形の拡張」である。
   → この擴張が必要であることが **DESIGN-DEFERRED** の主因である。
4. 閉塞手段そのもの（FC-FORM-5）は admission チェックであり、
   リサンプラの順序変更・ストリーミング化といった**設計変更を要しない**。
   したがって R1-D1〜R1-D5 の裁定が得られれば、次回で CONTRACT-FROZEN に到達しうる。
5. FC-8（`computeIRHash`）は R1 でも **実装要求として維持**する（§6）。
   必要条件は (b) streaming または (d) byte admission のいずれか。

**R1 はここで停止する。Implementation Gate（WORK102-IG）へは進まない。**

---

## 13. 監査メタ

```text
検証方法    : LoaderThread.cpp / LoadPipeline.cpp / StateAndUI.cpp / Runtime.cpp /
              ResampleAndFallback.cpp / AllpassDesigner.cpp / ConvolverControlPanel.cpp /
              MainWindow.cpp / ConvolverProcessor.h / AlignedAllocation.h /
              juce_HeapBlock.h / juce_ThreadPool.cpp / juce_AudioSampleBuffer.h /
              r8brain-free-src/CDSPFracInterpolator.h の行単位実測
              + 容量 3 段の再計算 + R-1 worst case の数値導出
              + doc/work95（SR-01B 凍結）との cross-check
              + src/ doc/ 全走査（P1 の出典が存在しないことの確認）

Production 変更     : 0
CMake 変更          : 0
UI 変更             : 0
tests 変更          : 0
ConvoPeq.md 再生成  : 0（FRESH 維持）
commit / push       : 0
README E-G3-1 差分  : 維持
```

### FC 文書からの主要訂正（要約）

| # | FC 文書の記述 | R1 の訂正 |
|---|---|---|
| 1 | §4.2「外部仮定を置かない」 | **誤り**。P1（完全利用の受入義務）を無ラベルで導入していた。P3 も policy |
| 2 | §4.3「K = 8 は実在する IR を拒否しない最小値」 | **不適切**。導出値ではなく (i) 予算基準 + (ii) 外部情報の合成 = **policy value** |
| 3 | §4.6「worst resident 合計 ≒ 1,194 MiB」 | **過小評価**。リサンプル中間を無視。正しくは改訂前 ≒ 38.4 GB / 改訂後 ≒ 1.45 GiB |
| 4 | §4.6 R-1 を「本契約の対象外」 | **scope 外にできない**（37.40 GB > 原欠陥 34.36 GB）。Case B として FC-FORM-5 を追加 |
| 5 | §6.4「N > 8 の IR は製品入力として存在しない」を互換根拠に使用 | **外部情報**。proof に使わず、SR-01B 凍結境界条件の**改訂**として計上 |
| 6 | §6.4 の matrix を compatibility proof として提示 | **条件付き帰結**。K の policy 裁定が先行することを明記 |
| 7 | §12 判定 CONTRACT-FROZEN | **DESIGN-DEFERRED**（本 R1） |
| 8 | （未記載）preview 経路の graceful failure | **新規指摘**: 現行 FC-INV-5 FAIL（job 例外で `irPreviewInProgress` が恒久 true） |
| 9 | （未記載）FC-FORM-5 の int 溢れ | **新規指摘**: r8b `getMaxOutLen` の `(int)` 変換が N=1 極長で溢れる → 64-bit/double で事前判定 |
| 10 | §7 D-2 で `MAX_IR_LATENCY` 非統合を明記 | 維持。ただし FC-FORM-5 が output-length bound であることを明示し、allocation bound（FC-FORM-1）と独立であることを追記 |

### FC-8 の扱い（ユーザー指示の遵守確認）

```text
「FC-8 computeIRHash は必ず implementation requirement として維持する」
→ 維持。§6 で (a)/(b)/(c)/(d) の契約上の要否を再整理し、
   必要条件 = (b) 単独 or (d) 単独 と確定した。(a)/(c) 単独では不十分。
```
