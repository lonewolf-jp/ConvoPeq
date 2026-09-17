# WORK102-FC-R2 — Policy / Contract Arbitration & Revised Contract Freeze（read-only）

- **作成日**: 2026-09-17
- **種別**: Contract Arbitration（**実装は行わない**）
- **目的**: R1 で未確定になった policy と FC-FORM-5/6 を裁定し、**実装可能な契約へ凍結する**
- **被監査対象**: `big18_pre_audit_*` → `big18_failure_contract_*` → `big18_failure_contract_reconciliation_*`
- **前例フォーマット**: `doc/work98/m02_pre_audit_failure_contract_freeze_20260915.md`

> **R2 の結論（先出し）**
>
> - R1 の未決 12 項目を**すべて裁定した**。3 層（PROVEN FACT / DERIVED REQUIREMENT / POLICY CHOICE）を維持。
> - 中核の裁定: **R2-D2 = IR ファイル SR 包絡 44.1 kHz–768 kHz を新規契約事項として明示**。
>   これにより `kMaxIRLoadBytes = 1 GiB` が「無ラベルの前提」ではなく
>   **明示された envelope からの DERIVED REQUIREMENT** になる（R1 の批判を閉じる）。
> - **R2-D3 = Option B（`kMaxIRLoadChannels = 8`）+ SR-01B supersede を契約化**。
> - **R2-D4 = FC-FORM-5 採用。値は `U`（DSP usable length）に整合、`+1` のテールを許容。
>   適用点は「末尾無音トリム後・リサンプラ構築前」**（raw `fileLength` ではない）。
> - **R2-D5 = FC-8 は (b) streaming hash を採用**（(d) は新規 policy 値を要し、確保を消さず上限化に留まるため不採用）。
> - **Preview は WORK102-PREV-01 として分離。ただし FAIL は消さない**（OPEN として明示追跡）。
> - **R2 の新規実測 2 件**:
>   (i) FC-FORM-5 を raw `fileLength` に適用すると**無音パディングされたファイルを不当に拒否**する
>       （UI は実効長で判定するため受理する）→ **trim 後・resample 前**に適用する。
>   (ii) whole-path の同時滞在ピークは **2 × kMaxIRLoadBytes = 2 GiB**。
>       JUCE `AudioBuffer::setSize(..., keepExistingContent=true)` と `shrinkToFit` が
>       「新規確保 → copy → swap」を行うため、trim 局面で旧+新が同居する。R1 の 1.45 GiB は誤り。
>
> **最終判定: CONTRACT-FROZEN**（§13。governance countersign 2 件を IG 前提条件として明示）

---

## 1. Baseline（再確認）

```text
HEAD        = 35461e2e3a5fd42fd009416a20adaf457b9b4031
origin/main = 35461e2e3a5fd42fd009416a20adaf457b9b4031   (一致)

ConvoPeq.md  mtime = 2026-09-17 00:07:39 JST（本 R2 でも不変。再生成なし）
src/ のうち ConvoPeq.md より新しいもの = 0 件（FRESH 維持）

参照した凍結文書:
  doc/work95/sr01b_implementation_gate_contract_freeze_20260914.md （SR-01B 境界条件）
  doc/work101/work92_rt_residual_runtime_bug_retriage_20260916.md  （big 1-8 の位置付け）
  doc/work102/big18_pre_audit_bounded_allocation_contract_20260917.md
  doc/work102/big18_failure_contract_20260917.md                    （FC 文書）
  doc/work102/big18_failure_contract_reconciliation_20260917.md     （R1 文書）
  doc/Practical Stable ISR Bridge Runtime.md    （RT/NonRT 分離原則）

本 R2 の変更:
  doc/work102/big18_failure_contract_arbitration_20260917.md の新規作成のみ
  production source = 0 / tests = 0 / CMake = 0 / UI = 0
  ConvoPeq.md 再生成 = 0 / commit = 0 / push = 0
  README.md E-G3-1 差分 = 維持
```

### 1.1 一次ソース（ConvoPeq.md / src）での順序再確認

`ConvolverProcessor.LoaderThread.cpp::doLoadIRStep`（行番号実測）:

```text
:364      computeIRHash(file)                        ← ① O(file bytes) 確保（admission 前）
:389-390  fileLength / numChannels 取得
:391-397  MAX_FILE_LENGTH 判定                       ← ② 既存 admission
:398-402  numChannels <= 0 判定                      ← ③ 既存 admission
:414-415  tempFloatBuffer / tempAligned              ← ④⑤ transient
:422      loadedIR.setSize(N, fileLength)            ← ⑥ ピーク確保
```

`R1 の指摘どおり「hash → admission → allocation」の順序であり、byte 次元の admission は 0 箇所。`
別経路 `loadImpulseResponsePreviewFile`（`ResampleAndFallback.cpp:271-331`）も同一の `MAX_FILE_LENGTH` のみを持ち、
`:307` に**非チャンクの全長 float バッファ**を持つ。

### 1.2 RT/NonRT 原則（維持事項）

`doc/Practical Stable ISR Bridge Runtime.md:3`

```text
「RTスレッドは絶対に待たない。絶対に解放しない。絶対に判断しない。
  すべての危険操作はNonRTへ橋渡し(Bridge)し、状態遷移は観測可能で、停止時には完全排水(Drain)を保証する」
```

同 `:704`「RTスレッドを『実行主体』ではなく『観測主体』に限定し、すべての寿命管理・状態遷移・障害回復を
NonRT側へ隔離すること」。

**本契約は IR ロード（`LoaderThread` = NonRT、`ResampleAndFallback` = NonRT / thread pool）のみを対象とし、
RT path へは一切波及しない。** これは §11 の禁止事項に含める。

---

## 2. 裁定の方法と 3 層の維持

R2 は各項目を次の手順で裁定した:

```text
手順 1  その量はコード／凍結済み文書から証明できるか        → PROVEN FACT
手順 2  明示した前提の下で必然となる要求は何か              → DERIVED REQUIREMENT
手順 3  コードから決まらない選択は何か（誰が決めるか）      → POLICY CHOICE
手順 4  POLICY CHOICE を本 R2 で決定し、根拠と反転条件を記録
```

**R2 の裁定は「決定」であり、「無ラベルの前提」ではない。** 各裁定に ID（R2-D1..D6）を与え、
「決定内容 / 根拠 / 反転した場合の影響」を明記する（§3〜§8）。
R1 が指摘した「導出値と policy の混同」は、この形で解消する。

---

## 3. R2-D1 — P1 の再定義（契約目的としての明確化）

### 3.1 R1 の指摘の確認

R1 §3.1 のとおり、`src/` と `doc/` の全走査で
「DSP usable window を完全に利用できる source を loader が必ず受け入れる」を要求する記述は **0 件**。
現行コードが要求するのは `fileLength ≤ INT32_MAX` と `numChannels ≥ 1` のみ。

**→ P1 は既存仕様ではなく、採用するなら新規の product compatibility requirement。**

### 3.2 R1-D1 の分離要求への回答

| 論点 | 回答 |
|---|---|
| これは既存仕様か | **いいえ。** 現行コードは INT32_MAX まで無条件受理する（byte 次元の要件は 0 件） |
| 採用する場合の性質 | **新規 product compatibility requirement**（「この SR 包絡内の完全利用可能ファイルは拒否しない」という対外約束） |
| 採用しない場合 | `kMaxIRLoadBytes` を「完全利用保証」から導出してはならない。別の policy（例: memory safety のみ）へ切り替える |
| memory safety を最低契約とする場合 | byte bound の値は「完全利用」と無関係に選ぶ。互換の喪失は CC として別途明示する |

### 3.3 R2 の裁定（P1 の再定義）

P1 を単独の前提として置かず、**R2-D2 の envelope から従属的に導出される要求へ再定義する**。

```text
R2-D1（決定）
  P1 を「無制約の完全利用保証」として採用しない。
  代わりに次の条件付き requirement として採用する:

    IR ファイル SR が [44.1 kHz, 768 kHz]（= R2-D2 の envelope）にあり、
    かつ duration ≤ hardMaxSec(sr)（= FC-FORM-5 を満たす）であるファイルは、
    loader は受理しなければならない。

  これは「無条件の完全利用保証」ではなく
  「明示された envelope 内での完全利用保証」である。
```

**効果**: `kMaxIRLoadBytes` の導出は
「任意の fileSR を仮定した policy」ではなく
「**R2-D2 で明示宣言した envelope からの DERIVED REQUIREMENT**」になる。
R1 が指摘した「無ラベルの前提」状態は解消する。

**反転した場合の影響**: envelope を狭めれば byte bound は小さくなるが、
envelope 外の高 SR ファイルが新規拒否される（CC が増える）。§4 で比較する。

---

## 4. R2-D2 — IR ファイル SR 包絡の裁定

### 4.1 論理的な非同一性の確認（ユーザー指摘の確認）

```text
PROVEN:   ConvoPeq の processing SR 範囲 = 44.1 kHz – 768 kHz
          （SAFE_MAX_SAMPLE_RATE = 768000.0 / kMaxInternalRate = 768000。
            doc/audit/ConvoPeq_SampleRate_Support_Audit_2026-09-13.md:41,175）

NOT PROVEN: IR *ファイル* の SR が 768 kHz まで存在しうること。
            現行コードは reader->sampleRate を検証しない（0 より大きければ受理）。
            doc/ に IR ファイル SR の上限を述べた記述は 0 件。
```

**→ 「processing SR の上限 = IR ファイル SR の上限」は論理的に同一ではない。**（R1-D2 の指摘どおり）

### 4.2 候補比較

| 候補 | envelope | required = 2 × U × (E/44100) × 8 | 値（最小 2 冪） | 新規拒否される範囲 |
|---|---|---|---|---|
| **P3-A** | 192 kHz | 139.33 MiB | **256 MiB** | 384 kHz / 705.6 kHz / 768 kHz の完全利用ファイル |
| **P3-B** | 384 kHz | 278.64 MiB | **512 MiB** | 705.6 kHz / 768 kHz の完全利用ファイル |
| **P3-C** | 768 kHz | 557.28 MiB | **1 GiB** | なし（envelope 全域をカバー） |
| **P3-D** | その他 | 任意 | 任意 | 任意 |

**P3-D を採る場合の注意**: `E` を 768 kHz 超にしても `sr_min = 44.1 kHz` が下限なので
`E/sr_min` が増えるだけで、required は線形に増加する。上限を設けないと byte bound は無意味になる。

### 4.3 R2 の裁定

```text
R2-D2（決定）

  IR file sample-rate envelope = 44.1 kHz – 768 kHz

  これは ConvoPeq の **新規契約事項** である（既存仕様ではない）。
  processing SR 範囲と数値は一致するが、論理的に独立した宣言である。

  根拠:
    (a) 製品が宣言する SR 範囲と対称にすることで、外部データ（IR ライブラリの実勢 SR）に
        依存しない自己完結した契約になる。R1 が「外部情報を proof に使うな」と指摘した点に整合。
    (b) envelope の下端 44.1 kHz は processing SR の下端として既に証明済み（PF-9）。
    (c) P3-A/P3-B は「IR ライブラリの実勢上限」という外部情報を必要とする。
        R2 はこれを採用しない（コード外情報を契約根拠にしない方針）。
    (d) envelope を processing SR 範囲と一致させると、
        「製品が対応を宣言しているレートの IR はすべて扱える」という説明可能性が得られる。

  帰結: kMaxIRLoadBytes = 557.28 MiB 以上の最小 2 冪 = 1,073,741,824 B (1 GiB)
        これは R2-D2 からの DERIVED REQUIREMENT であり、policy ではない。
```

**反転した場合の影響**: P3-B を採ると `kMaxIRLoadBytes = 512 MiB`、P3-A なら `256 MiB`。
その場合「705.6/768 kHz の完全利用ファイル」が新規拒否され、CC が 1 件増える。
**値は R2-D2 のみに依存する**（他は不変）。

---

## 5. R2-D3 — channel policy の裁定

### 5.1 ソースから証明できる範囲

```text
PROVEN (A): N ≥ 1 で受理される（:398-402 は <= 0 のみを弾く）
PROVEN (B): build が使用するのは ch0/ch1 の 2ch（:166-167）。N≥3 は dead weight
PROVEN (C): SR-01B が「>2ch→先頭2」を **凍結境界条件** として明記（上限なしで受理）
NOT PROVEN (D): 実在する IR ファイルのチャンネル数（外部情報）
```

### 5.2 候補比較（ユーザー提示 A/B/C）

| 候補 | 内容 | T1 = N × 262,144 × 4 | bounded-allocation 目標 | SR-01B への影響 |
|---|---|---|---|---|
| **Option A** | `K = 2` | 2 MiB | 成立 | **>2ch を新規拒否 = 凍結条件を大きく改訂** |
| **Option B** | `K = 8` | 8 MiB | 成立 | N≤8 で凍結条件を維持、**N>8 のみ supersede** |
| **Option C** | 上限なし | **非有界** | **不成立** | 凍結条件を完全維持 |

**Option C の棄却（証明）**: byte bound だけでは N を実効的に拘束できない。
`N × fileLength × 8 ≤ B` かつ `fileLength ≥ 1` から得られるのは `N ≤ B/8 = 134,217,728` であり、
`T1 ≤ 134,217,728 × 1 MiB ≒ 137 TB`。WAV のチャンネルフィールド上限 65,535 を考慮しても
`T1 ≤ 64 GiB`。**いずれも bounded allocation 目標と両立しない。**
→ R1 の判定（Option C 不成立）を確認・維持する。

### 5.3 R2 の裁定

```text
R2-D3（決定）

  Option B を採用する。

  kMaxIRLoadChannels = 8          ← ★ policy value（導出値ではない）

  根拠:
    (1) Option A は SR-01B の凍結条件「>2ch→先頭2」を N ∈ [3,∞) で全面改訂する。
        Option B は N ∈ [3,8] でこれを維持し、改訂範囲を最小にできる。
    (2) K = 8 は T1 を 8 MiB、stepTrimmed を 128 MiB に収める。
        いずれも FC-FORM-1 の C_src 上限 (1 GiB) より十分小さい。
    (3) 8 は「parse 可能な最大の一般的マルチチャンネル IR 配置」に合わせた選択であり、
        **導出値ではなく policy value である**。根拠 (3) は外部情報を含むため、
        契約文書では policy と明記し、proof には使用しない。

  ★ SR-01B supersede（必須・契約化）

    doc/work95/sr01b_implementation_gate_contract_freeze_20260914.md の
    「### 境界条件（凍結）」表の
      | channel count | 2ch 正規化（mono→複製、>2ch→先頭2）現行不変（LoaderThread.cpp:166-169） |
    を次のとおり改訂する（Implementation Gate で適用）:

      | channel count | 2ch 正規化（mono→複製、>2ch→先頭2）。
                       ただし numChannels > kMaxIRLoadChannels (=8) は loader admission で reject
                       （big 1-8 Failure Contract FC-FORM-2）。|

    = N ≤ 8 では凍結済み挙動を維持し、N > 8 のみ新規拒否。
    この supersede は本 R2 の裁定に含まれるが、既凍結契約の正式改訂であるため
    §13 の countersign 対象とする。
```

**反転した場合の影響**: Option A を採ると N ∈ [3,8] の IR も拒否され CC-1 が拡大する。
Option C は bounded allocation 目標と両立しないため選択不可。

---

## 6. R2-D4 — FC-FORM-5 の裁定（R-1 の閉塞）

### 6.1 採用の可否（R1 の 37.40 GB 問題）

R1 §5.3 の実測:

```text
FC-FORM-1 を満たす範囲で（N=2, C_src = 1 GiB, fileSR=44.1k, sr=768k）
  L_res = 1,168,698,585 → C_res（chData + result 同時）= 37.40 GB
  → big 1-8 の原欠陥 34.36 GB を上回る
```

**→ FC-FORM-5 を採用する。**（ユーザー指示の方向と一致）

### 6.2 ★ 適用点の裁定（R2 の新規実測）

R1 は FC-FORM-5 を **raw `fileLength`** に対して書いていた。R2 の実測でこれは不適切と判明した。

```text
doTrimStep の順序（行番号実測）:
  :465-517  末尾無音トリム（threshold 1e-15、末尾から走査）
            → newLength < numSamples のとき setSize(N, max(1,newLength), true) + shrinkToFit
  :519-553  リサンプル（loadedSR != sampleRate のとき）
            → resampleIR(stepResult.loadedIR, ...)

すなわち **リサンプラの入力は「トリム後」の長さ L_trim（≤ fileLength）** である。
```

**raw fileLength に適用した場合の問題**:

```text
例: 10 分のファイルの先頭 1 秒に IR、残りは完全無音（値 0）
  - 現行: loadedIR を確保 → トリムで 1 秒へ縮小 → リサンプル成功 → 正常動作
  - UI  : estimateEffectiveIRLengthSamples は「実効長」を測るため 1 秒と判定 → 受理
  - raw 適用: duration = 600 s > hardMaxSec(768k) = 2.73 s → **拒否**
  → UI が受理するファイルを loader が拒否する **新規不整合**（回帰）
```

**→ 適用点は「末尾無音トリム後・リサンプラ構築前」とする。**

### 6.3 適用点の契約化

```text
R2-D4（決定）

  FC-FORM-5 は次の位置で評価する:

    ① doLoadIRStep で FC-FORM-1/2/3/4 を評価（O(fileLength) 確保の前）
    ② loadedIR を確保しチャンク読込
    ③ doTrimStep :465-517 で末尾無音トリム（L_trim 確定）
    ④ ★ FC-FORM-5 を L_trim に対して評価（:517 と :519 の間）
    ⑤ リサンプラ構築（:527）と r8b getMaxOutLen の (int) 変換

  すなわち FC-FORM-5 は「resampler construction / int conversion より前」にあり、
  かつ「trim 後」である。raw fileLength では評価しない。

  実装前提（FC-INV-10 として凍結）:
    FC-FORM-5 の比較は **double で行い、64-bit を超える中間値を生成しない**。
      (double)L_trim / fileSR  ≤  (double)kMaxIRResampleOutputSamples / sr
    理由: r8b の getMaxOutLen(MaxInLen, DstSR, SrcSR) は
          (int) ceil(MaxInLen × DstSR / SrcSR) + 1 を返し（CDSPFracInterpolator.h:831）、
          N=1・L=134,217,728・fileSR=44.1k・sr=768k では 2,337,397,169 > INT32_MAX となり
          int 変換が溢れる。FC-FORM-5 を先に評価すればこの入力は r8b に到達しない。
```

### 6.4 値の裁定

| 候補 | 意味 | C_res（N=8）= 2 × N × v × 8 | 判定 |
|---|---|---|---|
| `v = U` | DSP usable length。`L_res ≤ U` は「リサンプル結果が DSP 窓に収まる」 | 256 MiB | 境界で `+1` のテールが落ちる可能性 |
| **`v = U + 1`** | r8b の `+1` テールを許容 | 256 MiB（256 MiB + 16 B） | **採用** |
| `v > U + 1` | 無駄を許容 | 増加 | 不採用（DSP は `U` で trim するため純粋な無駄） |

```text
R2-D4（値）
  kMaxIRResampleOutputSamples = U + 1 = 2,097,153

  根拠:
    (a) DSP は targetLength = min(sr × irLenSec, U) ≤ U でしか使わない（PF-3）。
        L_res > U + 1 の部分は必ず捨てられる。
    (b) r8b の契約が ceil(...) + 1 であるため、U ちょうどの duration で境界を落とさないには +1 が必要。
    (c) 値は MAX_IR_LATENCY に整合するが、これは **output length（utility）bound** であり
        allocation bound ではない。FC-FORM-5 から導出される allocation
        （N × (U+1) × 8 ≤ 128 MiB/チャンネル）は FC-FORM-1 とは独立である（§11 FC-INV-3）。
    (d) 既に SR-01B が凍結した hardMaxSec(sr) = U/sr と同値であり、
        **新規数値を発明していない**。
```

### 6.5 FC-FORM-5 の互換効果（R1 の CC-4 を精緻化）

FC-FORM-5 は「UI が既に拒否している条件」を loader にも適用するものである。

| 経路 | 現行 | FC-FORM-5 適用後 | 差分 |
|---|---|---|---|
| interactive ControlPanel | `exceedsHardLimit` で拒否済（`:1202-1220`） | 同じ | **なし** |
| preview (`analyzeImpulseResponseFile`) | `exceedsHardLimit` で拒否済 | 同じ | **なし** |
| CLI (`MainWindow.cpp:796,838`) | 受理（暗黙 trim + SR-01 C-3 ログ） | **拒否** | **新規** |
| state 復元 (`StateAndUI.cpp:417`) | 同上 | **拒否** | **新規** |

**→ FC-FORM-5 は「loader を UI と同じ判定へ揃える」変更であり、新規の policy 判断ではない。**
ただし CLI / state 復元 の 2 経路では挙動が変わるため CC-4 として計上する（§10.4）。

---

## 7. R2-D5 — FC-8（`computeIRHash`）の裁定

### 7.1 前提（R1 から維持）

```text
現行順序: computeIRHash (:364) → fileLength/channel admission (:391-402) → allocation (:422)
FC-INV-1（admission が先行）: 現行 FAIL
FC-INV-5（graceful failure）: 現行 FAIL
  - HeapBlock<uint8_t> は throwOnFailure=false（juce_HeapBlock.h:43）→ 失敗時 nullptr、throw しない
  - 直後の memcpy(fileData.getData() + writePos, ...)（:643）が null 書込
  - 例外モデルは /EHsc（CMakeLists.txt:1522,1527,1617。EHa は 0 件）→ SEH は catch(...) に翻訳されない
```

R1 の必要条件判定は **維持**する: `(a)` 単独・`(c)` 単独は不十分。

### 7.2 戦略比較（ユーザー要求の表を R2 で確定）

| 方法 | 新規 policy 値 | 実装影響 | boundedness | 確保の性質 | 判定 |
|---|---|---|---|---|---|
| **(b) streaming hash** | **なし** | 中（`computeIRHash` の書き換え） | **O(1)** | **確保を消す** | **○ 採用** |
| **(d) file-size admission** | **必要**（`kMaxIRFileBytes`） | 小（チェック 1 箇所） | 有界 | **確保は残る（上限化のみ）** | △ 不採用 |
| (a) hash を後方へ移動 | なし | 小 | **不変（非有界）** | 確保は残る | ✗ 不十分 |
| (c) nullptr 検査 | なし | 小 | **不変（非有界）** | 確保は残る | ✗ 不十分 |
| (a) + (c) | なし | 小 | **不変（非有界）** | 確保は残る | ✗ 不十分 |

**「小変更だから (d)」を選ばない理由（R2 の明示）**:

```text
(1) (d) は確保を **消さない**。kMaxIRFileBytes を kMaxIRLoadBytes と同値にすると、
    最悪ケースで hash バッファ (≤1 GiB) と loadedIR (≤1 GiB) が **同時に** 存在しうるため、
    whole-path ピークが最大 1 GiB 増える（§9 の位相表に影響）。
(2) (d) は新規 policy 値を 1 つ増やし、その値の根拠を別途用意する必要がある
    （R1/R2 が「値の根拠なき policy 追加」を繰り返し問題視してきた）。
(3) (b) は確保そのものを除去するため、policy 値も位相ピークの増加も生じない。
```

### 7.3 裁定

```text
R2-D5（決定）

  FC-8 は (b) streaming hash を採用する。

  契約形 FC-FORM-6:
    computeIRHash は物理ファイルサイズに比例する allocation を行ってはならない（O(1) メモリ）。
    かつ admission 順序（FC-INV-9）に従うこと。

  等価性の確認（R2 で検証）:
    現行実装は「全ファイルを buffer へ読み → buffer を hash → before/after の size+mtime 検証
    （AllpassDesigner.cpp:655）」である。TOCTOU 保護は size+mtime 検証が担っており、
    buffer は「hash 対象のスナップショット」を作るためだけに存在する。
    逐次 hash でも同一の digest が得られ、同じ size+mtime 検証が同じ保証を与える。
    → **挙動等価**。
```

**副次効果**: `(b)` を採ると位相 A の hash バッファ（最大 fileSize）が消え、
§9 の位相表から当該項が除去される（契約上は「0」）。

---

## 8. R2-D6 — Preview 経路の scope 裁定

### 8.1 現状（R1 の実測を維持）

```text
startAsyncIRLoadPreview (ConvolverControlPanel.cpp:1135-1168)
  → setIRPreviewInProgress(true)  (:1143)
  → g_irPreviewThreadPool.addJob(...)  (:1146、単一スレッド pool :16)
  → analyzeImpulseResponseFile (StateAndUI.cpp:455)
  → loadImpulseResponsePreviewFile (ResampleAndFallback.cpp:271-331)
  → resampleIR / applyAsymmetricTukey / estimateEffectiveIRLengthSamples

job lambda に try/catch が無く、JUCE ThreadPool::runNextJob が
catch (...){ jassertfalse; }（juce_ThreadPool.cpp:389-393）で握るだけ。
→ bad_alloc 時は finishAsyncIRLoadPreview に到達せず、
   irPreviewInProgress が true のまま残る（clear は :1176 のみ）。
→ :1437 の updateIRInfo が以後 "Analyzing IR..." を表示し早期 return（恒久的な UI 状態欠陥）。
```

### 8.2 裁定（境界の固定）

本 R2 の制約は **UI = 0** である。preview の graceful failure を閉じる最小の変更は
`ConvolverControlPanel.cpp` の job lambda、または `analyzeImpulseResponseFile`（convolver 側）の
try/catch であり、前者は UI ファイルである。

```text
R2-D6（決定）

  境界を次のとおり固定する:

    WORK102 main contract
        FC-INV-5 = main loader（LoaderThread::performLoad の catch 群）に適用
        preview 経路の admission（FC-FORM-1/2/3/4/5）は main contract と**同一**に適用

    WORK102-PREV-01   Preview allocation failure graceful completion   ← 別 work（追跡 ID 付き）
        OPEN defect として登録。**FAIL は消さない。**
        closure 条件: job 例外時に preview が必ず終端状態（成功 or errorMessage）へ到達し、
                      irPreviewInProgress が確実に false へ戻ること。
        実装位置の候補: analyzeImpulseResponseFile の try/catch（convolver 側ファイル）または
                        job lambda（UI ファイル）。

  ★ 本契約は「whole loader が graceful failure を保証する」とは主張しない。
     主張するのは「main loader が graceful failure を保証し、
     preview 経路の graceful failure は WORK102-PREV-01 として OPEN」である。
```

**この形が R1 の要求（別 work にする ≠ FAIL を消す）を満たす。**

---

## 9. Whole-path capacity の再計算（最終値で）

### 9.1 前提

```text
kMaxIRLoadBytes = 1,073,741,824 B (1 GiB)   （R2-D2 → DERIVED）
kMaxIRLoadChannels = 8                       （R2-D3 → POLICY）
kMaxIRResampleOutputSamples = 2,097,153      （R2-D4 → 凍結済み hardMaxSec に整合）
FC-8 = (b) streaming hash                    （R2-D5 → hash buffer = O(1)）
U = 2,097,152 / kStreamChunk = 262,144 / sizeof(double) = 8
```

### 9.2 individual allocation upper bound（個々の確保の上限）

| 記号 | 実体 | 実測位置 | 上限 |
|---|---|---|---|
| hash buffer | `computeIRHash` の `fileData` | `AllpassDesigner.cpp:626` | **O(1)**（R2-D5 (b) により 0） |
| T1 | `tempFloatBuffer` | `LoaderThread.cpp:414` | 8,388,608 B = 8 MiB |
| T2 | `tempAligned` | `:415` | 2,097,152 B = 2 MiB |
| **C_src** | `loadedIR` | `:422` | **1,073,741,824 B = 1 GiB**（FC-FORM-1） |
| C_res(ch) | `chData[ch]`（`std::vector<double>`） | `ResampleAndFallback.cpp:46,62` | 134,217,792 B = 128 MiB |
| C_res(out) | `result` AudioBuffer | `:93` | 134,217,792 B = 128 MiB |
| Tukey | `window_vals` + `aligned_data` 等 | `:135,191`（1 チャンネル分） | ≤ 2 × L × 8 |
| stepTrimmed | `stepTrimmed` | `LoaderThread.cpp:586` | 134,217,728 B = 128 MiB |
| irL / irR | `makeAlignedArray` × 2 | `:163-164` | 16,777,216 B × 2 = 32 MiB |
| displayIR | `result.displayIR`（可視化 ON 時） | `:175` | 134,217,728 B = 128 MiB |

### 9.3 simultaneous residency（位相ごとの同時滞在）— 重要

**個々の上限の総和は同時滞在ピークではない。** 位相ごとに評価する。

| 位相 | 同時に生存するもの | ピーク |
|---|---|---|
| **A** チャンク読込（`doLoadIRStep` :414-450） | T1 + T2 + C_src | 8 + 2 + 1024 = **1034 MiB** |
| **B** 末尾無音トリム（`doTrimStep` :512-516） | `setSize(..., keepExistingContent=true)` が新規確保 → copy → swap するため **旧 + 新が同居**。続く `shrinkToFit`（`Internal.h:56-60`）も同型 | **2 × C_src = 2048 MiB ← PEAK** |
| **C** リサンプル（:519-553） | C_src(trim 後) + `chData[ch]` + `result` | 1024 + 256 = **1280 MiB** |
| **D** Tukey（:570-581、ch ごと） | C_src(trim 後) + 当該 ch の補助バッファ | ≤ 1024 + 2×L×8 ≤ **~1536 MiB** |
| **E** `stepTrimmed` 確保（:586） | C_src(trim 後) + stepTrimmed | ≤ 1024 + 128 = **1152 MiB** |
| **F** build（`:163-164` 以降） | stepTrimmed + irL + irR (+ displayIR) | ≤ 128 + 32 (+128) = **288 MiB** |

**whole-path simultaneous residency ≤ 2 × kMaxIRLoadBytes = 2,048 MiB（2 GiB）。**

**R1 の 1.45 GiB の誤り**: R1 は (i) 同時に生存しない項を加算し、
(ii) trim 局面の copy-on-resize（位相 B）を計上していなかった。

**規約（契約に明記）**:

```text
FC-INV-2 の表明は 2 段で行う:
  (i)  individual allocation bound : 単一の確保は kMaxIRLoadBytes (1 GiB) を超えない
  (ii) simultaneous residency      : ≤ 2 × kMaxIRLoadBytes (2 GiB)
                                     （JUCE AudioBuffer の copy-on-resize に由来）
この 2 つを混同しない。
```

### 9.4 任意の改善候補（IG の判断事項・契約要求ではない）

```text
位相 B の 2 GiB は JUCE の copy-on-resize に由来する。
`setSize(..., keepExistingContent=true, clearExtraSpace=false, avoidReallocating=true)`
（`juce_AudioSampleBuffer.h:381`）とすると、縮小の場合に remap が不要になり
（同 `:399-402` の分岐）、旧+新の同居が発生しない。
この場合 whole-path ピークは位相 C の 1,280 MiB（または D の ~1.5 GiB）へ下がる。

ただし shrinkToFit（:515）は独立に同サイズの複製を作るため、併せて扱う必要がある。
→ 本 R2 では **契約要求としない**（IG の任意改善候補 IS-7 として記録）。
```

---

## 10. Compatibility matrix（policy 後に再生成）

### 10.1 順序の遵守

```text
手順 1  Policy arbitration  : R2-D1（P1 条件付き採用）/ R2-D2（envelope 44.1k–768k）
                              R2-D3（K=8 + SR-01B supersede）/ R2-D4（FC-FORM-5 採用・値 U+1）
                              R2-D5（FC-8 = (b)）/ R2-D6（preview 分離）
手順 2  contract values     : kMaxIRLoadBytes = 1 GiB / K = 8 / kMaxIRResampleOutputSamples = U+1
手順 3  compatibility matrix: 本節
```

**R1 のように「K=8 を仮定して matrix を proof とする」順序は取らない。**

### 10.2 判定式（最終契約）

```text
newly rejected ⟺
    (1) N > kMaxIRLoadChannels (= 8)                       … FC-FORM-2
  ∨ (2) N × fileLength × 8 > kMaxIRLoadBytes (= 1 GiB)     … FC-FORM-1
  ∨ (3) ceil(L_trim × sr / fileSR) + 1 > U + 1             … FC-FORM-5
  ∨ (4) fileLength > INT32_MAX                             … FC-FORM-3
```

### 10.3 カテゴリ別 matrix

**N 別の実効上限（`FC-FORM-1` 対 `FC-FORM-5`）**

| 行 | N | FC-FORM-1 の L 上限 | どちらが先に効くか | 判定 |
|---|---|---|---|---|
| 1ch | 1 | 134,217,728 | FC-FORM-5 | 受理 |
| 2ch | 2 | 67,108,864 | FC-FORM-5 | 受理 |
| 3ch | 3 | 44,739,242 | FC-FORM-5 | 受理 |
| K = 8ch | 8 | 16,777,216 | FC-FORM-5 | 受理 |
| **K+1 = 9ch** | 9 | — | — | **拒否（CC-1）** |

※ fileSR ≤ sr のとき FC-FORM-5 が、fileSR > sr のとき FC-FORM-1 が binding になる。

**SR × 完全利用ファイルの受理可否**（`sr = 44.1 kHz` が最悪条件。fileSR 別に
`fileLength = (U−1) × fileSR / sr` が受理されるか）

| N | 44.1k | 48k | 96k | 192k | 384k | 768k |
|---|---|---|---|---|---|---|
| 1ch | OK | OK | OK | OK | OK | OK |
| 2ch | OK | OK | OK | OK | OK | OK |
| 3ch | OK | OK | OK | OK | OK | OK |
| 8ch | OK | OK | OK | OK | **NG** | **NG** |

→ **high-SR 行**: N ≤ 3 は全 SR で完全互換。N ≥ 4 は CC-2。
→ **low-SR 行**: FC-FORM-5 により `L ≤ U` が binding。全 N で「DSP が使える長さ」は完全に受理。

**duration > hardMax（FC-FORM-5）の具体例**

| 例 | sr | L_res | FC-FORM-1 | FC-FORM-5 | UI は既に拒否していたか |
|---|---|---|---|---|---|
| 1 s @44.1k | 44.1k | 44,100 | PASS | PASS | いいえ |
| 1 s @44.1k | 768k | 768,001 | PASS | PASS | いいえ |
| 3 s @44.1k | 44.1k | 132,300 | PASS | PASS | いいえ |
| **3 s @44.1k** | **768k** | **2,304,001** | PASS | **REJECT** | **はい** |
| **10 s @44.1k** | **768k** | **7,680,001** | PASS | **REJECT** | **はい** |
| 2 s @48k | 768k | 1,536,001 | PASS | PASS | いいえ |

**経路別の差分**

| 経路 | FC-FORM-1/2/3/4 | FC-FORM-5 | FC-INV-5 |
|---|---|---|---|
| main loader（`LoaderThread`） | **新規** | **新規**（CC-4） | 既存維持 |
| preview（`loadImpulseResponsePreviewFile`） | **新規** | 変更なし（既に `exceedsHardLimit`） | **FAIL（WORK102-PREV-01）** |
| CLI（`MainWindow.cpp:796,838`） | **新規** | **新規**（CC-4） | main 経路に同じ |
| state 復元（`StateAndUI.cpp:417`） | **新規** | **新規**（CC-4） | main 経路に同じ |

### 10.4 compatibility change 一覧（最終）

| ID | 内容 | 性質 | 実害評価 |
|---|---|---|---|
| **CC-1** | `N > 8` を新規拒否 | **SR-01B 凍結境界条件の supersede**（R2-D3） | 追跡 ID なしでは proof にしない。supersede として明示承認する |
| **CC-2** | `N ≥ 4` かつ高 SR（384k/768k）の完全利用ファイルを新規拒否 | 新規拒否 | mono/stereo/3ch は全 SR で影響なし |
| **CC-3** | mono / stereo / 3ch は 44.1k–768k で完全互換 | 互換（変更なし） | — |
| **CC-4** | `duration > hardMaxSec(sr)` を loader で新規拒否（FC-FORM-5） | 新規拒否（interactive は変更なし） | **CLI と state 復元のみ**。705.6k/768k では 3 秒 IR が該当 |
| **CC-5** | FC-8 (b) により hash の確保が消える | 挙動等価（§7.3） | 変更なし |

---

## 11. 最終契約（凍結）

### 11.1 契約形

```text
FC-FORM-1  source loadedIR byte bound
    numChannels × fileLength × sizeof(double) ≤ kMaxIRLoadBytes
FC-FORM-2  channel bound
    numChannels ≤ kMaxIRLoadChannels
FC-FORM-3  INT32 representation bound
    fileLength ≤ MAX_FILE_LENGTH = 2,147,483,647        （既存維持）
FC-FORM-4  degenerate input bound
    numChannels ≥ 1  ∧  fileLength ≥ 1                  （既存維持）
FC-FORM-5  resample output bound
    (double)L_trim / fileSR  ≤  (double)kMaxIRResampleOutputSamples / processingSampleRate
    L_trim = 末尾無音トリム後の loadedIR 長。fileSR ≤ 0 のとき vacuous。
FC-FORM-6  hash allocation bound
    computeIRHash は物理ファイルサイズ比例の allocation を行わない（O(1) メモリ）
```

### 11.2 契約値

```text
kMaxIRLoadBytes             = 1,073,741,824 B (1 GiB)   ← R2-D2 からの DERIVED REQUIREMENT
kMaxIRLoadChannels          = 8                          ← R2-D3 POLICY VALUE
kMaxIRResampleOutputSamples = 2,097,153 (U + 1)          ← R2-D4 / SR-01B hardMaxSec に整合
MAX_FILE_LENGTH             = 2,147,483,647              ← 既存維持（representation precondition）
IR file SR envelope         = 44.1 kHz – 768 kHz         ← R2-D2 新規契約事項
```

### 11.3 不変条件

```text
FC-INV-1  [Contract] admission-before-allocation
    O(fileLength) および O(file bytes) の全確保は、対応する admission の後にのみ発生する。
    ★ 現行は computeIRHash(:364) で FAIL（R2-D5 で閉塞）。
FC-INV-2  [Contract] whole loader bounded allocation
    (i)  individual allocation ≤ kMaxIRLoadBytes          (1 GiB)
    (ii) simultaneous residency ≤ 2 × kMaxIRLoadBytes     (2 GiB)
    ※ (i) と (ii) を混同しない。上限は loadedIR だけでなくロード経路全体に適用する。
FC-INV-3  [Impl] MAX_IR_LATENCY ≠ loader allocation bound
    MAX_IR_LATENCY = 2,097,152 は DSP usable length の上限。
    FC-FORM-5 は **output length（utility）bound** であり allocation bound ではない。
    FC-FORM-5 から導出される allocation（N × (U+1) × 8 ≤ 128 MiB/ch）は
    FC-FORM-1 とは独立の量である。
    禁止: kMaxIRLoadBytes を MAX_IR_LATENCY × N × sizeof(double) として定義すること。
FC-INV-4  [Contract] deterministic rejection
    同じ (N, fileLength, L_trim, fileSR, sr) に対し常に同じ判定。音声内容・圧縮形式に依存しない。
FC-INV-5  [Contract] graceful failure
    main loader: 既存の performLoad の catch 群を維持（std::terminate しない）。
    preview: **WORK102-PREV-01 として OPEN**（本契約は preview の graceful failure を主張しない）。
FC-INV-6  [Impl] chunked read retained
    kStreamChunk = 262,144 のチャンク読込（:404-450）を維持する。
FC-INV-7  [Impl] no streaming / incremental partition redesign
    IncrementalRebuildJob（Rebuild.cpp:177）は dormant のまま維持する。
FC-INV-8  [Impl] diagnostic single source
    FC-FORM-1/2/3/5 のガードと文言は 1 箇所で定義し、LoaderThread と
    loadImpulseResponsePreviewFile の 2 経路はそこを参照する。
    dimension + actual + limit を開示する。UI 表示形式は変更しない。
FC-INV-9  [Impl] admission ordering
    1. FC-FORM-4   2. FC-FORM-3   3. FC-FORM-2   4. FC-FORM-1
    5. ★ ここより後に O(file bytes) / O(fileLength) の確保を開始する（FC-FORM-6）
    6. loadedIR 確保 / chunk read
    7. 末尾無音トリム（L_trim 確定）
    8. ★ FC-FORM-5 を L_trim に対して評価
    9. リサンプラ構築 / r8b getMaxOutLen / (int) 変換
FC-INV-10 [Impl] overflow-safe pre-resampler arithmetic
    FC-FORM-5 の比較は double で行い、int 変換や resampler 構築より前に評価する。
    根拠: r8b getMaxOutLen の (int) ceil(MaxInLen × DstSR / SrcSR) + 1 は
          N=1 極長で 2,337,397,169 > INT32_MAX となり溢れる。
```

### 11.4 禁止事項（波及させない範囲）

```text
H-01 / H-02 / M-01 / M-02 / M-03 / SR-01（hardMaxSec の定義を除く参照のみ）/
SR-01B の channel count 行（R2-D3 の supersede の対象である。それ以外の行は不変）/
SR-02 / SR-03 / Publish / Crossfade / Retire / Epoch / RuntimeWorld /
Lifetime Budget / RT path / DELAY_BUFFER_SIZE / MAX_IR_LATENCY / NUC スレッド契約 /
UI 表示形式 / README E-G3-1 差分 / ConvoPeq.md 再生成
```

---

## 12. Implementation Gate の前提条件

```text
IG-1 [countersign] R2-D2 の envelope「IR file SR = 44.1 kHz – 768 kHz」を
      新規契約事項として承認する（P3-C の採用）。
IG-2 [countersign] R2-D3 の SR-01B supersede（channel count 行の改訂）を承認する。
IG-3 [countersign] CC-1 / CC-2 / CC-4 の compatibility change を承認する。
      ※ CC-4 は CLI / state 復元 の 2 経路のみに作用する（interactive は変更なし）。
IG-4 [必須] WORK102-PREV-01 を起票する（preview の graceful failure）。
      本契約は preview の graceful failure を主張しない。FAIL は OPEN として維持する。
IG-5 [任意] IS-7: trim 局面の copy-on-resize を avoidReallocating で回避し、
      whole-path ピークを 2 GiB → ~1.5 GiB へ下げる（契約要求ではない）。
IG-6 [必須] 診断文言は §11 FC-INV-8 に従い単一情報源とする。
      既存 "IR too large (Out of Memory)"（:130）は allocation 失敗に対応する別事象であり変更しない。
```

**IG-1〜IG-3 は「技術的な未決」ではなく「既凍結契約・対外互換の承認」である。**
R1 の判定基準（「1 項目でも product-policy の裁定が必要なら IG へ進まない」）に対する回答は、
**裁定自体は本 R2 で完了しており、残るのは countersign という governance 手続きのみ**である。

---

## 13. Final classification

```text
最終判定: CONTRACT-FROZEN
```

### 13.1 ユーザー提示の CONTRACT-FROZEN チェックリスト

| # | 項目 | 状態 | 裁定 |
|---|---|---|---|
| 1 | P1 | **裁定済** | R2-D1: 無条件採用せず、R2-D2 の envelope 内での条件付き requirement として採用 |
| 2 | file-SR envelope | **裁定済** | R2-D2: 44.1 kHz – 768 kHz（新規契約事項として明示） |
| 3 | kMaxIRLoadBytes | **裁定済** | 1 GiB（R2-D2 からの DERIVED REQUIREMENT） |
| 4 | kMaxIRLoadChannels | **裁定済** | 8（policy value と明記） |
| 5 | SR-01B supersede | **裁定済** | R2-D3: N ≤ 8 で凍結挙動維持、N > 8 のみ改訂。改訂文面を §5.3 に固定 |
| 6 | FC-FORM-5 | **裁定済** | R2-D4: 採用。適用点は trim 後・resampler 構築前 |
| 7 | FC-FORM-5 value | **裁定済** | U + 1 = 2,097,153（SR-01B hardMaxSec に整合、新規数値の発明なし） |
| 8 | FC-8 strategy | **裁定済** | R2-D5: (b) streaming hash（(d) は確保を消さないため不採用） |
| 9 | preview scope | **裁定済** | R2-D6: WORK102-PREV-01 として分離。FAIL は OPEN として維持 |
| 10 | compatibility changes | **裁定済** | §10.4 の CC-1〜CC-5（policy 後に再生成） |
| 11 | whole-path capacity | **裁定済** | §9: individual ≤ 1 GiB / simultaneous ≤ 2 GiB（R1 の 1.45 GiB を訂正） |
| 12 | overflow-safe arithmetic | **裁定済** | FC-INV-10（double 比較、resampler 構築前） |

**全 12 項目が明示的に裁定された。** 3 層（PROVEN FACT / DERIVED REQUIREMENT / POLICY CHOICE）は §11 と
各裁定節で維持されている。

### 13.2 他判定の棄却

- **CONTRACT-HOLD**: 残るのは countersign（governance）のみで、技術的な未決は 0。
  R1 が残した「P1 が無ラベルの前提」は R2-D2 の明示宣言で解消した。→ 棄却。
- **DESIGN-DEFERRED**: R-1（リサンプル中間 37.40 GB）は FC-FORM-5 の admission 追加で閉じる。
  これは admission チェックであり、ストリーミング化・順序再設計を要しない。→ 棄却。
- **NO-GO**: whole-path は 2 GiB で有界。安全性は保証される。→ 棄却。

### 13.3 次工程

```text
CONTRACT-FROZEN のため、次工程は WORK102-IG — Implementation Gate。
ただし IG の着手は IG-1〜IG-3 の countersign と IG-4 の WORK102-PREV-01 起票を前提とする。
本 R2 はここで停止する（実装は行わない）。
```

---

## 14. 監査メタ

```text
検証方法    : ConvoPeq.md（一次ソース）+ src の行単位実測
              LoaderThread.cpp / ResampleAndFallback.cpp / AllpassDesigner.cpp /
              ConvolverProcessor.Internal.h / ConvolverControlPanel.cpp / MainWindow.cpp /
              StateAndUI.cpp / juce_AudioSampleBuffer.h / juce_HeapBlock.h /
              juce_ThreadPool.cpp / r8brain-free-src/CDSPFracInterpolator.h
              + 容量・位相ピークの数値再計算 + 凍結文書（SR-01B / work101）との cross-check

Production 変更     : 0
CMake 変更          : 0
UI 変更             : 0
tests 変更          : 0
ConvoPeq.md 再生成  : 0（FRESH 維持）
commit / push       : 0
README E-G3-1 差分  : 維持
```

### R1 文書からの主要変更（要約）

| # | R1 の記述 | R2 の裁定 |
|---|---|---|
| 1 | P1 は無ラベルの policy 前提 | **R2-D1**: envelope 内の条件付き requirement へ再定義（無条件採用をやめる） |
| 2 | P3 は policy（未裁定） | **R2-D2**: 44.1 kHz – 768 kHz を新規契約事項として明示裁定 |
| 3 | K は Option A/B/C の比較提示 | **R2-D3**: Option B / K=8 / SR-01B supersede 文面を固定 |
| 4 | FC-FORM-5 は raw fileLength に適用 | **R2-D4**: trim 後・resampler 構築前に変更（無音パディング回帰を回避） |
| 5 | FC-FORM-5 の値は未裁定 | **R2-D4**: U + 1 = 2,097,153 |
| 6 | FC-8 は (b) or (d) | **R2-D5**: (b) を採用。(d) は確保を消さないため不採用 |
| 7 | preview は分離候補 | **R2-D6**: WORK102-PREV-01 として分離、FAIL は OPEN 維持 |
| 8 | whole-path ≒ 1.45 GiB | **§9**: individual ≤ 1 GiB / simultaneous ≤ 2 GiB（trim の copy-on-resize を計上） |
| 9 | 判定 DESIGN-DEFERRED | **CONTRACT-FROZEN**（全 12 項目裁定済み） |
