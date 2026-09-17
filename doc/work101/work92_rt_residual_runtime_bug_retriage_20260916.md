# WORK92-RT — Residual Runtime Bug Re-Triage（read-only）

- **作成日**: 2026-09-16
- **種別**: read-only 監査（production 変更 0 / CMake 変更 0 / UI 変更 0 / commit 0）
- **目的**: work92 の残存 Bug List と現行ソースを突き合わせ、**「古い記述が残っている」ことと「現行コードに実欠陥がある」ことを分離**して、次に実装すべき項目を決めるための証拠を作る。

## 0. Baseline

```text
HEAD        = 35461e2e3a5fd42fd009416a20adaf457b9b4031
origin/main = 35461e2e3a5fd42fd009416a20adaf457b9b4031   (ahead/behind = 0/0)
M-03        = CLOSED

Source（authoritative）:
  ConvoPeq.md  Generated = 2026-09-16 22:18:13
  --check      NEWER_SRC_COUNT = 0 / STATUS = FRESH

Production changes : 0
CMake changes      : 0
UI changes         : 0
commit             : 0
```

> 監査に提示された `ConvoPeq.md` は Generated 2026-09-16 20:13:19 だが、commit 済みの
> authoritative snapshot は **22:18:13（FRESH）** である（M-03 Final Gate §13.7 の取り決め）。
> 本監査は 22:18:13 を基準とし、production `src/` については両者に差がない。

## 1. 対象と方法

対象:

- `doc/work92/CORRECT_INTEGRATED_BUG_LIST.md` §1「重大度別 整理（バグとして残存するもの）」= 18 項目
- 後続の再トリアージ記録（`.auto/rb/next_work_triage_20260913.md` / `.auto/rb/rb03_ledger_closure_gate.md`）
- 現行 Runtime / Publication / Retire / Shutdown 実装（Phase 2）

方法: 各項目の「実測根拠」に記載された `file:line` を現行ソースで直接照合し、
**(i) 旧対策が実装済みか (ii) 契約で閉じているか (iii) 実欠陥として残るか** を判定した。
`★ work92` マーカーコメントの有無を実装完了の一次根拠として使用した。

## 2. Part A — work92 §1（18 項目）の現行照合

### A-1. 🔴 P0（3 項目）→ **3/3 CLOSED**

| # | 旧 Bug | 旧記載の根拠 | 現行ソース実測 | 判定 |
|---|---|---|---|---|
| 1 | **big 1-1** nucHCMode/nucLCMode が ValueTree 永続化から欠落 | `StateAndUI.cpp` getState :202-250 / setState :289- | **対策実装済み**。`StateAndUI.cpp:260-261` に `v.setProperty("nucHCMode", nucHc)` / `("nucLCMode", nucLc)`、`:391`/`:394` に `v.getProperty("nucHCMode"/"nucLCMode")` の復元。ファイル内の出現は 23 箇所 | **CLOSED** |
| 2 | **big 1-3** `_mm256_store_pd`（アライン要求）が非契約 `dst` へ | `InputBitDepthTransform.h:114-115` / `MKLNonUniformConvolver.cpp:1407,1668` | **3 サイトとも解消**。①`InputBitDepthTransform.h:116-117` は `_mm256_storeu_pd` 化。②`MKLNonUniformConvolver.cpp:1768-1773` は `aligned`（dst∧src の 31bit マスク）で `store_pd`/`storeu` を実行時分岐。③`MKLNonUniformConvolver.cpp:1494-1496` は `load_pd`/`store_pd` 対だが、`accumBuf = mkl_malloc(partStride * sizeof(double), 64)`（64B アライン）かつ `partStride = (complexSize*2 + 7) & ~7`（**8 の倍数**）のため `k += 4` ループは整列・範囲とも安全 | **CLOSED** |
| 3 | **big 1-6 (= BUG-034)** IPP FFT 戻り値無視 | `MklFftEvaluator.h:270-271, 425-426` | **対策実装済み**（`★ work92 A-3`）。`:280-292` で `const IppStatus stL/stR` を捕捉し、`!= ippStsNoErr` で L/R を `memset` ゼロクリア＋`Result{fftFailed=true}` を返す。`:449-458`（computeFft）も同一 semantics ＋ `ippFailureCount_`（`:465`）で観測。**部分成功を保存しない**規定もコメントで固定 | **CLOSED** |

### A-2. 🟠 P1（8 項目 − 解決済 1 = 7 項目）→ **7/7 CLOSED（うち 1 件は契約閉じ）**

| # | 旧 Bug | 現行ソース実測 | 判定 |
|---|---|---|---|
| 4 | **big 1-7** `emitRetireIntentRT` が輻輳時 mutex | **`emitRetireIntentNonRT` に改名済み**（`ISRRetire.cpp:93`、`★ Finding 9 + work92 B-1 (big 1-7 リネーム)`）。`:95-101` に「『RT』は RealTime thread safety を意味しない／呼び出し元は全て非 RT（`AudioEngine.Commit.cpp:485`）／将来 Audio Thread から呼ぶ場合は mutex 非使用の別実装を用意すること」を明記 | **CLOSED（契約）** ※下記 §5-1 の残置あり |
| 5 | **big 1-8** LoaderThread が `MAX_FILE_LENGTH = INT32_MAX` まで一括確保 | ガードは実在（`LoaderThread.cpp:391-397`: `fileLength > MAX_FILE_LENGTH` → errorMessage + return false）。ただし**ストリーミング化は未実装**で、`:422` は `loadedIR.setSize(numChannels, fileLength)` で**ファイル長比例に確保**する（2ch×8B×2^31 ≒ 34GB が上限内で依然可能） | **CONTRACT DEFERRED**（§5-2） |
| 6 | **big 1-9** `fftSize * sizeof(double)` の int 溢れ | **対策実装済み**（`★ work92 B-6`）。`MKLNonUniformConvolver.h:355` `std::int64_t fftSize = 0;`（`:351-354` に int64 化理由を明記） | **CLOSED** |
| 7 | **big 1-10** `m_pendingIRChange` の公開前クリア | **acknowledgement protocol として契約化**（`★ work92 B-2`）。`AudioEngine.Snapshot.cpp:95-102` に「本 `exchange(false)` が唯一の acknowledge 点／flag は level marker／writer（`Timer.cpp:800`, `UIEvents.cpp:177` の 2 箇所）は必ず直前で `submitRebuildIntent(Structural)` を先行発行（G1 CFG 確認済）／**この契約を壊す変更は禁止**」を明記 | **CLOSED（契約）** |
| 8 | **big 2-6** `cachedThreadHash` 衝突による reader 二重登録 | **対策実装済み**（`★ work92 B-4`）。`ThreadHash.h:33-43` に `acquireUniqueThreadId()`（`static std::atomic<uint64_t> s_counter` を `fetch_add(relaxed) + 1` で 0 起点予約・プロセス単調）。`RCUReader.h:150-157` の `currentThreadToken()` が本関数へ統一。役割タグ（`DspNumericPolicy.h` の `isAudioThread` 等）は衝突許容のため `cachedThreadHash()` を維持（RECONCILIATION §5-3 設計確定） | **CLOSED** |
| 9 | **big 2-9** タイミング計算の uint64 減算 underflow | **対策実装済み**（`★ work92 B-8`）。`AudioEngine.Processing.AudioBlock.cpp:633` `convo::saturatingSubUs(nowUs, cbStartUs)` / `:639` `saturatingSubUs(cbStartUs, cbPrevEndUs)`（クロック逆転時 wrap を saturate 0 に置換）。`:626` に `cbStartUs != kNeverStartedUs` ガード | **CLOSED** |
| 10 | **big 2-10** `NoiseShaperType` enum cast に範囲チェックなし | **対策実装済み**（`★ work92 B-3`）。`AudioEngine.StateIO.cpp:93-98` が `Psychoacoustic(0) <= raw <= Fixed15Tap(3)` を検査してからのみ `setNoiseShaperType`。`:128` に「NoiseShaperType と同型の無検証キャスト」への拡張記録あり | **CLOSED** |
| 11 | ~~BUG-065 残存~~ | 旧記載どおり work92 B-7a（2026-09-10）で解消済み。本監査でも変更なし | **CLOSED**（既報） |

### A-3. 🟡 P2（7 項目）→ **7/7 CLOSED**

| # | 旧 Bug | 現行ソース実測 | 判定 |
|---|---|---|---|
| 12 | **big 2-1** 非 ASCII 識別子 `SoftClipPadéPolicy` | **ASCII 化済み**。`src/dsp/math/FastTanhApprox.h:63` `struct SoftClipPadePolicy`、使用側も `DSPCoreDouble.cpp:127` / `:191` とも `SoftClipPadePolicy` | **CLOSED** |
| 13 | **big 2-2** 入力 DC ブロッカー後の NaN/Inf スクラブ欠如 | **対策実装済み**（`★ work92 C-3`）。`AudioEngine.Processing.DSPCoreIO.cpp:253-257`（`processInput`）と `:310-312`（`processInputDouble`）で `dc.inputL/R.process(...)` の**直後**に `sanitizeFiniteChunk` を追加。入力前スクラブ（`:231-232`/`:288-289`）と合わせ計 8 呼び出し | **CLOSED** |
| 14 | **big 2-3** テストの矛盾条件 | **対策実装済み**（`★ work92 C-2`）。`src/tests/EQProcessorMaxGainTests.cpp:353-379` に旧条件の説明と、`kEpsilon` 直下（切り捨て）／直上（加算）を**明示的に分けた**検証を実装。`logBoundTruncated == 0.0` と `logBoundJustAbove > 0.0` を assert | **CLOSED** |
| 15 | **big 2-4** CacheManager strict-aliasing 違反 | **対策実装済み**（`★ work92 C-4`）。`src/CacheManager.cpp:278-288` で `reinterpret_cast<const double*>` の逆参照を廃し、`uint8_t*` カーソル + `std::memcpy`（行単位）へ置換 | **CLOSED** |
| 16 | **big 2-5** `LockFreeRingBuffer::size()` の read 順序未固定 | **対策実装済み**（`★ work92 C-5`）。`src/LockFreeRingBuffer.h:76-84` で `size()` が **writeIndex を先に** acquire 読取（コメントに順序固定の理由を明記） | **CLOSED** |
| 17 | **big 3-6** SnapshotFactory の NaN 等価誤判定 | **対策実装済み**。`src/core/SnapshotFactory.cpp:62-65` に `std::isnan(...)` を sampleRate / inputHeadroomGain / outputMakeupGain / convInputTrimGain の 4 系統で明示 | **CLOSED** |
| 18 | **big 3-9/3-10 + R-9** `/fp:fast` と `/QxCORE-AVX2` の全ターゲット適用 | **対策実装済み**（`★ work92 C-8`）。`CMakeLists.txt:1519` で MSVC の `/fp:fast` を**除去**（既定 `/fp:precise` へ復帰）。`:1632-1637` で `/QxCORE-AVX2` は `$<CONFIG:Release>` ∧ `$<COMPILE_LANGUAGE:CXX>` に config-gate。icx `/fp:fast`（`:1617-1618`）は LLVM OOM 回避の意図的選択として `:1590-1593`/`:1612-1613` に文書化済み | **CLOSED** |

### A-4. Part A 集計

```text
work92 §1 の 18 項目:
  CLOSED            17
  CLOSED（契約）     2  … big 1-7, big 1-10（いずれも CLOSED に含む）
  CONTRACT DEFERRED  1  … big 1-8（ガード有・ストリーミング未実装）
  HOLD / SEPARATE    0
  OPEN               0
```

> **重要**: 旧リストの「🔴 P0 3 件」はいずれも現行ソースで**対策が実装済み**である。
> 旧記述の行番号（`StateAndUI.cpp:202-250` / `InputBitDepthTransform.h:114-115` /
> `MklFftEvaluator.h:270-271`）は現行と不一致であり、**旧リストの記述をそのまま OPEN として扱うのは誤り**。

## 3. Part B — Phase 2: Runtime 構造不変条件の現行確認

`Practical Stable ISR Bridge Runtime` の lifetime / publication / retire / shutdown 系。
M-01〜M-04 が DSP 数値・RT containment 側だったのに対し、こちらは構造的不変条件に直接関係する。

### B-1. INV-ISR-01〜07（最上位不変条件）

**すべて現行コードにコメントとして固定されている**（`src/audioengine/ISRRuntimePublicationCoordinator.h:100-116`）。

```text
INV-ISR-01  isFullyDrained == true の意味（:102）
INV-ISR-02  pendingIntentCount_ は queue size ではなく transport residency + producer（:106、:973 で再掲）
INV-ISR-03  異なる semantic state を一つの counter で表現しない（Intent / DSP ...）（:108）
INV-ISR-04  ShutdownQuiescent reclaim は readerRegistrationClosed なしには絶対許可しない（:110）
INV-ISR-05  completion watermark ≠ publication committed（§6.2 X2 と整合）（:112）
INV-ISR-06  退役・ownership の identity source は publish() の oldWorld / Lifetime（:113）
INV-ISR-07  RuntimeStore::current の publication identity と publish transaction の整合（:115）
```

INV-ISR-04 / 05 は実装側にも波及している:

| 不変条件 | 実装側の裏付け |
|---|---|
| INV-ISR-04 | `EpochDomain.h:598` reader registration permanently closed フラグ／`AudioEngine.CtorDtor.cpp:233`・`AudioEngine.Processing.ReleaseResources.cpp:269` の CloseReaderRegistration／テスト `invariant_INV3_INV5.cpp:195, 444` |
| INV-ISR-05 | `AudioEngine.h:2350`（Committed state 更新点）／`:3817`（committed ≠ completed の明示） |

### B-2. Phase 2 確認対象 10 領域

| # | 領域 | 現行ソース実測 | 状態 |
|---|---|---|---|
| 1 | **Publish authority** | `ISRRuntimePublicationCoordinator.h:65` `enum class PublishAuthority : uint8_t { Granted = 1 };`（単一 authority） | 実在 |
| 2 | **Crossfade authority** | `CrossfadeAuthority.h:23` `class CrossfadeAuthority`／`:7` `CrossfadePolicy: immutable POD（メソッド・状態を持たない）`／`ISRDSPHandle.h:238` `class CrossfadeAuthorityRuntime` | 実在 |
| 3 | **Retire authority** | `ISRRuntimePublicationCoordinator.h:66` `enum class RetireAuthority`／`ISRRetireRouter.h:62` `class TerminalReclaimAuthority`／`RuntimeWorldAuthority.h:143` | 実在 |
| 4 | **Retire transport overflow** | OverflowRing → `emitRetireIntent` → MPSC queue → `processIntent` → `reclaim`（`AudioEngine.h:1576`）。`AudioEngine.Commit.cpp:493` が `pendingRetireGenerationCount_` を publish。`overflowCount` telemetry（`:699`, `CtorDtor.cpp:63`） | 実在 |
| 5 | **Quarantine / EmergencyQuarantine / Terminal** | `QuarantineReason::RetireDeferralTimeout`（`Commit.cpp:605, 625`）／`RetireLane::Quarantine`（`:647`）／`dspQuarantineManager_.reclaimSlot`（`:661`）／`quarantineResident`（`AudioEngine.h:1635`）／`terminalResident`・`terminalPeakResident`（`:1655-1656`）／`emergencyQuarantineResident`（`:1663`） | 実在 |
| 6 | **Shutdown full-drain** | `CtorDtor.cpp:277-299` — D + Q + E + **Terminal** の完全 drain。`drainAll()`（epoch-gated）＋ `drainAllQuarantineStore()`（Q+E+T, epoch 非依存, Audio Thread 停止後のみ = `RetireQuarantineStore.h:59` 契約）。`activeReaderCount()==0` 不成立時は stuck-reader fallback（`:289`） | 実在 |
| 7 | **Lifetime Budget** | `RuntimePolicyEngine.h:220` `struct RecoveryBudget`／`:231` `kBudgetWindowUs = 10 * 60 * 1'000'000`（10 分）／`RuntimePolicyEngine.cpp:247, 260` で窓判定 | 実在 |
| 8 | **Reservation / residency accounting** | `AudioEngine.h:1681` `enum class ResidencyAuthority`／`:1721-1740` で `quarantineResident_` + `retireQuarantineResident` を統合（`★ BUG-015/027 (work88)`）／`:1760-1766` で terminal / emergency を集約 | 実在 |
| 9 | **Reader registration / epoch** | `EpochDomain.h:598` `readerRegistrationClosed()`／`RCUReader.h:154-157` `currentThreadToken() = acquireUniqueThreadId()`／`reserveReaderThread` / `releaseReaderThread`（`RCUReader.h:169` 系）／`activeReaderCount`（`AudioEngine.h:1660`, `CtorDtor.cpp:244, 285`） | 実在 |
| 10 | **RuntimeWorld immutability** | `FrozenRuntimeWorld.h:37` `class FrozenRuntimeWorld`／`RuntimeWorldAuthority.h:143`／`Commit.cpp:275` `ref.mutability = 1u; // immutable payload`／`:285-311` `PayloadTier::InlineImmutable` / `ImmutableShared` | 実在 |

### B-3. Runtime telemetry 観測点

`activeReaderCount` / `pendingRetireCount` / `quarantineResident` / `terminalResident` /
`terminalPeakResident` / `emergencyQuarantineResident` / `readerRegistrationClosed` /
`activeReadersZero` が `AudioEngine.h:1660-1663`, `:1730`, `:4524-4526` に実在。
`ShutdownAuthority`（`:67`）も定義済み。

## 4. 最終分類（4 分類）

```text
A. CLOSED                      17 項目（work92 §1 の 18 のうち）
   - P0 3/3（big 1-1 / 1-3 / 1-6）
   - P1 7/7（big 1-7 契約閉じ・1-9・1-10 契約閉じ・2-6・2-9・2-10 ＋ BUG-065 既報）
   - P2 7/7（big 2-1・2-2・2-3・2-4・2-5・3-6・3-9/3-10）

B. CONTRACT / DESIGN DEFERRED   1 項目
   - big 1-8（LoaderThread 大容量 IR の一括確保）
     * 現状: 2^31 サンプル超はガードで拒否（errorMessage）。よって旧主張の
       「MAX_FILE_LENGTH まで無制限に確保」は成立しない。
     * 残置: ガード内（≦2^31 サンプル）では loadedIR をファイル長比例で確保するため
       大容量 IR で依然として巨大確保が起こりうる。ストリーミング化は未実装。
     * 実害の見込み: IR は通常 ms〜秒オーダーであり、実運用での到達可能性は低い。
     * 閉じ方の選択肢: (a) 最大 IR サンプル数の契約値化（現状受容＋上限明文化）
                       (b) ストリーミング読み込み（設計変更・別 work）

C. HOLD / SEPARATE CANDIDATE    0 項目（work92 §1 内）
   ※ M-03 D3（OFF 側 PDC 過大申告）は本再トリアージの対象外であり、
     HOLD / SEPARATE CANDIDATE のまま維持する（変更なし）。

D. 現行コードで再確認が必要な OPEN  0 項目

Phase 2（Runtime 10 領域 / INV-ISR-01〜07）: すべて現行ソースに実在。欠落・未接続なし。
```

## 5. 残置（CLOSED だが記録すべき付随事項）

### 5-1. big 1-7 の自動 enforce 不在（実欠陥ではない）

`emitRetireIntentNonRT` は**命名と契約コメント**で RT 誤用を抑止しているが、
`ISRRetire.cpp:101` に明記のとおり `jassert(!isAudioThread())` は JUCE ヘッダ未インクルードのため
**使用不可**＝自動 enforce は存在しない。将来 Audio Thread から呼ばれても実行時に検出されない。

- 現状の実害: 呼び出し元は `AudioEngine.Commit.cpp:485` の 1 系統のみで非 RT（確認済み）→ **実欠陥なし**
- 位置付け: **hardening 候補**（OPEN でも CONTRACT DEFERRED でもない）。起票はしない。

### 5-2. big 1-8 の残置

§4-B のとおり。**実装必要性は「上限の明文化」であり「ストリーミング実装」ではない**可能性が高い。
判断材料として、現行の `computeTargetIRLength` が `MAX_IR_LATENCY`（2097152 サンプル）で
**使用長**を上限化している点を付記する（＝使用側は既に有界。無界なのは読み込みバッファ側）。

## 6. 次アクション判断用の整理

```text
work92 由来で「今すぐ実装すべき OPEN」は 0 件。
→ M-03 の残件（D3）を実装に戻す理由もない（HOLD / SEPARATE を維持）。

次に実装候補となりうるもの（本監査の範囲では未着手のまま）:
  1. big 1-8 の (a) 上限明文化  … 文書＋小改修。ただし OPEN ではなく DEFERRED の解消。
  2. big 1-7 の自動 enforce     … hardening。OPEN ではない。
  ※ いずれも「次に何を実装するか」の判断材料であって、本 work では実装しない。

別トラック（work92 外・既に管理下）:
  - WI-E-G3-1（icx Debug AVX 不整合）        NO-GO（README 明文化済み）
  - WI-E-G3-3（build.bat × identity gate）    NO-GO（pre-audit CLOSED・ユーザー判断待ち）
  - work68 v7.4 → v7.5 反映                   条件付き GO
  - M-03 D3（OFF 側 PDC 過大申告）            HOLD / SEPARATE CANDIDATE 維持
```

## 7. 監査メタ

```text
検証した項目           : work92 §1 の 18 項目 + Phase 2 の 10 領域 + INV-ISR-01〜07
一次根拠               : 現行ソースの行単位実測（`★ work92` マーカーコメント照合を含む）
Production 変更        : 0
CMake 変更             : 0
UI 変更                : 0
ConvoPeq.md 再生成     : 0
commit / push          : 0
禁止事項の遵守         : D3 実装なし / M-03 latency 修正なし / HC/LC direct-head 変更なし /
                         PDC semantics 変更なし / Publish・Crossfade・Retire authority 追加なし /
                         RT lock・allocation・block・decision 追加なし
```

**本監査は実装ではなく「次に何を実装するかを決めるための証拠作り」である。**

## 8. UPDATE 2026-09-17 — big 1-8 の CONTRACT DEFERRED 解消（supersede）

> 本節は追記であり、上記の監査記録（§1〜§7・2026-09-16 時点・HEAD 35461e2e 実測）は一切変更していない。

- **§4-B row 5（big 1-8 CONTRACT DEFERRED）、§5-2、§6-1 の「big 1-8 の (a) 上限明文化」は解消済み。**
- 解消チェーン: **WORK102 `859718e4`**（FC-FORM-1/2/3/4/5 admission — 1 GiB destination bound + `IRLoadAdmission.h` 単一情報源 + streaming hash＝「上限の明文化」の実施）+ **WORK102-PREV-01 `eab40c23`**（preview 経路の同一契約 parity）。
- したがって big 1-8 = **RESOLVED**。CONTRACT DEFERRED のまま残るものは存在しない。
- 同時に §5-1（big 1-7 自動 enforce 不在）は **HARDENING ONLY** として維持（W105 実測: debug assertion 1-2 行で満たせる候補・OPEN / CONTRACT DEFERRED には戻さない）。
- 正式状態表と全証跡: **`doc/work92/WORK92_LEDGER_RECONCILIATION_20260917.md`**（WORK104/105/106 監査連鎖・ConvoPeq 13:10:15 FRESH 基準）。
