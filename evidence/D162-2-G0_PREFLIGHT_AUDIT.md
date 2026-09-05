# D162-2-G0 — S3/V-D Staged Re-enable Preflight Audit（read-only）

- Work item: D162-2-G0（D162-2-G 開始前の read-only preflight。production source 変更 **0**）
- Date: 2026-09-04
- 基準ソース: **ConvoPeq.md `Generated: 2026-09-04 18:29:10`**（D162-2-F 修正込み・最新 snapshot）
  - 本監査の全行番号（`md:`）はこのファイルの行番号。実ファイル行番号は別途注意。
- 判定: **PASS（G1 実施可）** — ただし G2 に 1 件の設計分岐（V-D-a/V-D-b）と、
  G1 の観測可能性条件（S3 発火条件）を残課題として明示（§6, §7）
- 上位文書: D162-2-F PASS（doc/work88/D162-2-F_ALLOCATOR_CONTRACT_REPAIR_REPORT.md）、
  D162-2-C INV-D162-6..9（evidence/D162-2C_SHUTDOWN_TEARDOWN_AUDIT.md）

---

## 0. Executive Summary

1. **S3 の switch は「 retire 呼び出しの有無」**（コンパイルスイッチ/ランタイムフラグ不在）。
   `clearDeferredForShutdown()`（RuntimePublicationOrchestrator.cpp）内の disposition block が
   注記付きで空のまま。G1 は E-4d（`drainDeferredClearIfRequested`）と同一パターンの
   `retireRegisteredDSP(..., "shutdown-clear")` 挿入が最小変更。
2. **V-D の switch は `if (false && ...)` x 2 行**（ReleaseResources.cpp VerifyDrained・
   md:38372/:38381）。コード形状は B-era direct destroy（T2）のまま休眠。
3. **重要な新規所見（G0-2）**: V-D direct destroy を有効化した場合、
   `runtimeDSPHandleMap_` のエントリ（dsp→handle）が **erase されずに残存**し、
   registry slot は Reclaimed→freelist 再発行可能になる。E-2 の `retireByHandle` は
   handle 値一致で map lookup するため、**slot 再利用が起きた場合に破壊済み DSP への
   enqueueWithRetry（二重破壊）が静的に成立し得る**。shutdown 後は admission closed で
   slot 再利用が発生しないため現行契約下では発火しないが、**V-D-b（map erase を含む
   authority retire）であればこの latent risk が構造的に排除される**。
4. **B-era 分離マトリクス（S3-EBR で crash / S3-direct で clean）は F 修正後は予測力を持たない**。
   D0 §7.1 の通り、B-era の AV は allocator mismatch による MKL registry 汚染の
   「破壊後 churn による顕在化」であり、churn のタイミング差（EBR/direct の破壊時点の違い）で
   発火有無が揺れたもの。F 修正後は汚染源が消滅するため、G1〜G3 で再検証する。
5. **S3 は標準 soak では発火しない可能性がある**（EmergencyDrain は
   `m_healthMonitor.isEmergencyDrainRequested()` が true の場合のみ body 実行・
   md:38194。C1 fallback は RebuildThread 停止後のみ）。G1 の PASS 証跡には
   「S3 が実行されたことの観測」（DIAG `CLEAR_SHUTDOWN_DISPOSITION`）を必須とし、
   60-gen で 0 件の場合は targeted trigger の特定を G3 前提条件とする（§7.3）。

---

## 1. G0-1: S3/V-D enable/disable 実装箇所の特定

### 1.1 S3（shutdown clear 時 deferred DSP disposition）

| 項目 | 内容 |
| --- | --- |
| 関数 | `RuntimePublicationOrchestrator::clearDeferredForShutdown() noexcept` |
| 実ファイル | `src/audioengine/RuntimePublicationOrchestrator.cpp` |
| md 位置 | md:67397-67442 |
| 現在の状態 | **OFF（disposition なし）** — md:67406-67417 に D162-2-B 残課題→D0→F 繰り越しの注記 block のみ。`deferredSlot_.reset()`（md:67427）が slot を無処分で破棄 |
| switch 形態 | **なし** — 「retire 呼び出しの有無」そのものが switch。B-era は retire 挿入→実測 AV→撤去、の履歴 |
| DIAG | `[D162-2E_DEFERRED] event=CLEAR ... (no disposition — pre-existing mismatch)`（md:67418-67426） |

**呼び出し経路（全 3 経路・md 実測）**:

| 経路 | 呼び出し点 | thread | 発火条件 |
| --- | --- | --- | --- |
| (a) EmergencyDrain | ReleaseResources.cpp md:38203（`runtimeOrchestrator_->clearDeferredForShutdown()`） | MessageThread | **`m_healthMonitor.isEmergencyDrainRequested()` が true の場合のみ** body 実行（md:38193-38203）。RebuildThread は md:38046 `stopRebuildThread()`（join）**済み** |
| (b) C1 fallback | Orchestrator md:67455-67458（`requestDeferredClear()` 内・`rebuildThreadShouldExit` true 時） | MessageThread（Timer C2/C3/C4 由来） | RebuildThread 停止済みのみ。D135-9 Gate F で assert-free 契約 VALID 判定済み |
| (c) midrun clear | Orchestrator md:67495（`drainDeferredClearIfRequested()` 末尾） | RebuildThread（jassert 付き・md:67473） | **E-4d により md:67493 で既に S1 disposition（`timer-clear-midrun`）済み** → ここに S3 disposition を追加しても二重 retire は構造的に no-op（§3.3） |

### 1.2 V-D（VerifyDrained 時 最終 active/fading DSP 破壊）

| 項目 | 内容 |
| --- | --- |
| 関数 | `AudioEngine::releaseResources()` 内 VerifyDrained block |
| 実ファイル | `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` |
| md 位置 | md:38284-38390 |
| 現在の状態 | **OFF（明示的 compile-time disable）** — md:38372 `if (false && activeDSPToDestroy != nullptr)` / md:38381 `if (false && fadingDSPToDestroy != nullptr)`。B-era T2 direct destroy（`destroyRolledBackDSP`）のコード形状で休眠 |
| switch 形態 | **`if (false &&` 2 行** — G2 の ON 化はこの 2 行が対象 |
| 周辺の既存処理 | resolve（md:38309-38311・state=Active 中に実施）→ `dspHandleRuntime_.retire(handle)`（md:38314/:38323・**map は erase されない**）→ `tryShutdownQuiescentReclaim`（md:38317/:38324・Proof→Permit→reclaim、**state→Reclaimed・slot freelist 返却**）→ world clear（md:38347-38353）→（無効化された）destroy block（md:38370-38390） |

### 1.3 関連 authority の現状（S1/S2/S4/E-4d/quarantine）

D162-2-B/E の修正は全て現行 source に維持されていることを md で確認済み:

| site | md 位置 | 状態 |
| --- | --- | --- |
| S1 deferred-overwrite | md:67304-67313（`retireRegisteredDSP(..., "deferred-overwrite")`） | **有効** |
| S2 deferred-discard | md:67682（`"deferred-discard"`） | **有効** |
| S4 Rejected* x4 | md:67205 / :67225 / :67242 / :67253 | **有効** |
| dormant retry-exhausted | md:67345（`"retry-exhausted-discard"`） | 有効（dormant 分岐内） |
| E-4d timer-clear-midrun | md:67486-67494（`"timer-clear-midrun"`） | **有効** |
| E-2 dtor retireByHandle | CtorDtor md:31339-31349 | **有効** |
| E-1 drainForShutdown | CtorDtor md:31459 | **有効** |
| E-3 INV-D162-8 assert | CtorDtor md:31432-31440（pendingRetireCount==0・DIAG + jassert） | **有効** |

---

## 2. G0-2: S3/V-D destruction / EBR / direct 経路の実コード再構成

### 2.1 共通 authority 経路（`retireRegisteredDSP` → EBR）

```text
retireRegisteredDSP(req, origin)            [Orchestrator md:67535-67552]
  ├─ resolveDSPHandle(req.newDSP)           [AudioEngine.h md:48093-48109]
  │    → DSPHandleRuntime::resolve          [ISRDSPHandle.cpp md:51165-51183]
  │      ・generation 不一致 → nullptr (stale)
  │      ・state Reclaimed / Quarantined → nullptr
  │      ・Constructing / Active / Retired → instance（返る）
  ├─ nullptr → no-op（未登録/rollback/quarantine/reclaimed 済み）… INV-D162-3 の二重処分禁止の根拠
  └─ DSPLifetimeManager::retire(dsp)        [DSPLifetimeManager.cpp md:50124-50161]
       ├─ engine_.retireDSPHandleForRuntime(dsp)   [AudioEngine.h md:48111-48142]
       │    ・runtimeDSPHandleMap_.find(dsp)（pointer value）→ 不在なら false → **EBR enqueue も起きない**
       │    ・map_.erase(dsp)            ← ★ map erase はここでのみ行われる
       │    ・dspHandleRuntime_.retire(handle)     → registry state→Retired（md:51226-51231）
       │    ・requestReclaimHandle(handle)         → epoch 安全なら即 reclaim / 不安全なら pendingReclaimHandles_
       └─ router_->enqueueWithRetry(dsp, &AudioEngine::destroyDSPCoreNode, currentEpoch, Generic)
            → EBR D-queue → tryReclaim（epoch 安全確認後）→ destroyDSPCoreNode [Threading md:41116-41144]
```

**map erase は `retireDSPHandleForRuntime` のみ**（rollback `rollbackRegistration` は CAS で
Constructing→Reclaimed のみ・md:51261-51269、map に触れない）点が、S3/V-D の安全性議論の中心。

### 2.2 S3 lifecycle（現行 OFF → G1 ON 時）

```text
[G1 ON 時の想定経路]
deferredSlot_ (state=Constructing の registered DSP を保持)
  ↓ clearDeferredForShutdown [md:67403 hasDeferred_ 確認]
  ↓ (新規挿入位置) DIAG event=CLEAR_SHUTDOWN_DISPOSITION
  ↓ retireRegisteredDSP(deferredSlot_->request, "shutdown-clear")
  │    → resolve（Constructing → instance 取得可）
  │    → retire: map erase + registry Retired + requestReclaim + EBR enqueue
  ↓ deferredSlot_.reset() [md:67427] / hasDeferred_=false [md:67428]
  ↓ invalidateDeferredObligation() [md:67432]
  ↓
EBR D-queue entry
  → digest 点: EmergencyDrain tryReclaim [md:38209]
             / drainAllQuarantineStore [md:38364-38365]（Q/E/T）
             / waitForDrain(2000,2) [md:38399]
             / ~AudioEngine D5 graceful loop [CtorDtor md:31372-31394]
             / D8 drainAll [CtorDtor md:31420-31427] + E-3 assert [md:31432-31440]
  → 全て dtor body 内（全 member 生存）→ INV-D162-8 適合
```

- INV-D162-6（teardown 後 reclaim 禁止）: retire は EmergencyDrain/C1 時点＝dtor body 前 or
  releaseResources 内 → member 全生存。**適合**。
- INV-D162-7（dangling slot 値の map lookup 禁止）: S3 は `deferredSlot_->request.newDSP`
  （handle）起点であり、`activeRuntimeDSPSlot` 等の dangling legacy slot を使わない。**適合**。
- INV-D162-9（slot 出口全数カバー）: CLEAR 出口に disposition が付く。**適合**
  （DIAG イベント名は `CLEAR_SHUTDOWN_DISPOSITION` を新設し、E-4 会計を閉じる。§7.1）。
- 二重 retire の構造的不可能性: (c) 経路では E-4d の retire が既に map erase 済み →
  S3 の retire は find 失敗で no-op。**INV-D162-3 適合**。

### 2.3 V-D lifecycle（現行 OFF → G2 ON 時の 2 案）

```text
VerifyDrained [md:38285]
  ↓ resolve active/fading handle（state=Active 中・md:38309-38311）★Reclaimed 後は resolve 不能なため先出し必須
  ↓ dspHandleRuntime_.retire(handle)（Active→Retired・map 不変）
  ↓ tryShutdownQuiescentReclaim(handle)（Proof→Permit→reclaim → Reclaimed・slot freelist 返却・md:48192-48227）
  ↓ uiConvolverProcessor/uiEqEditor releaseResources [md:38331/:38335]
  ↓ world clear（requestShutdownClearNonRt + retirePublishedRuntimeWorldNonRt [md:38347-38353]）
  ↓ drainAllQuarantineStore（activeReaderCount==0 時・md:38364-38365）
  ↓ destroy block [md:38370-38390]   ← ★ G2 の対象

[V-D-a: B-era direct destroy をそのまま有効化]
  lifetimeMgrForFinalDSP.destroyRolledBackDSP(activeDSPToDestroy)   [md:38379]
    → AudioEngine::destroyDSPCoreNode(p) 直接呼び [md:50233-50241 → md:41116]
    → ★ map entry (dsp→handle) は erase されないまま残存
    → ★ registry は Reclaimed（instance=dangling）だが freelist に slot が戻る

[V-D-b: authority retire に差し替え（推奨）]
  lifetimeMgrForFinalDSP.retire(activeDSPToDestroy)                 [DSPLifetimeManager.cpp md:50124]
    → map erase + registry Retired(冪等) + requestReclaim(冪等) + EBR enqueue
    → destroy は EBR digest（dtor D5/D8 内）→ INV-D162-8 文言適合
```

### 2.4 G0 新規所見: V-D-a の stale map entry（latent 二重破壊経路）

```text
V-D-a 実行後の状態:
  runtimeDSPHandleMap_: { X → H{slot=s, gen=g} }   … erase されず残存
  registry_[s]: { instance: (X は破壊済み・dangling), state: Reclaimed, generation: g }
  freelist: s 返却済み

発火シナリオ（slot 再利用が起きた場合）:
  1. shutdown 後に新 DSP Y が create() → slot s 再発行（generation g+1）
  2. ~AudioEngine E-2: activeHandleAtDtor = dspHandleRuntime_.getActiveRuntimeDSPHandle()
     → retire() は activeRuntimeDSPHandle_ atomic をクリアしないため（ISRDSPHandle.cpp md:51226-51231）
       atomic は H{slot=s, gen=g} を保持し続ける（※「V-D 実行済みなら null」ではない）
  3. lifetimeMgr.retireByHandle(H) → findAndEraseByHandle(H) → map の X→H エントリが HIT
     → toDelete = X（**破壊済み**）→ enqueueWithRetry(X, destroyDSPCoreNode) → 二重破壊/UAF
  ※ slot 再利用（generation 不一致）の場合 reclaim 側は防御するが、
    findAndEraseByHandle の HIT 判定と enqueueWithRetry は generation 防御の外側で起きる。

現行契約下の評価: shutdown 完了後は admission closed（publish 不成立）のため slot 再利用は
発生しない → 実発火は現行フローでは不可能。ただし「将来 VerifyDrained 以降で DSP を
構築する変更」が入った瞬間に発火し得る latent 欠陥であり、V-D-b（map erase あり）なら
構造的に発生しない。
```

### 2.5 スレッド / ownership マトリクス（G1/G2 適用後）

| site | thread | 実行時点の member 状態 | ownership 遷移 |
| --- | --- | --- | --- |
| S3 (a) EmergencyDrain | MessageThread | RebuildThread join 済み・全 member 生存 | deferredSlot → EBR → destroy（dtor body 内 digest） |
| S3 (b) C1 fallback | MessageThread | RebuildThread 停止済み・全 member 生存 | 同上 |
| S3 (c) midrun 経由の二重呼び | RebuildThread | E-4d 済みで map 不在 → no-op | 変化なし |
| V-D (a/b) | MessageThread（releaseResources 内） | 全 member 生存・admission closed | resolved pointer → (a) 直接破壊 / (b) EBR |

- `enqueueWithRetry` の producer 契約は NonRT（ISRRetireRouter.h md:55929-55930）→
  MessageThread からの呼び出しは合法。
- plain member（`deferredSlot_` 等）の単一 writer 契約は D135-9 Gate F で確認済みのまま
  （S3 挿入は契約対象メンバの読み書き順序を変えない）。

---

## 3. G0-3: D162-2-F allocator 修正の ownership chain 効果

### 3.1 F 修正の現状確認（md 実測）

| 確認項目 | 結果 |
| --- | --- |
| `create()` が `convo::aligned_malloc` を使用 | **確認**（AudioSegmentBuffer.h md:3847-3850） |
| 所有者 `ScopedAlignedPtr`（`leftSamples_`/`rightSamples_`）の free | `aligned_free` → `CONVOPEQ_ALIGNED_FREE` → `mkl_free`（MKL ビルド時・DiagnosticsConfig.h md:11743-11744） |
| alloc/free pair 一致 | **一致**（同一 allocator identity） |
| 失敗時 free も `convo::aligned_free` | **確認**（md:3853-3854・nullptr free no-op 安全） |
| `#include <malloc.h>` 削除 | **確認**（md:3814-3819 に存在しない） |
| 契約注記（D162-2-F） | **確認**（md:3835-3842） |

### 3.2 AudioSegmentBuffer の S3/V-D destruction path への非含有（最終確認）

全 source の `AudioSegmentBuffer` 参照は次の 4 ファイルのみ（md 全走査・337 ファイル）:

| ファイル | 役割 | S3/V-D path に関与? |
| --- | --- | --- |
| `src/AudioSegmentBuffer.h` | 定義本体 | — |
| `src/NoiseShaperLearner.cpp:23039` | 唯一の `create()` 呼び出し・`segmentBuffer` は unique_ptr member | **NO** — `~NoiseShaperLearner`（md:23060-23064・`shutdownWorkerThread` のみ）は engine member teardown で走り、DSPCore（S3/V-D の破壊対象）とは別オブジェクト |
| `src/NoiseShaperLearner.h:24805` | `std::unique_ptr<AudioSegmentBuffer> segmentBuffer` 宣言 | NO |
| `src/audioengine/RuntimeHealthMonitor.{h,cpp}` | `setLearnerSegmentBuffer` — **caller なし（dead code）・non-owning 観測 pointer** | NO |

**結論: DSPCore の破壊経路（EBR `destroyDSPCoreNode` / direct `destroyRolledBackDSP`）に
AudioSegmentBuffer は含まれない。** F mismatch と S3/V-D destroy の因果は
「直接のオブジェクト関係」ではなく D0 §7.1 の **ヒープ状態依存の顕在化** のみ:

```text
B-era（mismatch 存在時）:
  ~AudioSegmentBuffer の foreign mkl_free が MKL block registry を汚染
    → S3/V-D destroy が shutdown 窓で大量の mkl_malloc/mkl_free churn を追加
    → 汚染 registry への接触で AV（churn のタイミングで発火有無が揺れる
       = S3-EBR で crash / S3-direct で clean の揺れを説明）

F 修正後:
  汚染源（foreign free）消滅 → S3/V-D destroy の追加 churn は無害
  → B-era 分離マトリクスの結果は予測力を持たない（G1-G3 で再証明する意味）
```

### 3.3 undo リスク（G0 項目 9）

- `create()` が唯一の構築経路（private ctor + factory・F-2 監査を md で再確認）。
  `leftSamples_`/`rightSamples_` への外部 injection path なし。
- CRT `_aligned_malloc`/`_aligned_free` の残存使用は `DiagnosticsConfig.h` の
  `system_aligned_malloc`/`system_aligned_free`（md:11716-11731・**非 MKL フォールバック専用**）のみで、
  alloc/free が同一ファミリーで pair — mismatch 構造なし。
- 構造的に mismatch が再混入する経路は存在しない。回帰ベクトルは「create() の手作業による
  書き戻し」のみで、契約注記（md:3835-3842）+ 本監査で防御済み。

### 3.4 S3/V-D 破壊対象 DSPCore の allocator 整合（D0 F1 の維持確認）

`destroyDSPCoreNode`（md:41116-41144）: `core->~DSPCore()` + `convo::aligned_free(core)`。
DSPCore は `RuntimeBuilder` が `aligned_make_unique`（AlignedAllocation.h md:2831-2844・
`aligned_malloc`↔`aligned_free` pair）で構築 — **破壊側 pair 一致**。D0 F1 判定の再確認済み。

---

## 4. jassert `detectStuckReaders` の扱い（別シグネチャ規定）

- 位置: `EpochDomain.h` `detectStuckReaders`（md:77222・3 パス評価: Chronic→Warning→EpochGap）。
  F-era dump 26984 の jassert は pend=1 滞留時の Debug-only assert（doc/work88/D162-2-F §5）。
- **D162-2-F 固有 allocator crash とは独立の pre-existing Debug-only issue** として
  G1〜G3 の各段階で以下の分類規約を適用する（§7.2 Signature 表）。
- `detectStuckReaders` の jassert が Debug 6-gen で発火した場合:
  (1) dump 取得 + symbolize、(2) stack に AudioSegmentBuffer / D0 chain が**無い**こと、
  (3) E-3 jassert（CtorDtor md:31438）が発火して**いない**ことを確認の上、
  「pre-existing Signature B」として記録（gate blocking としない）。
- 逆に **E-3 jassert の発火（S3/V-D ON 時の pendingRetire 残留）は新規 Signature C** —
  INV-D162-8 の実行時違反であり、その段階で即 stop。

---

## 5. G0 チェックリスト（指示 9 項目の回答）

| # | 項目 | 結果 |
| --- | --- | --- |
| 1 | D162-2-F allocator pair が現行 source に残っているか | **YES**（§3.1） |
| 2 | AudioSegmentBuffer の alloc/free が convo::aligned_malloc / aligned_free で一致 | **YES**（§3.1・非 MKL fallback も同一ファミリー pair） |
| 3 | S3 の実装箇所 | Orchestrator `clearDeferredForShutdown()` md:67397-67442・OFF（§1.1） |
| 4 | V-D の実装箇所 | ReleaseResources VerifyDrained md:38284-38390・`if (false &&` x2 で OFF（§1.2） |
| 5 | S3/V-D の switch | S3=retire 呼び出しの有無（switch なし）/ V-D=`if (false &&` 2 行（§1） |
| 6 | D162-2-B direct/EBR の現在設定 | S3: 両形状とも撤去済み（注記のみ）・V-D: direct 形状で休眠。**現在 = S3 OFF / V-D OFF**（§1） |
| 7 | shutdown / world-clear / destroy ordering | EmergencyDrain(38193) → quarantine cleanup(38243) → VerifyDrained(38285) → V-D resolve/retire/reclaim(38309-38327) → world clear(38347) → destroy block(38370, 無効) → waitForDrain(38399) → …dtor D5/D8（§2.2-2.3） |
| 8 | AudioSegmentBuffer が S3/V-D destruction path に含まれるか | **含まれない**（NoiseShaperLearner 専属・§3.2） |
| 9 | F 修正 undo で D0 signature が戻る構造か | **戻る構造なし**（唯一構築経路 + 注記防御・§3.3） |

---

## 6. G0-4: staged gate matrix（G-1 → G-2 → G-3 → G-4）

### 6.0 共通規則

- 基準ソース: ConvoPeq.md `Generated: 2026-09-04 18:29:10` の working tree。
- 各 stage の production 変更は最小（1 関数 block 単位）とし、stage 間で全 revert 可能にする。
- Gate 実施順（ユーザー指示どおり short-soak first）:
  `Build(Debug/Release/RWDI) → CTest → Debug 6-gen → RWDI 6-gen → RWDI 60-gen`。
- **Signature 分類規約（§4）**:
  - Signature A（D0 chain: `~AudioSegmentBuffer → aligned_free → mkl_free`）= いかなる dump にも出現禁止（F 回帰 = block）。
  - Signature B（`detectStuckReaders` jassert・Debug のみ）= 記録のみ・非 block。
  - Signature C（上記以外・E-3 jassert 発火を含む）= **即 stop・dump 解析まで次 stage 禁止**。

### 6.1 Stage G1 — S3 のみ staged re-enable（S3=ON / V-D=OFF）

**変更（1 ファイル 1 block）**: `RuntimePublicationOrchestrator::clearDeferredForShutdown()` 内、
`deferredSlot_.reset()`（md:67427）の直前に E-4d と同型の disposition を挿入:

```cpp
// ★ D162-2-G1 (S3 re-enable): shutdown 時 deferred DSP disposition。
//   D0 PASS-B で proven された allocator mismatch は D162-2-F で修正済みのため、
//   D162-2-B で AV を誘発した本破壊を authority（EBR・INV-D162-8 準拠）で再有効化する。
//   二重 retire は map erase 済み（E-4d midrun 経由）の場合 no-op（INV-D162-3）。
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        juce::Logger::writeToLog(juce::String::formatted(
            "[D162-2E_DEFERRED] event=CLEAR_SHUTDOWN_DISPOSITION gen=%llu dsp=%p",
            (unsigned long long)deferredSlot_->request.generation,
            (void*)engine_.resolveDSPHandle(deferredSlot_->request.newDSP)));
#endif
        retireRegisteredDSP(deferredSlot_->request, "shutdown-clear");
```

- 既存の `event=CLEAR ... (no disposition)` DIAG ログ（md:67422）は disposal 完了後に
  誤解を生まないよう文言更新（`no disposition` 表記の除去）。
- V-D は**一切触れない**（`if (false &&` 維持）。

**G1 PASS 条件（ユーザー指示 8 項目 + 補強）**:

| # | 条件 | 測定手段 |
| --- | --- | --- |
| 1 | `0xC0000005` なし | 全 run exit code + CrashDumps 0 件（Signature B を除く） |
| 2 | D0 allocator signature（Signature A）なし | 全 dump symbolize |
| 3 | S3 destruction count が期待値に一致 | `[D162-2B_RETIRE] origin=shutdown-clear` 件数 == `CLEAR_SHUTDOWN_DISPOSITION` 件数 == `[D117_DESTROY]` 該当 DSP 件数 |
| 4 | EBR `pend` が永続増加しない | soak `[MEM_SNAP]`/EBR telemetry: pend 収束・最終 0（E-3 DIAG 一致） |
| 5 | residual = 0 | enqueued == destroyed + placeholder |
| 6 | shutdown clean | `[D162-2E] INV-D162-8` ログ出力 0 件・`markShutdownComplete` 到達 |
| 7 | Audio drop / click / XRUN なし | soak XRUN/underrun telemetry 差分 0 |
| 8 | AudioSegmentBuffer destruction 正常 | Signature A 不出現（条件 2 と同値） |
| 9 | E-4 会計 closure | CREATE == CONSUME + DISCARD + OVERWRITE + CLEAR（全 CLEAR に disposition 対応） |
| 10 | INV-D162-6/7/9 静的適合 | 本監査 §2.2 の通り（コード review で最終確認） |

**G1 観測可能性条件（重要）**: EmergencyDrain は `isEmergencyDrainRequested()` が true の場合のみ
body を実行するため、標準 soak で S3 が発火しない可能性がある。G1 の 60-gen で
`CLEAR_SHUTDOWN_DISPOSITION` 件数が **0 の場合**、G1 は「構造 PASS・実行時未検証」として
マークし、**G3 開始前に S3 を確実に発火させる観測経路（EmergencyDrain 要求が発生する
soak 条件、または C1 fallback 到達条件の実測）を確定する**ことを必須とする。

### 6.2 Stage G2 — V-D のみ staged re-enable（S3=OFF / V-D=ON）

**G1 完全 PASS 後のみ実施。** S3 変更は revert（`shutdown-clear` retire を除去）し、
V-D のみを有効化して独立検証。

**設計分岐（G2 開始時に確定）**:

| 案 | 変更 | 利点 | 欠点 |
| --- | --- | --- | --- |
| **V-D-b（推奨）** | md:38379/:38388 の `destroyRolledBackDSP(...)` → `retire(activeDSPToDestroy/fadingDSPToDestroy)` に差し替え + `if (false &&` 解除 | INV-D162-8 文言適合（EBR 単経路）・map erase で stale entry 消滅（§2.4 の latent 二重破壊経路を構造排除）・S1-S4 と同一 authority | B-era で未実測の新しいコード形状（B 初版 VD-EBR は crash したが、それは mismatch 時代の結果・§3.2） |
| V-D-a（最小差分） | `if (false &&` → `if (` の 2 行のみ | B-era で実装済み形状・diff 最小 | §2.4 の stale map entry latent risk を内包（現行契約では不発だが将来脆弱）・INV-D162-8 文言（「shutdown 破壊は必ず EBR 経由」）との不整合を注記で解决する必要 |

推奨は **V-D-b**。理由: (i) INV-D162-8 が C 監査で確立した契約であり、direct destroy は
D0 以前の誤帰属（「EBR 遅延が AV の原因」という B-era 解釈）に基づく選択だったことが
D0 §7.1 で判明した、(ii) V-D-b は §2.4 の latent risk を構造的に排除する。
V-D-a を選択する場合は、md:38299-38305 の注記（T2 direct destroy の正当化）を
INV-D162-8 例外として明示的に書き換えることを G2 の変更内容に含める。

Gate ladder は G1 と同一。追加観測: `tryShutdownQuiescentReclaim` の jassert(reclaimed)
（md:38318/:38325）発火有無、V-D destroy 対象 DSP の `[D117_DESTROY]`/`[DSP_DESTROY_FOOTPRINT]`
一致、および G2 の V-D-b 採用時は EBR digest が dtor D5/D8 内で完了すること（E-3 assert）。

### 6.3 Stage G3 — S3 + V-D 同時 re-enable（本命 Gate）

- G1 変更 + G2 変更を両方適用（S3=ON / V-D=ON）。
- これは D162-2-B で `0xC0000005` を発生させた組み合わせの F 修正後再証明である。
- Gate ladder 同一。PASS 条件は G1 の 10 項目 + G2 追加観測の双方。
- Signature C 出現時は G1/G2 単独では検出できなかった交互作用と特定し、
  どちらの経路起因かを `[D162-2B_RETIRE] origin=` / `[D162-2B_DESTROY] origin=` で分離する。

### 6.4 Stage G4 — 60-gen final soak（S3+V-D ON）

必須観測値（ユーザー指示）+ ライフサイクル突合:

```text
exit = 0x00000000
crash = 0（Signature A/C 不出現）
residual = 0
EBR pend = 0（終端） / EBR overflow = 0
```

- construct / retire / destroy ライフサイクル突合:
  `[D162-2B_RETIRE]`（全 origin 合計）+ 既存 EBR retire（published old DSP）== `[D117_DESTROY]` 件数
  + terminal 残留 0。`[DSP_CREATE_FOOTPRINT]`（D162-1R-B 計装）総数 == `[DSP_FOOTPRINT_RELEASED]` 総数。
- Practical Stable ISR Bridge 観点: retire → Epoch → reclaim → delete の閉包。
  shutdown は Drain（D5 graceful loop）→ Reclaim（tryShutdownQuiescentReclaim / requestReclaimHandle
  digest）→ Verify Empty（E-3: pendingRetireCount()==0 DIAG + Debug jassert）まで確認。
- E-4 deferred 会計 closure（G1 条件 9）を G4 でも維持。
- G4 PASS → D162-2-G 完了。S3/V-D の OFF 注記（md:67406-67417, md:38296-38305, :38372/:38381）を
  ON 契約に書き換えるドキュメント整備を最終作業として含める。

### 6.5 stage 間 revert 規則

| stage 開始時の source 状態 | 方法 |
| --- | --- |
| G2 開始 | G1 の S3 retire 挿入を除去（注記を D162-2-G1 履歴込みで復元） |
| G3 開始 | G1 変更を再適用（G2 の V-D は維持） |
| G4 開始 | source 変更なし（G3 状態のまま 60-gen 再実行・決定論確認 x2 推奨） |

---

## 7. 残課題 / G1 への引き継ぎ事項

### 7.1 DIAG イベント設計（G1 に含める）

- 新イベント `CLEAR_SHUTDOWN_DISPOSITION`（Orchestrator・`shutdown-clear` origin と対）。
- 既存 `event=CLEAR ... (no disposition — pre-existing mismatch)` の文言更新
  （mismatch 解消済みのため表記が偽になる）。
- E-4 会計: CLEAR 出口 = `CLEAR`（無条件） + `CLEAR_SHUTDOWN_DISPOSITION`（slot 保持時） +
  `CLEAR_MIDRUN_DISPOSITION`（midrun 経由）の 3 層で収支が閉じることを 60-gen で確認。

### 7.2 Signature 分類の運用（§4 の実行手順）

Debug 6-gen で crash した場合の fixed 手順:
1. dump 取得（既存 CrashDumps flow・minidump + llvm-symbolizer）。
2. stack 内に `AudioSegmentBuffer` / `aligned_free` / `mkl_free` chain（Signature A）が
   あれば **F 回帰 = 即 stop**（G1 失敗・F の再監査）。
3. stack が `detectStuckReaders → common_assert → _wassert`（Signature B）のみなら、
   pend 値・滞留 reader 情報を記録して続行（pre-existing・別 work item）。
4. `EpochDomain` 以外の新規 frame 群 or E-3 jassert = Signature C → 即 stop。

### 7.3 S3 観測可能性の事前検証（G1 中に解決）

- 60-gen soak で `isEmergencyDrainRequested()` が true になる条件が既に存在するか
  （D162-2-B の分離試験で S3 が destroy を実行した実績から、EmergencyDrain 要求 or C1 が
  到達する条件は存在した可能性が高い）。
- G1 の Debug/RWDI 6-gen 時点で `[D162-2E_DEFERRED]` ログから発火有無を即確認し、
  0 件なら 60-gen 前に soak パラメータ（IR reload 頻度等）の見直しまたは
  EmergencyDrain 要求条件の実測を行う。
- 最悪時（標準 soak で全く発火しない）: G1 を「構造 PASS」として閉じず、
  CTest/harness レベルの targeted trigger（deferred 保持状態での releaseResources）を
  **テストコード変更 0 で**実現できる既存試験（SoakPublishIntegrationTests 等）の有無を
  確認してから G3 に進む。テスト変更が必要な場合はユーザー承認を取る。

### 7.4 判定

**G0 = PASS。** G1 の最小変更（§6.1 の 1 block + DIAG 文言更新）を実施可能。
G2 の V-D-a/V-D-b 選択は G1 PASS 後にユーザー指示で確定すること（推奨: V-D-b）。

---

## 8. 証跡

- 基準ソース: ConvoPeq.md `Generated: 2026-09-04 18:29:10`（4,821,896 bytes・md 行番号で引用）
- 参照監査: evidence/D162-2C_SHUTDOWN_TEARDOWN_AUDIT.md（§3/§4/§10/§13/§14）、
  evidence/D162-2-D0_HEAP_CORRUPTION_ORIGIN_AUDIT.md（§7.1 churn 顕在化機構）、
  doc/work88/D162-2-B_REPAIR_IMPLEMENTATION_REPORT.md（§2 S3/V-D 履歴・§5 residual）、
  doc/work88/D162-2-E_TEARDOWN_SAFETY_REPAIR_REPORT.md（E-1..E-4・§8 S3/V-D 無効維持）、
  doc/work88/D162-2-F_ALLOCATOR_CONTRACT_REPAIR_REPORT.md（F-1..F-9）
- 実コード抽出（md 行）: Orchestrator 66807-67741 / ReleaseResources 37842-38633 /
  CtorDtor 31133-31486 / AudioEngine.h 43752-48930 / DSPLifetimeManager 50082-50283 /
  ISRDSPHandle 51102-51713 / AudioSegmentBuffer.h 3809-3964 / AlignedAllocation.h 2703-2933 /
  DiagnosticsConfig.h 11691-12016 / RebuildDispatch 39662-39830
