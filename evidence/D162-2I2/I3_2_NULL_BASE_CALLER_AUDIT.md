# D162-2-I3-2 — Release Null-Base Admission CAS Caller Audit（read-only）

```text
Type:      read-only structural/root-cause audit（production/test/CMake 変更 0）
Baseline:  ConvoPeq.md 2026-09-05 12:02:18 / I2 PASS / I3-1
Tools:     BIGOBJ CodeView (.debug$S) 自作パーサ（evidence/D162-2I2/i3_2_cvparse.py）+
           Python mini-debugger stack capture + COFF symbol table 解析
判定:      **NO-GO（repair contract へ進めるには producer 未確定）—
           ただし faulting C++ statement と rdx identity は確定した**
```

---

## 1. faulting instruction → C++ statement（確定）

### 1.1 CodeView 行マッピング（Release obj・BIGOBJ .debug$S 解析）

crash 命令は COMDAT セクション 746 (.text$mn, 0xF5 = 245 バイト) の offset 0xAD。同セクションの
.companion .debug$S (section 747) の line fragment:

| section offset | source line | 対応コード（ISRRuntimePublicationCoordinator.cpp） |
| --- | --- | --- |
| 0x00 | 82 | `if (boundary != NonRTWorld \|\| newWorld == nullptr) {` |
| 0x07 | 83 | `publishAtomic(state_, CoordinatorState::Faulted, ...)` |
| 0x1A | 93 | `prevSeqId = prevWorld ? prevWorld->publication.sequenceId : 0` |
| 0x41 | 94 | `prevEpoch = ...` |
| 0x48 | 95 | `prevGen = ...` |
| 0x4F | 97 | `hasPrevious = ...` |
| 0x64 | 101 | `if (!(isAfter(sequenceId, prevSeqId) && ...))` |
| 0xA1 | 104 | `publishAtomic(state_, CoordinatorState::Faulted, ...)`（monotonicity 違反） |
| **0xAD** | **109** | **`convo::publishAtomic(state_, CoordinatorState::Publishing, ...)`** ← crash |
| 0xB1 | 110 | `publishAtomic(swapPending_, true, ...)` |
| 0xD1/0xE3/0xE7/0xEC | 117/119/120/121 | `pubWorld->publication = PublicationSemantic{...}` 等 |

**行番号と命令の対応は 4 重に検証済み**:
1. 書き込み値 **2 == `CoordinatorState::Publishing`**（enum: Bootstrapping=0, Ready=1,
   Publishing=2, Transitioning=3, Pressure=4, ShuttingDown=5, Faulted=6 — h:119-127）。
2. 隣接 write **1 == `swapPending_ = true`**（:110）。
3. 隣接 write **[r9+0x198/0x1A0/0x1A8] = r11/r8(6)/rbx** — line 119 の
   `pubWorld->publication = PublicationSemantic{sequenceId, epoch, mappedGeneration, prevSeqId}`
   の 4 フィールド（prevSeqId は 6 → r8 経由）。
4. **Faulted=6 の兄弟 write が同一 base [rdx+0x65] に存在**（offset 0xA1 = line 104 の
   違反 path・offset 0xEC 付近 = line 83/84 の guard path）— すべて `state_` の同一 member。

### 1.2 faulting C++ statement（確定）

```cpp
// ISRRuntimePublicationCoordinator.cpp:109
// RuntimeIntentCoordinator::commit(PublishAuthority, RuntimeBoundary,
//                                 const void* newWorld, u64 version,
//                                 SequenceId, Epoch, u64 mappedGen,
//                                 const RuntimeState* prevWorld)
convo::publishAtomic(state_, CoordinatorState::Publishing, std::memory_order_release);
// → mov byte ptr [rdx+0x65], 2   : rdx = this (RuntimeIntentCoordinator*) = NULL
```

**`this == nullptr` で commit が呼ばれた**。rdx は関数冒頭 `mov rdx,rcx`（arg1 = this）以降
一度も再代入されない（:04〜:AD の全命令走査で確認）ため、**rdx ≡ 呼び出し時の arg1 ≡ null**。

## 2. rdx の object identity（確定）

| 仮説 | 判定 |
| --- | --- |
| RecoveryAdmissionTable 本体 / slot / slots_ 配列 | **否** — 書き込み値 2 は `CoordinatorState::Publishing`（coordinator state）。table/slot の
 state は `ObligationState`（値域が別）。隣接フィールド (0x64=swapPending_, 0x65=state_) は coordinator の member pair |
| atomic storage / STL lock-pool base | **否** — lock-pool spinlock（`xchg/pause/backoff`、別関数 0x...F410）は無関係。crash 命令は
 `publishAtomic` の直接 store |
| **RuntimeIntentCoordinator `this`（state_ member: this+0x65, swapPending_: this+0x64）** | **確定** |

**Case 分離（指示 §4）**: **Case B（STL lock-pool null）は棄却**。Case A（null coordinator base）の
形で確定 — ただし「null base の producer」は下記 §5 のとおり未確定。

## 3. `recoveryAdmissions_` 全 call-site inventory（29 site・全bounded）

`RecoveryAdmissionTable<32> recoveryAdmissions_;`（h:1134・`std::array<LogicalRecoveryObligation, 32> slots_{}` —
value-initialized・slot size 320 bytes（RuntimeBuildSnapshot 含む））。全アクセス:

| # | site | 呼出 thread | アクセス | 境界 |
| --- | --- | --- | --- | --- |
| 1 | cpp:899-902 findByKey（recovery 可能判定） | Rebuild | slots_[i].lifecycle load | i<32 ループ ✓ |
| 2 | cpp:933 findByKey（admission） | Rebuild | 同上 | ✓ |
| 3 | cpp:936/951/958 slot(existing).lifecycle | Rebuild | load / CAS | existing < 32 ✓（findByKey 返却値） |
| 4 | cpp:964 tryInsert | Rebuild | capacity guard 付き新規 slot | liveCount_<32 ✓・i<32 ループ ✓ |
| 5 | cpp:972/988 slot(*ins) / slot(slotIdx) | Rebuild | payload write（line 119 系） | *ins/slotIdx < 32 ✓ |
| 6 | cpp:1054 resolve(obligationId, ...) | CoordinatorLoop | id 走査 + lifecycle CAS | i<32 ループ ✓ |
| 7 | cpp:1080-1081 adjudicate 前半 | CoordinatorLoop | 全 slot 走査 | ✓ |
| 8 | cpp:1120-1121 adjudicate 後半 | CoordinatorLoop | 全 slot 走査 | ✓ |
| 9 | cpp:1163-1164 redrive 走査 | CoordinatorLoop | load | ✓ |
| 10 | cpp:1192+ terminalize 走査 | CoordinatorLoop | load | ✓ |
| 11 | cpp:1215-1222 obligationId 検索 + slot(idx) | CoordinatorLoop | id 照合 | idx<32 ✓（見つからなければ capacity） |
| 12 | cpp:1415 slot(slotIdx).lifecycle CAS | CoordinatorLoop | CAS | slotIdx は 11 の検索結果 |
| 13 | cpp:1462-1463 drain/telemetry 走査 | CoordinatorLoop | load | ✓ |

**結論: recoveryAdmissions_ 経由の OOB write（i ≥ 32）は source 上存在しない**
（全ループが `i < kCapacity`、tryInsert は capacity guard、resolve は検索結果使用）。
**crash 命令は recoveryAdmissions_ ではなく commit の `state_` publish であり、
I3-1 記録の「RecoveryAdmissionTable 系コード」表記は本監査で訂正される。**

## 4. caller chain（確定分）

```text
CoordinatorLoop worker thread（runCoordinatorPhase）
  → runtimePublicationBridge_.processIntent(*this, lifetimeMgr)   (Threading.cpp:298)
    → PublishIntentHandler::handle → PublishExecutor{}.executePublish(
        ctx.engine.worldAuthority(), intent, ctx)                 (ProcessIntent.cpp:149)
      → authority.publish(owner, metadata, &committed)            (RuntimePublishExecutor.h:60)
        → coordinator_.commit(Granted, boundary, newWorld, version,
                              seq, epoch, gen, prevWorld)          (RuntimeWorldAuthority.h:299)
          → [this = NULL で到達] → line 109 で AV (WRITE @ 0x65)
```

- `coordinator_` は **RuntimeWorldAuthority の reference member**（RuntimeWorldAuthority.h:333）で、
  AudioEngine ctor（CtorDtor.cpp:29 `worldAuthority_(runtimePublicationBridge_)`）により
  `runtimePublicationBridge_`（member subobject・h:4964）に束縛される。
- **reference member の値が null になる正当パスは存在しない**（束縛は非 null subobject）。
  ⇒ **worldAuthority_ オブジェクトの coordinator_ フィールド（8 バイト）が構築後に 0 で
  上書きされた** = post-construction memory corruption（8 バイト zero write）が唯一の整合説明。
- crash は CoordinatorLoop worker（crash thread: 毎 run 新しい tid・stack に main thread frame なし・
  ucrtbase thread-start + harness thread-proc frame あり）で発生。**bootstrap commit（main thread・
  loop 起動前）は成功している**（crash が loop 起動後の最初の worker publish で発生する形）。

### AudioEngine layout 上の隣接性（writer 候補の絞り込み）

```text
AudioEngine members（宣言順・h:4964-4966）:
  runtimePublicationBridge_   ← RuntimeIntentCoordinator（巨大学: recoveryAdmissions_ 10KB・
                                 intentQueue_ (MpscBoundedRing<Intent,4096>)・
                                 quarantineFallbackQueue_ (1024×Intent)・quarantineService_ 等）
  worldAuthority_             ← RuntimeWorldAuthority { coordinator_ (reference, 8B) ← 先頭 member! }
```

- `RuntimeWorldAuthority` の **第 1 member = coordinator_（8 バイト reference）**。
- `runtimePublicationBridge_`（直前の member）の **クラス終端直後に worldAuthority_.coordinator_ が
 アライン配置される** → **bridge の tail member（quarantineService_ 等の内部 buffer）からの
  8〜16 バイト超過書き込み（OOB / memset overrun）が正確に coordinator_ を潰す位置関係**。
- 値が **0（NULL）で上書き**される点から、writer は「0 を書くコード」
  （例: atomic store(0)・memset(0)・zero-initialized object の copy 等）と推定される
  （garbage ではなく綺麗な 0 だった場合）。

## 5. lifetime chain（確定分 + 未確定分）

```text
確定:
  AudioEngine ctor → runtimePublicationBridge_ 構築 → worldAuthority_(bridge) 構築
    → initialize(): bootstrap commit（main thread・成功 — coordinator_ はこの時点で有効）
    → startCoordinatorLoop()（Init.cpp:123）→ worker 起動
    → worker: processIntent → ... → publish → commit → this=NULL で AV
  ⇒ coordinator_ フィールドは「bootstrap commit 成功後〜 worker の最初の publish まで」の
    間に 0 化された（時間窓は確定）。

未確定:
  0 化した writer（どの code / どの buffer からの溢れか）— data breakpoint でないと確定しない
  （write 時点で AV にならず、read 時（commit の reference load）に顕在化するタイプ）。
```

shutdown/reconfigure 状態との対応: crash run は harness のため reconfigure/shutdown は未発生
（bootstrap → rebuild intent 処理の初期 segment で crash）。P0（reconfigure × admission）とは
**別コンテキスト**。

## 6. Release-only 条件（仮説1-5 分離）

| 仮説 | 判定 | 根拠 |
| --- | --- | --- |
| 1 UB 顕在化 | **部分維持** — 「0 化する writer」自体が UB（境界外/破壊後書込み）。ただし crunky な
 コードではなく「8 バイトの綺麗な 0」のため、単純な 1 バイト overflow より構造的 | 確定 evidence なし |
| 2 data race | **有望** — bootstrap 成功後〜 loop worker 最初の publish の間に coordinator_ が 0 化。
 Release の timing（最適化で window が短くなる等）で顕在化差が出る | 時間窓確定のみ |
| 3 destruction ordering | **低** — crash は engine 構築直後（0.07s）で破棄は未発生 | 0.07s 実測 |
| 4 config/build 差 | **低** — member layout は config 非依存（同一ヘッダ）・RWDI は同一最適化系で PASS | RWDI PASS 実測 |
| 5 stale object | **低** — engine は 1 個体・再構築なし | harness 構造 |

**残る中核問い = 「誰が worldAuthority_.coordinator_ に 0 を書いたか」** — これは
data-breakpoint（write watch）を coordinator_ の実アドレスに設定しないと確定しない。
I3-2 の read-only 境界を超える（実行時 instrumentation）ため I3-3 以降に持ち越し。

## 7. I2 independence

- I2 修復（PrepareToPlay.cpp CallerDestroy 応答）は commit 呼出経路と無関係の TU。
- Release harness crash は G2 世代から存在（0xc0000374・当時は旧 test suite）— I2 前から
  の pre-existing。本日 live capture した現行 tree でも同一（I2 適用済み binary）。
- **I2-independent 確定**。

## 8. root cause（現時点の確定度）

```text
確定:
  faulting C++ statement = ISRRuntimePublicationCoordinator.cpp:109
                           publishAtomic(state_, Publishing)（commit 内）
  rdx = this = RuntimeIntentCoordinator* = NULL
  AV = WRITE @ null+0x65（state_ member）
  caller = RuntimeWorldAuthority::WriteAccess::publish（h:299・CoordinatorLoop worker）
  null の所在 = worldAuthority_.coordinator_ reference field の値が 0
  発生 window = bootstrap commit 成功後〜 loop worker 最初の publish

未確定（NO-GO 理由）:
  coordinator_ フィールドを 0 化した writer（overflow 元 buffer / race 構図）
  → data breakpoint / page-heap による writer 特定が次の手順
```

## 9. 最小修正境界（I3-3 用の草案・実装は禁止のまま）

1. **null guard の追加は禁止**（指示 §8 どおり — invariant 隠蔽の恐れ）。
2. 正しい修正単位は「coordinator_ フィールドを 0 化する writer の特定と除去」であり、
   そのための診断は:
   - x64dbg hardware breakpoint（DR0-3）を `&worldAuthority_.coordinator_` の実アドレスに
     設定して write を捕獲（engine instance address は 1 回の実行で特定可能）。
   - または page-heap / Application Verifier（要 admin・環境整備）。
3. writer 特定後、修正は「当該 buffer の境界修復」または「lifetime 排斥修正」となる。

## 10. 修正禁止事項（遵守確認）

- null guard / assert 追加: 未実施 ✓
- `if (!recoveryAdmissions_)` 系: 未実施 ✓（そもそも null は coordinator 側）
- release idempotent 化・admission reopen・I2 closure 変更: 未実施 ✓
- 本監査での生成物: i3_2_cvparse.py（解析ツール）+ 本記録のみ。
```
```