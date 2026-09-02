# D162-2-B Work Report — Repair Implementation

- Work item: D162-2-B（H1 修正の実装。案 C′ 基調・D162-2-A repair contract 準拠）
- Date: 2026-09-03
- 基準ソース: ConvoPeq.md `Generated: 2026-09-02 21:53:52` 相当の working tree（行番号は実装時点の実体に基づき再確認）
- 判定: **実装完了（部分適用）** — S1/S2/S4 を authority 統一で実装し、S3/V-D（shutdown 時破壊）は
  実測された exit AV のため意図的に無効化して **D162-2-C へ繰り越し**（§5 Residual Risk）
- Evidence: evidence/D162-2B_EVIDENCE.md

## 0. 実装結果サマリ

| 項目 | D162-1P（修正前） | D162-2-B（修正後・60 gen soak） |
| --- | ---: | ---: |
| enqueued generations | 60 | 59 |
| published | 11 | 11 |
| DSPCore destroyed（実行中） | 11 | 54 |
| **retained（orphan）** | **49 / 60（82%）** | **6 / 59（10%）** |
| DC live（MEM_SNAP） | 1 → **50** 線形増加 | min 1 / max **7** / last 6（収束） |
| Private | 381 → **7,740 MB** 線形増加 | 381 → **1,196 MB**（対前回 -6,544 MB） |
| EBR pend/ovf | 0 / 0 | 0 / 0（rec 46.3M 単調増加） |
| exit code | 0x00000000 | 0x00000000 |
| CTest | Debug 40/40 | **Debug 40/40 + Release 39/39**※1 |

※1 Release AudioEngineHarness は HEAD baseline（362dccd・本修正無し）でも 0xC0000374 で 3/3 再現する
pre-existing クラッシュ（D162-1P §0 の既知 Release DIAG 起動クラッシュと同一 signature）のため除外。

## 1. 変更ファイル（全変更の内訳）

| ファイル | 変更内容 |
| --- | --- |
| `src/audioengine/RuntimePublicationOrchestrator.h` | `retireRegisteredDSP(req, origin)` 公開宣言（registered DSP の terminal disposition authority）＋ `DeferredPublishView::peekRequestCopy()` 追加 |
| `src/audioengine/RuntimePublicationOrchestrator.cpp` | helper 実装（DSPLifetimeManager include 追加）＋ S1/S2/S4/dormant 修正 ＋ S3（無効化・D162-2-C 注記付き） |
| `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` | VerifyDrained で最終 active/fading DSP の resolve を追加し破壊 block を world clear 後に配置（**destroy 呼び出しは無効化**・D162-2-C 注記付き） |
| `src/audioengine/AudioEngine.Threading.cpp` | `quarantineSlot` を authority 経由（DSPLifetimeManager::retire）に変更 |
| `src/audioengine/AudioEngine.h` | `retireDSPHandleForRuntime` の misleading comment 修正（registry disposition primitive であることを明記・使用規則 INV-D162-1/3 を記載） |
| `src/convolver/ConvolverProcessor.Lifecycle.cpp` ほか | 変更なし（D162-1R-B 計装のみ・本件では触れない） |

## 2. 各 disposition site の before/after

### S1: deferred overwrite（H1 実測 48 件の主因）

- **before**: `RuntimePublicationOrchestrator.cpp` enqueueDeferred 内
  `engine_.retireDSPHandleForRuntime(oldDSP)` — 台帳解除のみで EBR enqueue なし → orphan。
- **after**: `retireRegisteredDSP(deferredSlot_->request, "deferred-overwrite")`
  → `DSPLifetimeManager::retire`（台帳解除 + EBR enqueue の 1 callee 実行）。
  D132 (INV-DEFERRED-2) の意味は不変（旧 deferred holder の retire）。slot 置換（:513 相当）前に実行し、
  同一 DSP への後続 finishView/discard は map 不在で no-op（二重 retire 構造的に不可）。

### S2: deferred discard（TTL/generation/sequence 経由）

- **before**: `processDeferredAdmission` 内 `view->discard(reason)` — slot 解放のみで DSP 処分なし。
- **after**: discard 前に `view->peekRequestCopy()` で request を値コピー →
  `retireRegisteredDSP(snapshot, "deferred-discard")` → `view->discard(reason)`。
  ownership 遷移: deferredSlot（owner）→ EBR → destroy。retire と discard の間で ownership は
  EBR が保持し、宙に浮かない。Single Thread Owner（RebuildThread）契約内の順序。

### S4: admission Rejected*（latent path repair）

- **before**: `submitPublishRequest` switch の RejectedStaleGeneration / RejectedNotFinalized /
  RejectedPressure / RejectedShutdown — telemetry のみで newDSP 処分なし。
- **after**: 各 case に `retireRegisteredDSP(req, "rejected-…")` を追加。
  - stale generation / not finalized / pressure / shutdown: registered・未 publish・World 到達不能 →
    authority retire（world-build-fail の `lifetime_.retire` 前例 Orchestrator.cpp:181/:249 と同処置）。
  - **RejectedPublishFailure は disposition しない**（指示 #6 の確認事項への回答）:
    trySubmitImpl 内で `destroyRolledBackDSP`（direct destroy）が完了しており、registry も rollback
    （Constructing→Reclaimed）済みのため resolve が nullptr を返す。retire を追加すると二重破壊に
    なり得るため明示的に触れない（INV-D162-3）。

### dormant RetryExhaustedDiscard（latent path repair・明記どおり分離）

- **before**: `engine_.retireDSPHandleForRuntime(dsp)`（台帳解除のみ）。
- **after**: `retireRegisteredDSP(req, "retry-exhausted-discard")`。
  H1 実測 49 件には不寄与（F6-6 どおり不発）のため「latent path repair」として S1-S4 と分離して計上。

### S3: shutdown clear（clearDeferredForShutdown）— **D162-2-C へ繰り越し**

- 初版: authority retire（EBR enqueue）。→ 実測: 破壊した場合のみ ~AudioEngine 後半の member teardown で
  `0xC0000005`（mkl_free 不正ポインタ・クラッシュダンプ解析: `mkl_serv_check_fast_memory_size` が
  address 0xFFFFFFFFFFFFFFFF を READ）。
- 第2版: T2 direct destroy（未 publish = RT 到達不能のため DSPGuard 同契約が正当）。→ 同じく exit AV。
- world clear 順序入れ替え（world 先に解放 → DSP 破壊）でも AV 再現。
- **分離試験で確定**: S3 破壊を無効化すると exit 0x00000000（3 回連続）。
- 結論: shutdown teardown 順序と DSPHandle registry / EQCacheManager / rcuSwapper の相互作用が
  原因であり S3 単独の修正範囲を超える。**現行挙動（slot 破棄のみ・D162-1P と同一）に戻し、
  D162-2-C で teardown 順序監査を行った後に有効化**。コードには無効化理由を含む注記を残した。

### V-D: VerifyDrained での最終 active/fading DSP 破壊 — **同様に D162-2-C へ繰り越し**

- resolve（state Active 中）+ 既存 slot 遷移（retire + tryShutdownQuiescentReclaim）は維持し、
  destroy のみ `if (false && …)` で無効化（注記付き）。EBR / direct / world clear 後 direct の
  3 バリアントすべてで exit AV を実測確認した。

### quarantine（Threading.cpp quarantineSlot）— **authority 統一を実装**

- 意味論確定（指示 #8）: (a) resolve は Quarantined state を拒否 → quarantine 後は誰も DSPCore に
  到達できない、(b) DSPQuarantineManager は metadata（audit log）のみ保持で DSPCore* を保持しない、
  (c) Recovery は buildSource から再 build し quarantined DSPCore を参照しない、(d) quarantine 解放
  （destroyQuarantineSlot）は slot 状態遷移のみ → **orphan 確定（意図的 resident ownership ではない）**。
- **after**: `DSPLifetimeManager lifetimeMgr(*this); lifetimeMgr.retire(dsp)` に変更。
  EBR の epoch gate が破壊タイミングを安全側に保証する。soak 実測では quarantine 不発のため
  動作確認は CTest 依存（全 PASS）。

## 3. Ownership transition と thread contract

| site | thread | ownership 遷移 |
| --- | --- | --- |
| S1 overwrite | RebuildThread | deferredSlot → DSPLifetimeManager::retire → EBR → destroyDSPCoreNode |
| S2 discard | RebuildThread（jassert 済） | 同上（retire と discard の間は EBR が単一 owner） |
| S4 Rejected | RebuildThread | 同上 |
| dormant | RebuildThread | 同上 |
| S3 clear | Message/RebuildThread（assert-free 契約） | 実装済み無効（D162-2-C） |
| quarantine | CoordinatorLoop | registry → EBR |
| V-D | Message Thread | 実装済み無効（D162-2-C） |

- `retireRegisteredDSP` は null-safe / 未登録 no-op / rollback・reclaimed no-op を保証し、
  RT 経路（B-I3 RT boundary）には一切触れない。
- AudioEngine.h の comment 修正（指示 #9）: 「retireDSPHandleForRuntime は handle registry disposition の
  primitive であり物理破壊は行わない。物理破壊までの完全 terminal disposition は
  DSPLifetimeManager::retire」を明記し、使用規則（registered→authority / unregistered→DSPGuard /
  rollback→destroyRolledBackDSP）を記載。API rename は行っていない。

## 4. Invariant への対応

| invariant | 実装での充足 |
| --- | --- |
| INV-D162-1 単一 terminal path | S1/S2/S4/dormant/quarantine を `DSPLifetimeManager::retire`（T1）に統一。T2（direct destroy）は未登録 DSP の DSPGuard 契約のみ |
| INV-D162-2 eventual destruction | 実行中 generation の orphan は解消（49→最大 4、§5 残課題あり） |
| INV-D162-3 二重化禁止 | helper は resolve nullptr（rollback/reclaimed）で no-op。S4 は RejectedPublishFailure で destroy を追加しない。S1 は slot 置換前に map erase が完了し以後 no-op |
| INV-D162-4 RT 不変 | 全変更は NonRT path。Audio Thread の API 触れず |
| INV-D162-5 published 破壊の epoch-safe | S1-S4 は全て未 publish DSP が対象（published old DSP の既存 EBR 経路は不変） |

## 5. Residual Risk（次工程 D162-2-C への引き継ぎ）

1. **S3 shutdown-clear deferred DSP（1 件/run）**: disposition 未実装（現行 D162-1P と同一挙動）。
   破壊実装は exit AV（0xC0000005・mkl_free 不正ポインタ）を誘発することが分離試験で確定済み。
   process exit で OS 回収されるため長時間運転での蓄積はないが、INV-D162-2 の技術的例外。
2. **V-D 最終 active/fading DSP（1 件/run）**: 同上。published World topology / EQCacheManager /
   rcuSwapper との teardown 相互作用の root-cause が未確定。クラッシュダンプ
   （CrashDumps/ConvoPeq.exe.448.dmp 等）を保存済み。
3. **60 gen soak の残存 retained 4 件**（gen 9/35/41/53）: S1/S2 の網羅性に関する residual
   （deferred admission の consume→再 submit 経路など、discard/overwrite のどちらにも到達しない
   窓口の存在が示唆される）。D162-2-C で 1 件ずつ遷移を追跡する必要がある。
4. **quarantine 経路の動作実績なし**: soak で quarantine 不発のため、authority 接続は CTest 依存の確認。
5. DSPGuard の既存二重解放 CAVEAT（RebuildDispatch.cpp:938-965）: 本修正では触れていないが、
   D162-2-C の teardown 監査で二重破壊の静的再検証を推奨。

## 6. Regression Assessment（要約・詳細は evidence）

| 領域 | 影響評価 | 根拠 |
| --- | --- | --- |
| D132 (INV-DEFERRED-2 overwrite retire) | 意味論不変・経路を authority に差し替え | S1 before/after（§2）|
| D135-8 / F6（deferred accounting・watchdog） | 不変 — retention≠retry の会計・kMax dormant guard・recovery wake は未改変 | enqueueDeferred の accounting block は本修正で触れず（dormant 分岐内の処分のみ） |
| D137 G-4.4-P2（redrive wake） | 不変 — consumeRedriveWake / recoveryPending 経路は未改変 | RuntimePublicationOrchestrator.cpp の該当 block 無変更 |
| D138 P3（fade-complete wake） | 不変 — Timer.cpp 側 fade-complete 経路は未改変 | 変更ファイルリスト外 |
| D105 recovery accounting（R5-9/R18/R20） | 不変 — resolveIfRecovery / markTransientFailure / rearm の呼び出しは全 case で元のまま（retire 追加は obligation 会計と独立） | submitPublishRequest switch の diff |
| shutdown drain | S3/V-D 無効化により D162-1P と同一挙動に回帰。waitForDrain/isFullyDrained 契約は不変 | 60 gen soak exit 0x0・pendingRetire=0 |

## 7. Acceptance Criterion の達成評価

主 acceptance criterion（指示）: 「registered DSP の terminal ownership が必ず
`DSPLifetimeManager::retire → enqueueWithRetry` に到達する構造になったこと」。

- **構造としては成立**: S1/S2/S4/dormant/quarantine の全 disposition が authority 経由に統一され、
  非 DIAG ビルドを含む CTest 40+39 で回帰なし。
- **ただし 2 点の意図的例外**（S3・V-D）が残存 — shutdown 時破壊は teardown 順序の root-cause 未確定のため
  D162-1P と同一挙動へ退避した。これを「修正未完了」と評価するのが正確であり、
  **完全達成には D162-2-C（shutdown teardown 順序監査を含む read-only validation）が必須**。
- 実効性: retained DSP は 49/60 → 6/59（-88%）、Private 増加は 7,740 MB → 1,196 MB（-85%）。
  H1 の主因（S1 deferred overwrite 48 件）は完全に解消。
