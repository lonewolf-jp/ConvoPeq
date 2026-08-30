# D135-8/9 Gate C-F4 — Retention Wake / TTL Contract Design Review

Date: 2026-08-30
Scope: **production source 0変更・test source 0変更・実装 0・テスト変更 0**。最新 working tree（ConvoPeq.md 2026-08-30 版と同一ソース）を基準。F1/F2/F3 の結論を前提とし、矛盾は最新ソース優先で訂正した。

## Verdict

> F3 推奨案（Option A + event wake + watchdog + obligation-keyed TTL）は**実装可能**だが、
> F4-1 の検証により **obligation identity は `generation` 単独では不十分**と確定した
> （recovery が同一 generation を再利用し別 payload で publish する、RebuildDispatch:1035-1040）。
> 最終契約: **identity = (generation, recoveryObligationId)**、**TTL = obligation.createdAt 起点の dwell**、
> **wake = fade-complete（主）+ watchdog（自己修復のみ）**、**retry accounting は retention を数えない**。
> F4 合格条件 14/14 充足（下表）。実装境界は §10 の 必須5点 / 別パッチ3点 に確定。

---

## F4-1. generation は obligation identity として十分か → **不十分（要訂正）**

`PublishRequest`（PublicationAdmission.h:19-27）:
`{ newDSP, generation, sealedSnapshot, buildAnalysis, oversamplingResult, buildDiagnostics, recoveryObligationId }`

実コード追跡結果:

| 経路 | generation 採番 | payload |
| --- | --- | --- |
| 通常 rebuild | `++rebuildRequestGeneration`（RebuildDispatch:655）— 要求ごと新規 | task 由来 |
| **recovery publish** | **`consumeAtomic(rebuildRequestGeneration)`（:1035-1036）— インクリメントせず現在値を再利用** | `recovery->buildSource`（別 DSP・別 snapshot・`recoveryObligationId≠0`、:1040/:1120） |

→ **同一 generation G が「通常 obligation（obligationId=0）」と「recovery obligation（obligationId≠0）」の
両ペイロードを持ち得る**。`sameObligation = (req.generation == deferredRetryGeneration_)`（Orch:472）は
この2者を区別できない（現行 retry 会計にも潜在的同型問題がある。ただし recovery wake は reset 経路なので
実害は限定的だった）。

**確定契約**:
```text
obligation identity := (generation, recoveryObligationId)
```
timestamp 引き継ぎ（F4-2）と retry 会計キーの両方にこの tuple を使う。

## F4-2. TTL の判定対象 → obligation dwell に確定

```text
obligation.createdAtUs = 当該 (gen, obligationId) が初めて DeferredFadingActive で enqueue された時刻
obligation.expiresAt   = createdAtUs + kDeferredPublishTTLUs (30s)
re-drive ≠ new obligation（同一 tuple の再 enqueue は createdAtUs を引き継ぐ）
```

**実装上の必須注意（F4 で新規確定）**: re-drive 経路は `consume()`（finishView で slot reset）の**後に**
`submitPublishRequest → enqueueDeferred` が走るため、**再 enqueue 時点で `deferredSlot_` は空**
（Orch:653-655 → :377-378）。旧 slot から timestamp を読むことは**できない**。
→ 引き継ぎは **obligation-keyed メンバ**（`deferredObligationCreatedAtUs` + key=(gen, obligationId)、
既存 `deferredRetryGeneration_/Count_` と同一パターン・同一所有権（RebuildThread single owner））で実装する。

命名（`enqueueTimestampUs` → `obligationCreatedAtUs` 等）は**本パッチでは変更しない**（F4-10 別パッチ #8 扱い）。
意味論だけ差し替える（`DeferredPublishMetadata.enqueueTimestampUs` に obligation createdAt を載せる）。

## F4-3. fade-complete wake の producer/consumer chain（証明）

```text
AudioThread: fade 完了（crossfadeRuntime 内部状態）
  ↓ （次の MessageTimer）
MessageThread: AudioEngine::timerCallback()（Timer.cpp:425-1586）
  ├ m_coordinator.tryCompleteFade() → true（:928）
  ├ notifyFadeComplete / endCrossfade / fadingRuntimeDSPSlot CAS クリア / submitObserve（retire intent）
  ├ crossfadeRuntime_.complete() / refresh snapshot / idle publish（commitRuntimePublication, :994）
  └ 【追加箇所: ブロック末尾 :1000 付近】
      if (runtimeOrchestrator_ && runtimeOrchestrator_->hasDeferredRequest()) {
          { std::lock_guard<std::mutex> lock(rebuildMutex); publishRetryReady = true; }
          rebuildCV.notify_one();
      }
  ↓
RebuildThread: rebuildCV wake（predicate: hasPendingTask || publishRetryReady || recoveryPending || shouldExit）
  → doDeferredPublish=true → processDeferredAdmission(wasRecoveryWake=false)
```

**lock ordering 証明**: timerCallback 本体（425-1586）は **いかなる mutex も保持しない**（grep 0件）。
`rebuildMutex` はリーフレール（保持中に他 lock を取らない）で、recovery handoff（Timer.cpp:1746-1751、
`onHealthEvent` 内）が**同一パターン**（lock{flag} → notify はロック外）を本番で使用済み。
fade-complete wake はこれと同型・同リーフレール。**新規 authority は生まれない**（wake は再評価依頼であり
publish/crossfade/retire の決定権は不変）。

**同時発生ケース（fade-complete × deferred clear）**: fade-complete が `publishRetryReady=true` を立てた直後に
`requestDeferredClear()`（Step 9 latch）または shutdown clear が入っても、rebuild 側の順序は
`drainDeferredClearIfRequested()`（RebuildDispatch:905-906）→ `processDeferredAdmission`（:914-921）で、
clear 後なら `peekDeferred` が `hasDeferred_==false` で nullopt → no-op。**二重解放・二重 publish なし**。

## F4-4. lost-wake / duplicate wake の安全性（証明）

- `publishRetryReady` は rebuildMutex 保護の plain bool、consumer が read-and-clear（RebuildDispatch:889-890）。
  fade-complete と watchdog が同時に true を立てても**単一の true に収束**し、wake が2回来ても2回目は
  `doDeferredPublish=false` で `processDeferredAdmission` を呼ばない。仮に呼んでも `peekDeferred` が
  `hasDeferred_` を atomic で見て nullopt → **二重 consume 不能**（consume は View 経由のみ、View は
  slot 借用 + state_ ガード + jassert）。
- `processDeferredAdmission` は RebuildThread 専用（jassert、Orch:675）→ 直列化保証。
- **recovery provenance 混入なし**: fade-complete / watchdog は `publishRetryReady` のみを設定し
  `recoveryRetryReady`（別 atomic、Orch 側消費は RebuildDispatch:888 の exchange）には触れない。
  → `wasRecoveryWake=false` が保証され、budget reset が誤発火しない。
- 正しさの根拠は CV 契約そのもの: producer は predicate 更新を lock 内で行うため、consumer の
  `wait(lock, pred)` は lost-wake しない（watchdog は「CV 使用ミスの保険」であり、通常は不要なはず）。

## F4-5. watchdog の位置づけ（周期は固定しない）

| 要素 | 役割 |
| --- | --- |
| fade-complete wake | **正常経路**（条件成立の即時再評価、遅延 ≈ 0） |
| watchdog | **自己修復専用**（event 欠落時のみ発火。正常系代替ではない） |
| 30s TTL（obligation-keyed 化後） | lifetime bound（dwell 上限） |
| shutdown clear | hard terminal |
| generation/sequence supersession | 新要求時の終端 |

**定義**: watchdog は「lost-wake recovery」であり「正常な fade completion の代替」ではない — この定義は
§F4-3/4 の chain 証明から成立する（正常時は fade-complete が必ず先に発火し、slot が空になれば watchdog は
`hasDeferredRequest()==false` で no-op）。
**周期**: 仕様固定しない。導出根拠は「watchdog 遅延 = 通知欠落時の追加遅延のみ」。正常 UX は event wake が
保証するため、watchdog は coordinator tick（1ms）の十分大きな倍数（例: 数十 ms〜数百 ms 級）でよく、
`kPublishReceiptWaitTimeoutMs=250`（AudioEngine.h:4686）と同オーダーの定数として実装時に named constant 化。
**本 F4 では数値を確定しない**（ユーザー指示遵守）。

## F4-6. retry accounting 最終状態表

| イベント | 現行 | **最終契約** |
| --- | --- | --- |
| ordinary wake（publishRetryReady）→ Deferred 再評価 | count++ | **count 不変**（retention） |
| recovery wake（recoveryRetryReady） | reset（peek 前） | **reset 維持**（D135-8 P3、変更不要） |
| Ready → Accepted → publish | — | Success 終端（count 無関係） |
| Ready → RejectedPressure | 要求消滅（re-enqueue なし） | 終端（§F4-7。retry 対象にしない） |
| Ready → RejectedPublishFailure | 要求消滅 | 終端（同上） |
| 将来の Type A retry（publish 失敗の再試行） | 経路なし | **count++ の唯一の対象**（導入時） |

**`kMaxDeferredRetries` の帰趨**: retention を除外すると、**現状 production に Type A 経路が存在しないため
cap は dormant 化する**。決定: **dormant のまま保持**（削除しない）し、契約コメントで
「retention を数えない / Type A 導入時にのみ有効 / 現行の churn bound は watchdog+TTL が担う」を明記する。
Rejected* を retry に接続するかどうかは §F4-7 の別パッチ（#7）で扱う（本パッチに混ぜない）。

## F4-7. terminal transition 完成表（Rejected* の明文化）

```text
Deferred
 ├─ Ready → Accepted → Success（publish commit、receipt）
 ├─ Ready → RejectedPressure → 終端（現行: 要求消滅。telemetry のみ）
 ├─ Ready → RejectedPublishFailure → 終端（現行: 要求消滅。FailureStage::Execution）
 ├─ Ready → RejectedStaleGeneration / RejectedNotFinalized → 終端（同上系）
 ├─ Ready → RejectedShutdown → 終端（shutdown telemetry）
 ├─ TTL 超過 → StaleDiscard（※Expired enum は未使用 — 別パッチ #6）
 ├─ generation mismatch → StaleDiscard
 ├─ sequence 後退 → StaleDiscard
 ├─ shutdown → ShutdownDiscard（clearDeferredForShutdown）
 ├─ 新要求 enqueue → SupersededDiscard（旧 slot DSP を明示 retire、Orch:460-468）
 └─ RetryExhaustedDiscard → **retention では到達不能化**（Type A 専用として dormant 維持）
```

**Rejected* の ownership 注記（重要・要 follow-up）**: Ready 経路で consume された req の `newDSP` は、
Rejected* 分岐（Orch:403-432）では **retire されないまま req が破棄される**（handle は map に残り、
DSPCore は shutdown まで生存する可能性）。これは本修正の中心問題ではないが、terminal 契約として
「Rejected* は DSP を retire しない」ことを明記し、**別パッチ #7** で
(a) retire 付き終端、または (b) re-enqueue + retry 会計接続、のいずれかに統一する（現状の放置は
D117 系の滞留パターンと同型である点を記録）。

## F4-8. DeferredHealth telemetry の意味論

| field | 現行（churn 時） | 新設計（retention + event wake） | 判定 |
| --- | --- | --- | --- |
| `deferredCount` | enqueue 毎 1 / finishView 毎 0（1ms 周期で振動） | 保持中は記録頻度が tick 依存 → event 依存に激減。意味（slot 占有有無）は維持 | ✅（レート変化のみ） |
| `oldestDeferredAgeMs` | **常に 0**（enqueue で 0 固定・他所で計算されず — dead field、TelemetryRecorder.h:124） | obligation-keyed createdAt があれば `now - createdAt` で**真の dwell を表現可能**。ただし計算追加は telemetry schema 変更 → **別パッチ #8** | ⚠ 本パッチでは 0 のまま（意味は改善方向に使用可能になる） |
| `overwriteCount` | re-drive では非増加（consume 済みのため）— 変更なし | 変更なし | ✅ |
| `lastDiscardReason` / `TimestampUs` | finishView が記録 | 変更なし（RetryExhaustedDiscard は dormant 化、Stale/Superseded/Shutdown は維持） | ✅ |

## F4-9. 最終 state machine（1枚）

```text
                     PublishRequest (gen G, obligationId O)
                                │
                     submitPublishRequest → 主 admission evaluate
                                │
                    ┌───────────┴────────────┐
                    │ DeferredFadingActive    │ Accepted
                    ▼                         ▼
        [RETAIN] enqueueDeferred          publish → Success
          key=(G,O) 一致 → createdAt 維持   （receipt 完了）
          key 相違 → 新 obligation（旧は SupersededDiscard+DSP retire）
                    │
        ┌───────────┼────────────────┬───────────────────┐
        ▼           ▼                ▼                   ▼
  fade-complete  watchdog        新 generation 採番   shutdown
  wake (P-new)   wake (P1 間隔化)  / sequence 後退    clearDeferredForShutdown
        │           │                │                   │
        └─────┬─────┘                ▼                   ▼
              ▼                evaluateDeferred     ShutdownDiscard
      processDeferredAdmission  → StaleDiscard      （slot reset）
        （RebuildThread 専用）        │
              │                      ▼
     evaluateDeferred        finishView（slot reset,
      ├ TTL(30s dwell)→ StaleDiscard  hasDeferred_=false, telemetry）
      ├ gen mismatch → StaleDiscard
      ├ seq 後退     → StaleDiscard
      └ Ready → consume → submitPublishRequest
                ├ Accepted → Success
                ├ Rejected* → 終端（DSP retire なし — §F4-7 follow-up #7）
                └ DeferredFadingActive → [RETAIN]（ループ、count 不変）

  ※ RetryExhaustedDiscard: retention では到達不能（Type A 専用・dormant）
  ※ recovery wake: processDeferredAdmission 先頭で budget reset（P3 維持）
```

## F4-10. 実装パッチ境界（確定）

**必須（本パッチ）**
1. retention accounting 分離（identity=(gen, obligationId)。Orch:377/470-503）
2. obligation timestamp 保存（obligation-keyed メンバ `deferredObligationCreatedAtUs`。Orch:445/505-521 + evaluateDeferred 入力）
3. fade-complete wake（Timer.cpp fadeCompleted 末尾、recovery と同型 handoff）
4. watchdog 化（Threading.cpp:280-289 の毎 tick 再通知 → named constant 間隔。数値は実装時確定）
5. 契約コメント/文書統一（Orch.h:283-289、State.h:16-20、D135-1 文書。`>` 意味論 + retention 除外 + dwell TTL）

**別パッチ（本パッチに混ぜない）**
6. `Expired` enum 活性化（TTL 終端の分離）
7. `Rejected*` の terminal semantics 再設計（DSP retire or retry 接続）
8. telemetry schema 変更（`oldestDeferredAgeMs` の dwell 化、メンバ改名）

## F4 合格条件チェックリスト

- [x] generation が obligation identity として十分である → **不十分と判明し (gen, obligationId) に拡張**（F4-1）
- [x] obligation timestamp の定義が確定（初回 DeferredFadingActive 時刻、re-drive で不変）
- [x] re-enqueue で timestamp が更新されない（obligation-keyed メンバ方式。slot 経由は不可と確定）
- [x] fade-complete wake の producer/consumer が証明済み（同型 handoff・リーフレール・thread 確定）
- [x] lost-wake 時の watchdog recovery が証明済み（CV 契約 + watchdog 保険）
- [x] duplicate wake が安全（bool 収束 + peek nullopt + RebuildThread 直列 + provenance 非混入）
- [x] watchdog は正常経路ではなく自己修復経路（§F4-5 定義成立）
- [x] retry / retention の意味が完全分離（§F4-6 表）
- [x] recovery wake semantics が維持される（reset 経路不変・provenance 独立）
- [x] Rejected* の terminal semantics が明文化（§F4-7。DSP retire なしの問題は follow-up #7 に分離）
- [x] DeferredHealth の意味が維持される（§F4-8。oldestDeferredAgeMs は dead field のまま＝本パッチ範囲外）
- [x] TTL が真の dwell bound になる（obligation-keyed 化により。F3 の失効問題の解消）
- [x] 変更範囲が今回の修正に限定される（§F4-10 必須5点のみ）
- [x] I4 ownership / authority invariants に違反しない（wake=依頼のみ、決定権不変、retire 経路不変、単一 slot 維持、CV predicate 不変）

## F3 からの訂正点（明示）

- F3 §11-4「同一 generation 再 enqueue 時に timestamp を引き継ぐ」→ **条件を (generation, recoveryObligationId) に強化**（F4-1）。
- F3 §11「slot から引き継ぎ」示唆 → **slot は consume 済みのため読めない。obligation-keyed メンバ必須**（F4-2）。
- F3 §10 watchdog 例「100ms」→ **数値確定を撤回し named constant として実装時決定**（F4-5）。
