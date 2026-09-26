# P1-5-IR-P2 — Step 5-AH / P3-5-R26: Publish Intent Origin / Window-Level Disambiguation Design Audit

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（P3-5-R26）
- **種別**: read-only design audit。**実装 0・build 0・run 0**。
- **目的**: R25-B（Coordinator take は共通 consumer・起源を保持しない）を前提に、
  起源を壊さず観測可能性を得る最小設計を再監査する。Coordinator counter を無理に作らない。
- **結論**: **R26-A（path A）**。最小 origin plumbing は source 上定義可能（§4）。
  既存観測のみでの C-1/C-2 十分区別は不能（§5）。次は別の実装 gate へ（§7）。

---

## 1. State Freeze（R25-B 直後・PASS）

```text
HEAD 1e9e63e34bed7adb9342ebc81259ded9689fc48a／kP15FullMatrix true／OFF
R16 take＋R19 B2＋R22 commit＋R10 accessor＋T6/T9＋R23 delta観測 保持
R24 design（Step5AF）＋R25-B finding（Step5AG）保持
ConvoPeq.md fresh（2026-09-23 21:32:56・5,521,528 B・最新src編集 20:40 より後・R25-B時と同一）
working tree = R23 vehicle（src差分： AudioEngine.h +78・RebuildDispatch.cpp +14・
  P1 TU +594 他・R25-B による差分 0・coordinatorTakeCount_ 出現数 0 のまま）
```

- R25-B 後に production／test とも変更なし。本 Step でも変更しない（§7 禁止遵守）。

## 2. R25-B finding（前提・解決済み設計事実として固定）

```text
Coordinator take = intentQueue_.pop()（ProcessIntent.cpp:47・Publish型・一意）
coord++ すると normal rebuild＋Recovery publish＋deferred re-enqueue を混在計数
  （ISRRuntimePublicationCoordinator.h:785-786 の三経路集約の自己文書化）
intent payload に起源 marker なし（AudioEngine.h:4872 recoveryObligationId=0 固定・
  Orchestrator.cpp:321 で同期解決済み・generation linkage 禁止）
→ main-site-only の 1-writer は構成不能 → cmt→coord→seq の3軸は作れない
```

- 本 Step では上記を再論証しない。起点として固定する。
- ただし §4 の精密化として：三経路のうち「deferred re-enqueue」は起源ではなく位相である。
  deferredSlot_ は req 全体を格納し（Orchestrator.cpp:548・`deferredSlot_->request`）、
  resubmit は `view->consume()` で元 req を move-out して `submitPublishRequest(req)` する
  （Orchestrator.cpp:908-910）。したがって deferral 前後で `recoveryObligationId` は保存される。
  **起源 taxonomy は2値（Main／Recovery）で十分**であり、deferred は直交する遅延位相である。
  この整理が Option A の変更範囲を縮小する（§4）。

## 3. Existing observability inventory（writer／reader／lifetime semantics）

### 3.1 Test-readable today（新規 getter なしで読めるもの）

| 観測 | writer | reader（test） | lifetime／意味 |
| --- | --- | --- | --- |
| `cmt`（rebuildCommitEnqueueCount_） | RebuildDispatch.cpp:1420（main-site caller のみ） | getRebuildCommitEnqueueCount（AudioEngine.h:2084） | cumulative・main-site commit-enqueue 到達数 |
| `seq`（lastCommittedPublicationSequence_） | Commit.cpp:402（commit 成功時） | getLastCommittedPublicationSequence（AudioEngine.h:1744） | cumulative（単調増）・全起源の publish 到達数 |
| `drp`（lastDroppedGeneration） | Commit.cpp:205／211（commit 側 monotonicity reject 時のみ） | getRuntimeLifecycleDiagnostics（AudioEngine.h:1732） | last-value・**commit 側**の generation／sequence 非単調 reject のみ記録 |
| `take/bld`（R16/R19） | :911／:1235 | 既存 getter | cumulative・本 Step の対象外だが window 文脈として保持 |
| `req/que/dup`（dispatch diagnostics） | rtAuxMutable_ 各 counter | getRebuildDispatchDiagnostics（AudioEngine.h:2057） | cumulative・build-stage 上流 |
| `blo`（getPublicationBacklogCount） | **production writer なし**（Coordinator.cpp:269-271 は TEST-ONLY setter・`⚠️ TEST-ONLY` 明記） | AudioEngine.h:3713 passthrough | production では恒常 0。R23 で blo=0 全点と整合。**観測として dead** |
| `publishCount`（lifecycle.publishCount） | AudioEngine.h:3639（reserveNextRuntimeGraphGeneration 内） | 同 diagnostics | **graph-generation 予約数であり publish 数ではない**。誤用禁止（本監査で確定） |
| `lastCommittedGeneration` | Commit.cpp:401 | 同 diagnostics | commit 成功世代・seq と対 |
| retire／reclaimCount | 各 retire／reclaim 経路 | 同 diagnostics | publish 経路の直接観測ではない |

### 3.2 Exists but NOT test-readable（+1 R10-class getter で読めるもの）

| 観測 | writer／reader（production） | lifetime／意味 |
| --- | --- | --- |
| `publicationIntentResidencyCount_` | +1：enqueuePublicationIntent push前 reservation（Coordinator.h:790）／−1：push失敗 rollback（:793）・pop Publish時（ProcessIntent.cpp:56）。reader：getPublicationIntentResidencyCount（Coordinator.cpp:435・**AudioEngine passthrough なし**） | **gauge（residency）**＝queue 内 Publish 数＋producer reservation。cumulative ではない（§5-§6 の核心） |
| `hasDeferredRequest` | writer：enqueueDeferred（:576）／finishView（:775）他。reader：Orchestrator.h:167（**AudioEngine passthrough なし**） | boolean latch・deferred 滞留の有無 |
| `pendingIntentCount_` | Observe／Quarantine／Recovery の transport residency（Coordinator.h:967-979・**Publish は非対象**と明記） | Publish 追跡に無関係 |
| admission Decision／telemetry failures | telemetryRecorder_.recordFailure（Orchestrator.cpp 各 reject／fail 分岐） | **test reader なし**。admission 却下は test から不可視 |

### 3.3 Residency gauge の精密意味（R26 §6 の要求事項）

```text
enqueue（reservation +1 → push 成功で維持 → pop Publish で -1）
```

- `pre=0／post=0` は「何もなし」と「window 内 enqueue＋pop 完了」を区別しない。
- `post>0` は「≥1 Publish が in-flight（Coordinator 未 take）」の証拠になるが、
  起源は不明（recovery の滞留も同一値に載る）。
- queue 容量は 4096（Coordinator.h:1159 `kIntentQueueCapacity`）。
  full 時の Publish は `enqueuePublicationIntent==false` → owner reclaim → publishImpl FAILED
  → RejectedPublishFailure 終端（Orchestrator.cpp:282-315）。**queue-full 終端にも test 可視 counter なし**。

## 4. Option A — origin marker plumbing（必要変更範囲の列挙・実装しない）

### 4.1 変更点（bounded・5点）

```text
A1. PublicationExecutor::publish／publishImpl（PublicationExecutor.cpp:8-60）
    signature ＋1（origin／obligationId 伝搬用）。呼出しは trySubmitImpl の1箇所。
A2. AudioEngine::commitRuntimePublication（AudioEngine.h:4900） signature ＋1。
A3. AudioEngine::enqueueRuntimePublicationFireAndForget（AudioEngine.h:4854 付近） signature ＋1。
    intent.payload.publish.recoveryObligationId = 搬送値（現行 :4872 の 0 固定を置換）。
A4. processIntent pop 側の分岐（ProcessIntent.cpp:53-66）
    type==Publish かつ origin==Main の場合のみ coord(Main)++（1 writer）。
    Recovery take は別途数えるか数えないかは実装 gate の設計選択（本 Step では決めない）。
A5. test 側 emitQDelta に coord field（R25 §6 予定どおり・新規 logger なし）。
```

### 4.2 変更不要なもの（source 確認済み）

```text
- Intent／PublishPayload struct 変更なし（recoveryObligationId field は既存・Coordinator.h:732）。
  trivially-copyable／standard-layout の static_assert（:755-758）維持。
- OwnerChannel 変更なし（key = seq／epoch／mappedGen のまま・OwnerChannel.h:31）。
- Deferred 経路変更なし（slot が元 req を保持するため deferral 前後で起源保存・§2）。
- enqueuePublicationIntent funnel 変更なし（単一 funnel・AudioEngine.h:4863 が唯一の production 生成点）。
- INV-X5-1（residency は全 Publish 計数）変更なし。coord(Main) と residency は別物として併存。
```

### 4.3 既存契約への影響（D105 completion authority・実装 gate への申送り）

```text
- onPublishCommitted(intent.sequenceId, intent.payload.publish.recoveryObligationId)
  （RuntimePublishExecutor.h:114）が recovery に対して非 no-op 化する。
  現行： async 側は id=0 → early-return（Coordinator.cpp:1033）し、
  sync 側（Orchestrator.cpp:321）が Live→terminal CAS の winner。
  変更後： receipt 待ち（commitRuntimePublication は receipt まで block）のため
  async 側が先に resolve を試み、sync 側は敗北する（両 call site とも `won` 戻り値を無視しているため
  終端状態 ResolvedSuccess 自体は同一だが winner が反転する）。
- 再監査必須項目（実装 gate の checklist とする・本 Step では結論しない）：
  D152 T3 の discardedPending 分類タイミング／postRecoveryFailureSignal との相互作用
  （D105-R20 中央集約の前提）／Retry 経路（resolveIfRecovery(Retry) は false 維持・:426-428 影響なし見込み）／
  shutdown-drain 順序／HANDLER-1（handler は新 field の read-only 参照のみ）。
- Route B コメント（AudioEngine.h:4872）および X5 §6.5 コメントの改訂が必要（文書変更）。
```

### 4.4 起源 taxonomy（§2 再掲・定義）

```text
origin = { Main（recoveryObligationId==0）, Recovery（!=0） } の2値で十分。
deferred re-enqueue は第3の起源ではなく遅延位相（slot が元 req を保持）。
per-task attribution（generation linkage）は不要。window-level の2値分類で足りる。
```

## 5. Option B — window-level disambiguation（既存観測のみの C-1/C-2 検証）

### 5.1 問いの厳密化

```text
Case C： cmtΔ>0 ／ seqΔ=0 に対して、
C-1（main commit → Coordinator 未到達）と C-2（Coordinator 到達後 → publish 前）を
追加 production instrumentation なしで区別できるか。
```

### 5.2 Indistinguishability set（cmtΔ>0／seqΔ=0／drpΔ=0 の window が取り得る実態）

```text
(i)   in-flight： OwnerChannel＋intentQueue_ に Publish 残留（Coordinator 未 take）。
(ii)  admission Deferred → deferredSlot_ 滞留（後日 resubmit or discard）。
(iii) admission Rejected* 終端： StaleGeneration／NotFinalized／Pressure／Shutdown
      （evaluate・Admission.cpp:12-48）。いずれも test 可視 counter なし。
(iv)  post-admission 終端： build 失敗／crossfade rebuild 失敗（→RejectedNotFinalized・
      Orchestrator.cpp:193／261）／executor publish 失敗（queue-full 等・→RejectedPublishFailure）。
(v)   commit 側 monotonicity reject（→drpΔ>0 の場合のみ分離可能・Commit.cpp:205／211）。
(vi)  window-boundary race（R23 §10）： post-read 後の trailing 解決は gap に落ちる。
```

- `drp` が分離できるのは (v) の commit 側 reject のみ。admission 側 (iii) の5 subtype は
  `drp` に触れないため、`drpΔ=0` は「admission 却下なし」を意味しない。
- `blo` は production writer なし（§3.1）のため寄与ゼロ。
- `seqΔ>0` は main-site publish を意味しない（R23 §10： seqΔ>cmtΔ の実測・recovery／idle 混在）。
- したがって **(i)〜(iv)(vi) は既存観測の pre／post 絶対値だけでは相互に区別不能**。
  特に C-1（≒(i) の main-site 分）と C-2（≒(ii)〜(iv) の main-site 分）の分離は不能。

### 5.3 Residency-reader 中間案の評価（+1 R10-class getter 仮定時）

```text
- post residency > 0 ⇒ ≥1 Publish in-flight（C-1 方向の証拠）。ただし起源不明のため
  recovery 滞留の可能性が排除できず、main-site C-1 の確定にはならない。
- post residency = 0 ＋ cmtΔ>0 ＋ seqΔ=0 ⇒ taken-and-stalled／terminal-reject／
  deferred のいずれかで indeterminate（gauge のため履歴復元不能）。
- window delta of gauge ≠ take count（§3.3）。pre／post 非ゼロ遷移は弱証拠に留まる。
→ residency reader は「in-flight 検出」を不能から部分可能へ上げるが、
   C-1／C-2 の十分区別には到達しない。Recovery 混入も未解決。
```

### 5.4 Option B 判定

```text
既存観測のみでの C-1／C-2 十分区別は不能（§5.2 の set が分離不能・§5.3 でも不足）。
```

## 6. Minimality comparison（事実ベース・総合優劣の判定はしない）

| 軸 | A: origin plumbing（§4） | B: existing-observation-only（§5） |
| --- | --- | --- |
| 変更量 | production 4 signature＋1 intent 代入＋1 pop 分岐＋test 1 field。struct／OwnerChannel／deferred 変更なし | 0 |
| 観測能力 | coord(Main) の cumulative 計数により C-1／C-2 を window-level で分離可能（recovery take を除外）。R26 の問いに直接回答 | C-1／C-2 不可分（§5.2）。residency-reader 追加時も in-flight 部分検出に留まる（§5.3） |
| 既存契約への影響 | D105 completion authority の winner 反転（async 先勝）＋関連コメント改訂。再監査は実装 gate の checklist（§4.3）。HANDLER-1／INV-X5-1／transport 制約は維持 | なし |
| per-task attribution の必要性 | 不要（2値 origin の window 分類・generation linkage なし・R15 制限維持） | 不要（そもそも個体追跡しない）が、その代償として不可分 |
| Recovery 混入 | 構造的に除外（origin==Main フィルタ）。recovery take の扱いは実装 gate の選択肢 | 未解決（seq 混在は R23 §10 で実証済み・residency 混在も同様） |

## 7. Reopen condition（実装 gate への申送り・本 Step では着手しない）

```text
1. D105 再監査（§4.3 の checklist）： winner 反転の安全性／T3 分類タイミング／
   failure-signal 相互作用／Retry 経路不変／shutdown-drain 順序。
2. HANDLER-1 確認： handler は origin field の read-only 参照のみ（Decision／World 書換なし）。
3. INV-X5-1 維持： residency は全 Publish 計数のまま（coord と二重計上しない・別物併存）。
4. R25 §5 準拠： coordinatorTakeCount_（Main）declaration=1／getter=1／writer=1
   （pop 側 origin==Main 分岐の1箇所）。recovery take の計数方針を明示。
5. Test-only： emitQDelta に coord 追加のみ（新規 logger／prefix family なし・pre／post／delta 運用）。
6. Build Gate： R19／R22 と同一条件（Release／OFF・harness target authoritative・
   MKL include-path 問題の分離記録を継承）。
7. 禁止の継続： retry／defer 個別計数・drop reason 分類・per-task linkage・P3-1-D・原因帰属は扱わない。
```

## 8. R26 Gate

```text
R26-A： ADOPTED（path A）
  最小 origin plumbing が source 上定義可能（§4.1 の bounded 5点・§4.2 の不変部・
  §4.4 の2値 taxonomy）。既存観測のみでの C-1／C-2 十分区別は不能（§5.4）。
  → 次は origin plumbing の実装 gate（§7 の条件付き）。本 Step では実装しない。
R26-B： 非該当（plumbing 範囲は bounded であり、別設計 gate を要する規模ではない。
  D105 再監査は実装 gate 内 checklist で処理可能）。
R26-C： 非該当（R25-B の source finding との矛盾なし。本監査で R25-B を覆す事実は発見されず、
  むしろ deferred 位相の整理（§2）で補強された）。
```

- F／R 実行・R26 measurement・P3-1-D・limiter／stale／crossfade 帰属なし。
- R25-B の禁止（§7 指示：coordinatorTakeCount_ 実装・origin field 実装・
  recoveryObligationId 変更・intent payload 変更・D105 契約変更・deferred 経路変更・
  各種 counter 追加）をすべて遵守。本 Step の source 差分は本ドキュメントのみ。
- source は R23 vehicle のまま保持する。revert なし。
- R4 境界・保留事項・P3-5 §7解釈制約・H-B 対象外を維持する。
